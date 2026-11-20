"""主平场制备 FastAPI 接口。

* ``GET  /docs``     Swagger 交互文档，可直接上传 FITS。
* ``GET  /health``   存活探针。
* ``POST /prepare``  上传 3~12 张平场 FITS 与饱和阈值，返回成品 FITS；
  可勾选附带逐帧逐像元的质检记录。
* ``POST /diagnose`` 上传成品 FITS 与矩形像素坐标，核对参与帧数与 COUNTS
  并按上传序号统计排除原因。
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .diagnose import diagnose_product
from .errors import PreparationError
from .pipeline import MAX_FRAMES, MIN_FRAMES, build_master_flat

logger = logging.getLogger("master_flat")

app = FastAPI(
    title="天文测光主平场制备与诊断接口",
    version="1.1.0",
    description=(
        "上传 3~12 张已扣偏置的无压缩二维 16 位整数平场 FITS，"
        "按 BLANK / BSCALE / BZERO 处理后进行每帧中位数归一与像素级中位数聚合，"
        "输出含校正系数主图与参与帧数扩展图的成品 FITS；可选附带逐帧逐像元的"
        "质检记录，并在 /diagnose 中按矩形区域诊断成品。"
    ),
)

# 单容器内存预算 256 MiB，串行处理避免并发制备/诊断叠加内存。
_preparation_lock = threading.Lock()
_diagnosis_lock = threading.Lock()


def _data_dir(env_name: str, default: str) -> Path:
    path = Path(__file__).resolve().parent.parent / os.environ.get(env_name, default)
    path.mkdir(parents=True, exist_ok=True)
    return path


WORK_DIR = _data_dir("FLAT_WORK_DIR", "work")
OUTPUT_DIR = _data_dir("FLAT_OUTPUT_DIR", "products")
UPLOAD_CHUNK = 4 * 2 ** 20


@app.exception_handler(PreparationError)
async def preparation_error_handler(
    request: Request, exc: PreparationError
) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _save_upload(upload: UploadFile, target: Path) -> None:
    """把上传内容分块写入目标文件；空文件按整批失败处理。"""
    try:
        with target.open("wb") as out:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
    except OSError as exc:
        raise PreparationError(f"上传文件保存失败：{upload.filename}：{exc}") from exc
    finally:
        await upload.close()
    if target.stat().st_size == 0:
        raise PreparationError(f"上传文件为空：{upload.filename}")


@app.post("/prepare", response_class=FileResponse)
async def prepare_master_flat(
    files: list[UploadFile] = File(
        ...,
        description=f"平场 FITS 文件，数量 {MIN_FRAMES}~{MAX_FRAMES} 张",
    ),
    saturation: float = Form(
        ..., description="正且有限的饱和阈值（与 BSCALE/BZERO 换算后同单位）"
    ),
    include_qc: bool = Form(
        False,
        description=(
            "附带逐帧逐像元的质检记录扩展（QF01…，2 bit/像元："
            "0=参与，1=BLANK，2=换算后非正，3=饱和）"
        ),
    ),
) -> FileResponse:
    if not (MIN_FRAMES <= len(files) <= MAX_FRAMES):
        raise PreparationError(
            f"平场帧数须在 {MIN_FRAMES}~{MAX_FRAMES} 张之间，收到 {len(files)} 张"
        )

    batch_dir = Path(tempfile.mkdtemp(prefix="batch_", dir=WORK_DIR))
    frame_paths: list[Path] = []
    source_names: list[str] = []
    try:
        for index, upload in enumerate(files, start=1):
            if not upload.filename:
                raise PreparationError(f"第 {index} 个上传项缺少文件名")
            # 每个上传文件放在独立的新空目录中，隔离不可信数据。
            frame_dir = batch_dir / f"frame_{index:02d}"
            frame_dir.mkdir()
            safe_name = Path(upload.filename).name or f"frame_{index:02d}.fits"
            target = frame_dir / safe_name
            await _save_upload(upload, target)
            frame_paths.append(target)
            source_names.append(upload.filename)

        with _preparation_lock:
            product = build_master_flat(
                frame_paths=frame_paths,
                source_names=source_names,
                saturation=saturation,
                output_dir=OUTPUT_DIR,
                work_dir=WORK_DIR,
                include_qc=include_qc,
            )

        logger.info(
            "主平场制备成功：%s <- %d 帧（质检记录：%s）",
            product.name,
            len(files),
            "含" if include_qc else "不含",
        )
        return FileResponse(
            product,
            media_type="application/fits",
            filename=product.name,
            headers={
                "X-Frame-Count": str(len(files)),
                "X-QC-Included": "1" if include_qc else "0",
            },
        )
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)


@app.post("/diagnose")
async def diagnose_upload(
    file: UploadFile = File(
        ..., description="带质检记录的成品 FITS（/prepare 勾选附带质检记录后下载）"
    ),
    x0: int = Form(..., description="矩形左上角列坐标（从 0 起算，含）"),
    y0: int = Form(..., description="矩形左上角行坐标（从 0 起算，含）"),
    x1: int = Form(..., description="矩形右下角列坐标（不含，须 ≤ 宽度）"),
    y1: int = Form(..., description="矩形右下角行坐标（不含，须 ≤ 高度）"),
) -> JSONResponse:
    """诊断上传的成品：只读落盘文件，只处理与矩形相交的块。"""
    if not file.filename:
        raise PreparationError("上传项缺少文件名")
    batch_dir = Path(tempfile.mkdtemp(prefix="diag_", dir=WORK_DIR))
    try:
        safe_name = Path(file.filename).name or "product.fits"
        target = batch_dir / safe_name
        await _save_upload(file, target)
        with _diagnosis_lock:
            diagnosis = diagnose_product(target, x0, y0, x1, y1)
        return JSONResponse(content=diagnosis.as_dict())
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)
