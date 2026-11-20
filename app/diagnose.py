"""成品诊断：按矩形区域核对参与帧数与 COUNTS，并统计排除原因。

输入是 :mod:`app.pipeline` 产出的成品 FITS（主图 + ``COUNTS``，可选
``QF01``… 质检记录）。诊断不需要原始曝光；但需要成品携带质检记录，
否则无法得知逐帧的排除原因。

只读约定
========

* 上传落盘后以 ``mode="readonly"`` + ``memmap=True`` 打开，全程不改写
  文件；诊断过程只读取与矩形相交的块。
* 坐标从 0 起算，矩形为 ``[x0, x1) × [y0, y1)``，右边和下边不计入。
* 先逐像元核对「质检记录中参与帧数」与 ``COUNTS``；再只对矩形内
  ``COUNTS < 3`` 的像元，按上传序号统计每帧的三类排除数与参与数。
* 记录缺失、文件截断、计数不符、矩形越界都只报错，不输出部分结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from .errors import PreparationError
from .qc import QC_BLANK, QC_NONPOSITIVE, QC_SATURATED, qc_extension_name

COUNTS_NAME = "COUNTS"
# 只对 COUNTS 小于该值的像元统计排除原因。
LOW_COUNT_THRESHOLD = 3
# 每次读取的最大字节预算，保证 256 MiB 容器内逐块读取。
READ_BUDGET_BYTES = 4 * 2 ** 20


@dataclass(frozen=True)
class FrameDiagnosis:
    """单帧在上传序号下的排除数与参与数（仅 COUNTS < 3 的像元）。"""

    index: int
    source: str
    blank: int
    nonpositive: int
    saturated: int
    participating: int

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "source": self.source,
            "blank": self.blank,
            "nonpositive": self.nonpositive,
            "saturated": self.saturated,
            "participating": self.participating,
        }


@dataclass(frozen=True)
class Diagnosis:
    shape: tuple[int, int]
    rectangle: tuple[int, int, int, int]
    frame_count: int
    pixel_count: int
    low_count_pixels: int
    frames: list[FrameDiagnosis]

    def as_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "rectangle": list(self.rectangle),
            "frame_count": self.frame_count,
            "pixel_count": self.pixel_count,
            "low_count_pixels": self.low_count_pixels,
            "frames": [frame.as_dict() for frame in self.frames],
        }


def _open_product(path: Path) -> fits.HDUList:
    try:
        return fits.open(
            path, mode="readonly", memmap=True, do_not_scale_image_data=True
        )
    except (OSError, ValueError, fits.VerifyError) as exc:
        raise PreparationError(f"成品文件截断或无法解析：{exc}") from exc


def _readonly_snapshot(path: Path) -> tuple[int, int]:
    """记录文件大小与修改时间，用于确认诊断全程未改写上传文件。"""
    try:
        stat = path.stat()
    except OSError as exc:
        raise PreparationError(f"成品文件不可读：{exc}") from exc
    return stat.st_size, stat.st_mtime_ns


def _header_int(header: fits.Header, key: str, name: str) -> int:
    if key not in header:
        raise PreparationError(f"成品缺少 {key} 关键字（{name}）")
    try:
        value = int(header[key])
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"成品的 {key} 关键字非法：{exc}") from exc
    if value <= 0:
        raise PreparationError(f"成品的 {key}={value} 非法")
    return value


def _image_shape(hdu: fits.ImageHDU, name: str) -> tuple[int, int]:
    """只依据头部取二维图像尺寸（不触碰数据块）。"""
    if int(hdu.header.get("NAXIS", 0)) != 2:
        raise PreparationError(f"成品缺少二维 {name} 图像")
    try:
        height = int(hdu.header["NAXIS2"])
        width = int(hdu.header["NAXIS1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparationError(f"成品的 {name} 尺寸关键字非法：{exc}") from exc
    if height <= 0 or width <= 0:
        raise PreparationError(f"成品的 {name} 尺寸非法：{width}x{height}")
    return height, width


def _check_data_span(path: Path, hdul: fits.HDUList, position: int, label: str) -> None:
    """按文件长度确认该 HDU 的数据块完整，截断即报错。"""
    try:
        info = hdul[position].fileinfo()
        needed = int(info["datLoc"]) + int(info["datSpan"])
        actual = path.stat().st_size
    except (OSError, KeyError, TypeError) as exc:
        raise PreparationError(f"成品文件截断或无法解析：{exc}") from exc
    if actual < needed:
        raise PreparationError(
            f"成品文件截断：{label} 数据块需要 {needed} 字节，文件仅 {actual} 字节"
        )


def resolve_rectangle(
    x0: int, y0: int, x1: int, y1: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """校验矩形是否完整落在成品内，返回 ``(x0, y0, x1, y1)``。

    坐标从 0 起算，右/下边界不计入；任何越界（或空矩形）都只报错，
    不做裁剪。
    """
    height, width = shape
    if any(value < 0 for value in (x0, y0, x1, y1)):
        raise PreparationError("矩形坐标不能为负")
    if x1 <= x0 or y1 <= y0:
        raise PreparationError(
            f"矩形非法：须满足 x1>x0 且 y1>y0，收到 ({x0}, {y0})-({x1}, {y1})"
        )
    if x1 > width or y1 > height:
        raise PreparationError(
            f"矩形越界：成品尺寸 {width}x{height}，矩形右下 ({x1}, {y1}) 超界"
        )
    return x0, y0, x1, y1


def _row_chunks(
    x0: int, y0: int, x1: int, y1: int, n_frames: int
) -> list[tuple[int, int]]:
    """把矩形按行切块，使每块读取量控制在预算内。"""
    per_row = max((x1 - x0) * (n_frames + 1), 1)
    rows = max(1, READ_BUDGET_BYTES // per_row)
    rows = min(rows, y1 - y0)
    return [(r0, min(r0 + rows, y1)) for r0 in range(y0, y1, rows)]


def _unpack_block(packed: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """把矩形对应的打包字节解包为 (rows, x1-x0) 的 2 bit 编码。"""
    expanded = np.empty((packed.shape[0], packed.shape[1] * 4), dtype=np.uint8)
    expanded[:, 0::4] = packed & 0b00000011
    expanded[:, 1::4] = (packed >> 2) & 0b00000011
    expanded[:, 2::4] = (packed >> 4) & 0b00000011
    expanded[:, 3::4] = (packed >> 6) & 0b00000011
    offset = x0 % 4
    return expanded[:, offset : offset + (x1 - x0)]


def diagnose_product(path: Path, x0: int, y0: int, x1: int, y1: int) -> Diagnosis:
    """诊断成品：只读、只处理与矩形相交的块，返回逐帧统计。"""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    before = _readonly_snapshot(path)
    with _open_product(path) as hdul:
        if hdul[0].header.get("IMAGTYPE") != "MASTERFLAT":
            raise PreparationError("上传文件不是主平场成品（缺少 MASTERFLAT 标识）")
        if len(hdul) < 2:
            raise PreparationError("成品缺少 COUNTS 扩展")

        master_shape = _image_shape(hdul[0], "主图")
        counts_shape = _image_shape(hdul[1], COUNTS_NAME)
        if hdul[1].name != COUNTS_NAME or counts_shape != master_shape:
            raise PreparationError("成品的 COUNTS 扩展缺失或尺寸与主图不一致")

        n_frames = _header_int(hdul[0].header, "NFRAMES", "帧数")
        if n_frames > 12:
            raise PreparationError(f"成品的 NFRAMES={n_frames} 超出 1~12 范围")

        rectangle = resolve_rectangle(x0, y0, x1, y1, master_shape)
        rx0, ry0, rx1, ry1 = rectangle
        height, width = master_shape

        qc_hdus = hdul[2:]
        if len(qc_hdus) != n_frames:
            raise PreparationError(
                f"成品缺少质检记录：需要 {n_frames} 幅，收到 {len(qc_hdus)} 幅"
            )
        qc_sources: list[str] = []
        for index, hdu in enumerate(qc_hdus, start=1):
            if hdu.name != qc_extension_name(index):
                raise PreparationError(
                    f"质检记录编号不符：期望 {qc_extension_name(index)}，"
                    f"收到 {hdu.name}"
                )
            if int(hdu.header.get("QCFRAME", 0)) != index:
                raise PreparationError(f"质检记录 {hdu.name} 的上传序号不符")
            qc_shape = _image_shape(hdu, hdu.name)
            if qc_shape != (height, (width + 3) // 4):
                raise PreparationError(
                    f"质检记录 {hdu.name} 尺寸 {qc_shape[1]}x{qc_shape[0]} 与成品不符"
                )
            _check_data_span(path, hdul, index + 1, hdu.name)
            qc_sources.append(str(hdu.header.get("QCSOURCE", "")).strip())

        # 记录齐全后再核对数据块长度，缺失/截断分别报错。
        _check_data_span(path, hdul, 1, COUNTS_NAME)

        blank = np.zeros(n_frames, dtype=np.int64)
        nonpositive = np.zeros(n_frames, dtype=np.int64)
        saturated = np.zeros(n_frames, dtype=np.int64)
        participating = np.zeros(n_frames, dtype=np.int64)
        low_count_pixels = 0

        x_byte0, x_byte1 = rx0 // 4, (rx1 + 3) // 4
        for r0, r1 in _row_chunks(rx0, ry0, rx1, ry1, n_frames):
            try:
                counts_block = np.asarray(hdul[1].data[r0:r1, rx0:rx1])
            except (OSError, ValueError, EOFError, TypeError) as exc:
                raise PreparationError(f"成品数据块截断：COUNTS：{exc}") from exc
            if counts_block.dtype.kind not in "iu":
                raise PreparationError("成品的 COUNTS 扩展不是整数图像")
            if counts_block.size == 0:
                raise PreparationError("矩形未与成品像元相交")
            if int(counts_block.max()) > n_frames:
                raise PreparationError("成品的 COUNTS 超出帧数范围")

            used_frames = np.zeros(counts_block.shape, dtype=np.int64)
            codes_per_frame = []
            for index, hdu in enumerate(qc_hdus):
                try:
                    packed = np.asarray(hdu.data[r0:r1, x_byte0:x_byte1])
                except (OSError, ValueError, EOFError, TypeError) as exc:
                    raise PreparationError(
                        f"成品数据块截断：{hdu.name}：{exc}"
                    ) from exc
                codes = _unpack_block(packed, rx0, rx1)
                if codes.shape != counts_block.shape:
                    raise PreparationError(f"质检记录 {hdu.name} 与矩形不匹配")
                if int(codes.max()) > 3:
                    raise PreparationError(
                        f"质检记录 {hdu.name} 含非法编码（仅允许 0~3）"
                    )
                used = codes == 0
                used_frames += used
                codes_per_frame.append(codes)

            # 逐像元核对参与帧数与 COUNTS。
            if not np.array_equal(used_frames, counts_block.astype(np.int64)):
                raise PreparationError("计数不符：质检记录的参与帧数与 COUNTS 不一致")

            # 再只统计 COUNTS < 3 的像元。
            low_mask = counts_block < LOW_COUNT_THRESHOLD
            low_count = int(np.count_nonzero(low_mask))
            low_count_pixels += low_count
            if low_count == 0:
                continue
            for index, codes in enumerate(codes_per_frame):
                low_codes = codes[low_mask]
                blank[index] += int(np.count_nonzero(low_codes == QC_BLANK))
                nonpositive[index] += int(
                    np.count_nonzero(low_codes == QC_NONPOSITIVE)
                )
                saturated[index] += int(np.count_nonzero(low_codes == QC_SATURATED))
                participating[index] += int(np.count_nonzero(low_codes == 0))

        total_pixels = (rx1 - rx0) * (ry1 - ry0)
        frames = [
            FrameDiagnosis(
                index=index + 1,
                source=qc_sources[index],
                blank=int(blank[index]),
                nonpositive=int(nonpositive[index]),
                saturated=int(saturated[index]),
                participating=int(participating[index]),
            )
            for index in range(n_frames)
        ]
        counted = sum(
            frame.blank + frame.nonpositive + frame.saturated + frame.participating
            for frame in frames
        )
        if counted != n_frames * low_count_pixels:
            raise PreparationError("诊断统计与低计数像元总数不符")

        result = Diagnosis(
            shape=master_shape,
            rectangle=rectangle,
            frame_count=n_frames,
            pixel_count=total_pixels,
            low_count_pixels=low_count_pixels,
            frames=frames,
        )

    if _readonly_snapshot(path) != before:
        raise PreparationError("诊断过程意外改写了上传成品", status_code=500)
    return result
