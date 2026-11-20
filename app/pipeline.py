"""主平场（master flat）制备管线。

设计要点
========

* 输入为 3~12 张已扣偏置、无压缩、二维 16 位整数单图 FITS。
* 用 ``do_not_scale_image_data=True`` 读入原始整数，自行按
  ``BSCALE`` / ``BZERO`` 换算，显式按 ``BLANK`` 排除缺测像元。
* 有效像元：换算后 ``0 < value < saturation``（且不等于 BLANK）。
* 每帧有效值先除以该帧有效像元中位数；再对同一位置的有效归一值取中位数；
  参与帧数少于 3 的位置输出 NaN。
* 16 位整数只有 65536 个取值，第一趟用直方图精确求每帧中位数，
  避免一次性存放整帧有效值。
* 第二趟按行带聚合，行带高度依据图像尺寸自适应，把常驻内存控制在
  256 MiB 容器预算内；每个文件仅重读有限次数。
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

from .errors import PreparationError
from .qc import (
    ensure_qc_budget,
    pack_codes,
    packed_width,
    qc_extension_name,
    classify_codes,
    sample_codes,
)

MIN_FRAMES = 3
MAX_FRAMES = 12
MAX_DIM = 4096
MIN_FRAMES_PER_PIXEL = 3
# 质检记录连同新增头部允许的最大字节数（49 MiB，按 1024² 计）。
MAX_QC_BYTES = 49 * 1024 * 1024
# 行带聚合缓冲上限（float32）。取值需为排序/花式索引等瞬时分配、
# 常驻结果图与解释器基线留出余量，使总峰值稳在 256 MiB 容器预算内。
BAND_BUDGET_BYTES = 8 * 2 ** 20
# 单次行块的行数，限制换算时的临时数组大小。
ROW_CHUNK = 256
# 质检记录逐帧写出时的行块（更小，第三趟与主图/COUNTS 常驻并存）。
QC_ROW_CHUNK = 64
# 成品数据分块写出的字节预算，避免整幅大端副本抬高峰值。
WRITE_CHUNK_BYTES = 2 * 2 ** 20
RAW_BIN_COUNT = 65536  # int16 全部取值数
RAW_OFFSET = 32768
FITS_BLOCK = 2880  # FITS 头部与数据块的最小对齐单位


@dataclass(frozen=True)
class FrameInfo:
    """单帧的元数据与第一趟统计结果。"""

    path: Path
    source_name: str
    shape: tuple[int, int]
    image_index: int
    bscale: float
    bzero: float
    blank: int | None
    median: float


def _open_raw(path: Path) -> fits.HDUList:
    """打开 FITS 且禁止 astropy 自动换算，保留原始整数数据。"""
    try:
        return fits.open(path, do_not_scale_image_data=True, memmap=False)
    except (OSError, ValueError, fits.VerifyError) as exc:
        raise PreparationError(f"FITS 文件截断或无法解析：{path.name}：{exc}") from exc


def _check_data_span(path: Path, hdu: fits.ImageHDU) -> None:
    """按文件长度核对数据块是否完整，截断文件在读数据前即报错。"""
    try:
        info = hdu.fileinfo()
        needed = int(info["datLoc"]) + int(info["datSpan"])
        actual = path.stat().st_size
    except (OSError, KeyError, TypeError) as exc:
        raise PreparationError(f"FITS 文件截断或无法解析：{path.name}：{exc}") from exc
    if actual < needed:
        raise PreparationError(
            f"FITS 文件截断：{path.name} 实际 {actual} 字节，至少需要 {needed} 字节"
        )


@contextlib.contextmanager
def _frame_section(path: Path, image_index: int):
    """按需读取主图的行块视图；不实体化整幅数据。

    ``hdu.section`` 只在取用时读取对应行，4096² 帧的原始整数（32 MiB）
    不会整幅常驻；文件被截断时在读取处按整批失败处理。
    """
    try:
        with _open_raw(path) as hdul:
            hdu = hdul[image_index]
            _check_data_span(path, hdu)
            yield hdu.section
    except PreparationError:
        raise
    except (OSError, ValueError, EOFError, TypeError, fits.VerifyError) as exc:
        raise PreparationError(f"FITS 文件截断或数据损坏：{path.name}：{exc}") from exc


def _validate_hdus(hdul: fits.HDUList, name: str) -> tuple[int, fits.ImageHDU]:
    """校验每张图恰好包含一幅无压缩二维 16 位整数主图，返回其 HDU 序号。

    只依据头部与文件长度判断，不实体化整幅数据，避免 4096² 帧（32 MiB）
    在读入前再占一份内存；数据由后续按行块读取。
    """
    image_hdus: list[tuple[int, fits.ImageHDU]] = []
    for index, hdu in enumerate(hdul):
        if isinstance(hdu, fits.CompImageHDU):
            raise PreparationError(f"帧 {name} 含压缩图像 HDU，仅接受无压缩 FITS")
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            raise PreparationError(f"帧 {name} 含表 HDU，每张 FITS 只能有一幅主图")
        if hdu.is_image and int(hdu.header.get("NAXIS", 0)) > 0:
            image_hdus.append((index, hdu))

    if len(image_hdus) != 1:
        raise PreparationError(
            f"帧 {name} 含 {len(image_hdus)} 幅图像，要求每张 FITS 恰好一幅主图"
        )

    index, hdu = image_hdus[0]
    bitpix = hdu.header.get("BITPIX")
    if bitpix != 16:
        raise PreparationError(f"帧 {name} 的 BITPIX={bitpix}，要求为 16 位整数（BITPIX=16）")

    naxis = int(hdu.header.get("NAXIS", 0))
    if naxis != 2:
        raise PreparationError(f"帧 {name} 主图为 {naxis} 维，要求二维图像")
    try:
        height = int(hdu.header["NAXIS2"])
        width = int(hdu.header["NAXIS1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparationError(f"帧 {name} 的 NAXIS1/NAXIS2 关键字非法：{exc}") from exc

    if height == 0 or width == 0 or height > MAX_DIM or width > MAX_DIM:
        raise PreparationError(
            f"帧 {name} 尺寸 {width}x{height} 非法，宽高均须在 1~{MAX_DIM} 之间"
        )
    return index, hdu


def _camera_filter_keys(header: fits.Header) -> dict[str, str]:
    """提取相机/滤镜标识；缺失关键字不参与比对（不强制要求这些头存在）。"""
    result: dict[str, str] = {}
    for key in ("INSTRUME", "FILTER"):
        if key in header and header[key] is not None:
            value = str(header[key]).strip()
            if value:
                result[key] = value
    return result


def _read_scaling(header: fits.Header, name: str) -> tuple[float, float, int | None]:
    try:
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"帧 {name} 的 BSCALE/BZERO 关键字非法：{exc}") from exc
    if not (math.isfinite(bscale) and math.isfinite(bzero)) or bscale == 0.0:
        raise PreparationError(f"帧 {name} 的 BSCALE/BZERO 非有限值或 BSCALE 为 0")

    blank: int | None = None
    if "BLANK" in header:
        try:
            blank = int(header["BLANK"])
        except (TypeError, ValueError) as exc:
            raise PreparationError(f"帧 {name} 的 BLANK 关键字不是整数：{exc}") from exc
        if not (-32768 <= blank <= 32767):
            raise PreparationError(f"帧 {name} 的 BLANK={blank} 超出 16 位整数范围")
    return bscale, bzero, blank


def _frame_median(
    path: Path,
    image_index: int,
    bscale: float,
    bzero: float,
    blank: int | None,
    saturation: float,
) -> tuple[int, np.ndarray]:
    """第一趟：基于原始值直方图求有效像元中位数。

    返回 ``(有效像元总数, 65536 桶直方图)``，由调用方换算精确中位数，
    避免一次性存放整帧有效值。
    """
    hist = np.zeros(RAW_BIN_COUNT, dtype=np.int64)
    with _frame_section(path, image_index) as section:
        height = int(section.shape[0])
        for r0 in range(0, height, ROW_CHUNK):
            r1 = min(r0 + ROW_CHUNK, height)
            block = np.asarray(section[r0:r1]).astype(np.int32)
            scaled = block.astype(np.float64) * bscale + bzero
            valid = (scaled > 0.0) & (scaled < saturation)
            if blank is not None:
                valid &= block != blank
            if not valid.any():
                continue
            indices = block + RAW_OFFSET
            hist += np.bincount(indices[valid], minlength=RAW_BIN_COUNT)

    total = int(hist.sum())
    if total == 0:
        raise PreparationError(f"帧 {path.name} 不存在任何有效像元，整批制备失败")
    return total, hist


def _median_scaled_from_hist(
    hist: np.ndarray, total: int, bscale: float, bzero: float
) -> float:
    """由原始整数直方图精确换算有效像元（缩放后）的中位数。"""
    k1 = (total - 1) // 2
    k2 = total // 2
    ordered = hist if bscale > 0 else hist[::-1]
    cumulative = np.cumsum(ordered)
    p1 = int(np.searchsorted(cumulative, k1 + 1, side="left"))
    p2 = int(np.searchsorted(cumulative, k2 + 1, side="left"))
    if bscale > 0:
        raw1, raw2 = p1 - RAW_OFFSET, p2 - RAW_OFFSET
    else:
        raw1, raw2 = RAW_OFFSET - 1 - p1, RAW_OFFSET - 1 - p2
    return ((raw1 * bscale + bzero) + (raw2 * bscale + bzero)) / 2.0


def _band_height(n_frames: int, width: int) -> int:
    """根据内存预算计算行带高度（float32 聚合缓冲）。

    行带按行主序连续读取，避免逐列分块的跨行随机读；高度至少 1 行。
    """
    per_row = n_frames * width * np.dtype(np.float32).itemsize
    rows = BAND_BUDGET_BYTES // max(per_row, 1)
    return max(1, min(rows, MAX_DIM))


def _aggregate_buffer(
    buffer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对帧轴求有效归一值中位数与有效帧数。

    不用 ``np.nanmedian``：它会对整个行带再复制一份并做类型提升，
    在 256 MiB 预算下峰值超标。这里原地排序（NaN 按浮点规则排到末尾），
    再按每像素的有效计数用索引直接取中位数。
    """
    counts = np.isfinite(buffer).sum(axis=0, dtype=np.uint8)

    buffer.sort(axis=0)  # 原地，避免整份行带的复制与 float64 提升
    median_band = np.full(buffer.shape[1:], np.nan, dtype=np.float32)
    valid_pixels = counts > 0
    rows, cols = np.nonzero(valid_pixels)
    k1 = ((counts[valid_pixels] - 1) // 2).astype(np.int64)
    k2 = (counts[valid_pixels] // 2).astype(np.int64)
    # 偶数计数取中间两个均值；奇数计数两索引相同，等价直接取值。
    median_band[valid_pixels] = (
        buffer[k1, rows, cols].astype(np.float64)
        + buffer[k2, rows, cols].astype(np.float64)
    ) / 2.0
    median_band[counts < MIN_FRAMES_PER_PIXEL] = np.nan
    return median_band, counts


def _fill_band(
    frames: list[FrameInfo],
    row0: int,
    row1: int,
    saturation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """聚合一个行带：返回归一值中位数行带与参与帧数行带。"""
    band_h = row1 - row0
    width = frames[0].shape[1]
    buffer = np.full((len(frames), band_h, width), np.nan, dtype=np.float32)

    for f_index, frame in enumerate(frames):
        with _frame_section(frame.path, frame.image_index) as section:
            for b0 in range(row0, row1, ROW_CHUNK):
                b1 = min(b0 + ROW_CHUNK, row1)
                block = np.asarray(section[b0:b1]).astype(np.int32)
                scaled = block.astype(np.float64) * frame.bscale + frame.bzero
                valid = (scaled > 0.0) & (scaled < saturation)
                if frame.blank is not None:
                    valid &= block != frame.blank
                normalized = (scaled / frame.median).astype(np.float32)
                target = buffer[f_index, b0 - row0 : b1 - row0, :]
                np.copyto(target, normalized, where=valid)

    return _aggregate_buffer(buffer)


def _write_qc_records(
    frames: list[FrameInfo],
    saturation: float,
    qc_dir: Path,
    width: int,
) -> None:
    """第三趟：逐帧按行生成 2 bit 排除原因记录，行主序写入临时文件。

    每帧依次读入、逐行块打包落盘，常驻内存仅一个行块；记录布局与成品
    中 ``QFnn`` 扩展的 ``NAXIS1 = ceil(width/4)``、``NAXIS2 = height``
    完全一致。
    """
    for index, frame in enumerate(frames, start=1):
        destination = qc_dir / f"qc.{index:02d}"
        height = frame.shape[0]
        with destination.open("wb") as handle:
            with _frame_section(frame.path, frame.image_index) as section:
                for r0 in range(0, height, QC_ROW_CHUNK):
                    r1 = min(r0 + QC_ROW_CHUNK, height)
                    block = np.asarray(section[r0:r1]).astype(np.int32)
                    scaled = block.astype(np.float64) * frame.bscale + frame.bzero
                    codes = classify_codes(scaled, block, frame.blank, saturation)
                    handle.write(pack_codes(codes).tobytes())
        if destination.stat().st_size != height * packed_width(width):
            raise PreparationError("质检记录写出长度校验失败", status_code=500)


def _sanitize_ascii(text: str, limit: int = 60) -> str:
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text).strip()
    return cleaned[:limit] or "unknown"


def build_master_flat(
    frame_paths: list[Path],
    source_names: list[str],
    saturation: float,
    output_dir: Path,
    work_dir: Path,
    include_qc: bool = False,
) -> Path:
    """制备主平场，成功后返回成品 FITS 的最终路径，失败则整体报错。

    输出文件先写到 ``work_dir`` 内的临时文件，校验通过后原子移动到
    ``output_dir``；任何一步失败都不会留下半成品。

    ``include_qc=True`` 时成品额外携带按上传顺序编号的逐帧逐像元排除
    原因扩展（``QF01``…），总占用连同新增头部不超过 49 MiB。
    """
    if not (math.isfinite(saturation) and saturation > 0.0):
        raise PreparationError("饱和阈值必须为正且有限的数值")
    if not (MIN_FRAMES <= len(frame_paths) <= MAX_FRAMES):
        raise PreparationError(
            f"平场帧数须在 {MIN_FRAMES}~{MAX_FRAMES} 张之间，收到 {len(frame_paths)} 张"
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreparationError(f"工作目录不可用，整批制备失败：{exc}", status_code=500) from exc

    frames: list[FrameInfo] = []
    common_shape: tuple[int, int] | None = None
    common_camera_filter: dict[str, str] | None = None

    # 第一趟：结构校验 + 每帧有效值直方图与中位数。
    for path, source_name in zip(frame_paths, source_names):
        with _open_raw(path) as hdul:
            image_index, hdu = _validate_hdus(hdul, path.name)
            _check_data_span(path, hdu)
            shape = (
                int(hdu.header["NAXIS2"]),
                int(hdu.header["NAXIS1"]),
            )
            bscale, bzero, blank = _read_scaling(hdu.header, path.name)
            camera_filter = _camera_filter_keys(hdu.header)

        if common_shape is None:
            common_shape = shape
        elif shape != common_shape:
            raise PreparationError(
                f"帧 {path.name} 尺寸 {shape[1]}x{shape[0]} 与首帧 "
                f"{common_shape[1]}x{common_shape[0]} 不一致，同批尺寸须一致"
            )

        if common_camera_filter is None:
            common_camera_filter = camera_filter
        elif camera_filter != common_camera_filter:
            raise PreparationError(
                f"帧 {path.name} 的相机/滤镜标识 {camera_filter} 与首帧 "
                f"{common_camera_filter} 不一致，同批相机与滤镜须一致"
            )

        total, hist = _frame_median(
            path, image_index, bscale, bzero, blank, saturation
        )
        median = _median_scaled_from_hist(hist, total, bscale, bzero)
        if not math.isfinite(median) or median <= 0.0:
            raise PreparationError(f"帧 {path.name} 的有效中位数非法（{median}）")
        frames.append(
            FrameInfo(path, source_name, shape, image_index, bscale, bzero, blank, median)
        )

    assert common_shape is not None
    height, width = common_shape
    n_frames = len(frames)

    master = np.full((height, width), np.nan, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint8)

    qc_dir: Path | None = None
    if include_qc:
        ensure_qc_budget(n_frames, height, width, MAX_QC_BYTES)
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            qc_dir = Path(tempfile.mkdtemp(prefix="qc_", dir=work_dir))
        except OSError as exc:
            raise PreparationError(f"质检记录暂存失败：{exc}", status_code=500) from exc

    try:
        # 第二趟：自适应行带聚合，限制常驻内存。
        band_h = _band_height(n_frames, width)
        for row0 in range(0, height, band_h):
            row1 = min(row0 + band_h, height)
            median_band, count_band = _fill_band(
                frames, row0, row1, saturation
            )
            master[row0:row1, :] = median_band
            counts[row0:row1, :] = count_band

        # 可选的第三趟：逐帧生成 2 bit 排除原因记录。
        if qc_dir is not None:
            _write_qc_records(frames, saturation, qc_dir, width)

        return _write_output(
            frames,
            saturation,
            master,
            counts,
            output_dir,
            work_dir,
            qc_dir=qc_dir,
            qc_height=height,
            qc_width=width,
        )
    finally:
        if qc_dir is not None:
            shutil.rmtree(qc_dir, ignore_errors=True)


def _write_fits_low_memory(
    master: np.ndarray,
    counts: np.ndarray,
    header0: fits.Header,
    header1: fits.Header,
    tmp_path: Path,
    qc_dir: Path | None = None,
    qc_headers: list[fits.Header] | None = None,
) -> None:
    """流式写出双 HDU FITS，避免 ``HDUList.writeto`` 的整图内部缓冲。

    主图 float32（BITPIX=-32），扩展图 uint8（BITPIX=8，帧数最多 12）。
    头部仍由 astropy 生成以保证关键字合规；数据块用 ``ndarray.tofile``
    直接按大端字节追加，并补齐到 2880 字节块。

    附带质检记录时，按上传顺序把 ``qc_dir/qc.NN`` 中已按行打包的
    2 bit 编码流式追加为 ``QF01``… 图像扩展。
    """

    def header_block(header: fits.Header) -> bytes:
        # 直接序列化调用方基于真实数组建好的头部；切勿再用
        # PrimaryHDU(header=...) 重建，那会把 BITPIX/NAXIS 重置。
        # padding=True 保证输出为整 2880 字节块且 END 卡对齐卡边界。
        return header.tostring(sep="", padding=True).encode("ascii")

    def pad_to_block(handle, size: int) -> None:
        remainder = size % FITS_BLOCK
        if remainder:
            handle.write(b"\0" * (FITS_BLOCK - remainder))

    def write_array(handle, array: np.ndarray, dtype: str) -> None:
        # 分块做字节序转换后直写，避免为整幅成品再分配一份大端副本。
        rows = max(1, WRITE_CHUNK_BYTES // max(array.strides[0], 1))
        for r0 in range(0, array.shape[0], rows):
            r1 = min(r0 + rows, array.shape[0])
            np.ascontiguousarray(array[r0:r1], dtype=dtype).tofile(handle)
        pad_to_block(handle, array.nbytes)

    with tmp_path.open("wb") as handle:
        handle.write(header_block(header0))
        write_array(handle, master, ">f4")
        handle.write(header_block(header1))
        write_array(handle, counts, "u1")

        if qc_dir is not None and qc_headers is not None:
            for index, header in enumerate(qc_headers, start=1):
                source = qc_dir / f"qc.{index:02d}"
                handle.write(header_block(header))
                copied = 0
                with source.open("rb") as qc_handle:
                    while True:
                        chunk = qc_handle.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        copied += len(chunk)
                pad_to_block(handle, copied)
        handle.flush()
        os.fsync(handle.fileno())


def _qc_extension_headers(
    frames: list[FrameInfo],
    saturation: float,
    height: int,
    width: int,
) -> list[fits.Header]:
    """为每帧生成承载 2 bit 排除原因记录的图像扩展头部。"""
    width_bytes = packed_width(width)
    headers: list[fits.Header] = []
    for idx, frame in enumerate(frames, start=1):
        header = fits.Header()
        header["XTENSION"] = ("IMAGE", "Image extension")
        header["BITPIX"] = (8, "array data type")
        header["NAXIS"] = (2, "number of array dimensions")
        header["NAXIS1"] = (width_bytes, "packed bytes per row (4 pixels/byte)")
        header["NAXIS2"] = (height, "number of rows")
        header["PCOUNT"] = (0, "number of parameters")
        header["GCOUNT"] = (1, "number of groups")
        header["EXTNAME"] = (qc_extension_name(idx), "QC exclusion codes")
        header["QCFORMAT"] = ("2BIT-REASON", "2-bit per pixel: 0=used,1=blank,2=<=0,3=sat")
        header["QCORDER"] = ("BLANK,NONPOSITIVE,SATURATED", "Exclusion decision order")
        header["QCFRAME"] = (idx, "Upload index of the source frame")
        header["QCSOURCE"] = (_sanitize_ascii(frame.source_name), "Source file name")
        header["BSCALE"] = (frame.bscale, "Raw-to-value scale of the source frame")
        header["BZERO"] = (frame.bzero, "Raw-to-value offset of the source frame")
        if frame.blank is not None:
            header["QCBLANK"] = (frame.blank, "Raw BLANK value of the source frame")
        header["QCMEDIAN"] = (frame.median, "Valid-pixel median of the source frame")
        header["QCSAT"] = (saturation, "Saturation threshold in data units")
        headers.append(header)
    return headers


def _write_output(
    frames: list[FrameInfo],
    saturation: float,
    master: np.ndarray,
    counts: np.ndarray,
    output_dir: Path,
    work_dir: Path,
    qc_dir: Path | None = None,
    qc_height: int = 0,
    qc_width: int = 0,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    token = os.urandom(3).hex()
    final_name = f"master_flat_{timestamp}_{token}.fits"
    tmp_path = work_dir / f".{final_name}.tmp"
    final_path = output_dir / final_name

    primary_hdu = fits.PrimaryHDU(data=master)
    header = primary_hdu.header
    header["IMAGTYPE"] = ("MASTERFLAT", "Master flat field correction coefficients")
    header["NFRAMES"] = (len(frames), "Number of input flat frames")
    header["SATURATE"] = (saturation, "Saturation threshold in data units")
    header["MINCNT"] = (MIN_FRAMES_PER_PIXEL, "Minimum frames for a valid pixel")
    header["CREATED"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Creation time (UTC)",
    )
    header["HASQC"] = (bool(qc_dir), "Per-frame QC exclusion extensions present")
    for idx, frame in enumerate(frames):
        header[f"MED{idx + 1:02d}"] = (
            frame.median,
            f"Valid-pixel median of frame {idx + 1}",
        )
    header.add_history(
        "Pixels scaled by BSCALE/BZERO; BLANK, non-positive and saturated "
        "pixels excluded; per-frame median normalization followed by pixelwise "
        "median; pixels seen by fewer than 3 frames are NaN."
    )
    if qc_dir is not None:
        header.add_history(
            "Per-frame 2-bit exclusion records are stored in QF01..QFnn "
            "extensions in upload order: 0=used, 1=BLANK, 2=non-positive, "
            "3=saturated."
        )
    for idx, frame in enumerate(frames):
        header.add_history(f"Input {idx + 1:02d}: {_sanitize_ascii(frame.source_name)}")

    counts_hdu = fits.ImageHDU(data=counts, name="COUNTS")
    qc_headers = (
        _qc_extension_headers(frames, saturation, qc_height, qc_width)
        if qc_dir is not None
        else None
    )
    # 取头部副本后立即释放 HDU，避免 HDU 额外持有数据引用。
    header0, header1 = header.copy(), counts_hdu.header.copy()
    del primary_hdu, counts_hdu, header

    expected_len = 2 + (len(frames) if qc_dir is not None else 0)
    try:
        _write_fits_low_memory(
            master, counts, header0, header1, tmp_path,
            qc_dir=qc_dir, qc_headers=qc_headers,
        )
        _verify_products(tmp_path, master, counts, expected_len, qc_dir is not None)
        os.replace(tmp_path, final_path)
    except PreparationError:
        tmp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise PreparationError(f"主平场写出失败：{exc}", status_code=500) from exc

    return final_path


def _verify_products(
    path: Path,
    master: np.ndarray,
    counts: np.ndarray,
    expected_len: int,
    with_qc: bool,
) -> None:
    """回读校验成品：结构、尺寸、扩展名、数据块范围与质检记录长度。

    头部校验不实体化数据，避免额外内存；最后用 memmap 确认数据块可解析。
    """
    with fits.open(path, memmap=False) as verify:
        if len(verify) != expected_len:
            raise PreparationError("输出 FITS 结构校验失败", status_code=500)
        if verify[0].shape != master.shape or verify[1].shape != counts.shape:
            raise PreparationError("输出 FITS 尺寸校验失败", status_code=500)
        if verify[1].name != "COUNTS":
            raise PreparationError("输出 FITS 扩展名校验失败", status_code=500)
        if with_qc:
            for index, hdu in enumerate(verify[2:], start=1):
                if hdu.name != qc_extension_name(index):
                    raise PreparationError("输出质检扩展名校验失败", status_code=500)
                if hdu.data is None:
                    raise PreparationError("输出质检记录缺失", status_code=500)
    with fits.open(path, memmap=True) as verify:
        sample0 = np.asarray(verify[0].data)
        sample1 = np.asarray(verify[1].data)
        if sample0.shape != master.shape or sample1.shape != counts.shape:
            raise PreparationError("输出 FITS 数据块校验失败", status_code=500)
        # 帧数图取值必须落在 0~NFRAMES；确认数据块整体可读。
        if int(sample1.max()) > int(verify[0].header.get("NFRAMES", 0)):
            raise PreparationError("输出 FITS 数据块校验失败", status_code=500)
        if with_qc:
            for index, hdu in enumerate(verify[2:], start=1):
                data = hdu.data
                if data is None or data.dtype != np.uint8:
                    raise PreparationError("输出质检记录校验失败", status_code=500)
                if data.shape != (counts.shape[0], packed_width(counts.shape[1])):
                    raise PreparationError("输出质检记录尺寸校验失败", status_code=500)
                if int(sample_codes(data, counts.shape[1]).max()) > 3:
                    raise PreparationError("输出质检记录取值校验失败", status_code=500)
                if int(hdu.header.get("QCFRAME", 0)) != index:
                    raise PreparationError("输出质检记录编号校验失败", status_code=500)
