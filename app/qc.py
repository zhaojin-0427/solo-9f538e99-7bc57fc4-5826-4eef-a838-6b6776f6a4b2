"""质检记录（QC）的编码与体积预算。

成品可选附带逐帧逐像元的排除原因记录。每帧一幅 ``uint8`` 二维扩展
``QF01`` … ``QFnn``（按上传顺序编号），每个像元用 **2 bit** 保存：

* ``0`` 参与（有效像元）；
* ``1`` BLANK 缺测；
* ``2`` 按 BSCALE/BZERO 换算后非正（``<= 0``）；
* ``3`` 换算后达到或超过饱和阈值。

判定顺序固定为 BLANK → 非正 → 饱和，先命中者生效。每行 4 个像元打包成
1 字节，行内高位存右侧像元。最坏规模 12 帧 × 4096² 的记录数据为
12 × 4096 × 1024 = 48 MiB，连同各扩展头部仍在 49 MiB 预算内。
"""

from __future__ import annotations

import numpy as np

from .errors import PreparationError

QC_PREFIX = "QF"
# 2 bit 排除原因编码，按判定顺序排列。
QC_PARTICIPATING = 0
QC_BLANK = 1
QC_NONPOSITIVE = 2
QC_SATURATED = 3
# 每个 FITS 扩展头部最少占一个 2880 字节块。
QC_HEADER_BYTES = 2880


def qc_extension_name(index: int) -> str:
    """上传序号（从 1 起）对应的质检扩展名。"""
    return f"{QC_PREFIX}{index:02d}"


def packed_width(width: int) -> int:
    """每行 2 bit 打包后的字节数。"""
    return (width + 3) // 4


def classify_codes(
    scaled: np.ndarray,
    raw: np.ndarray,
    blank: int | None,
    saturation: float,
) -> np.ndarray:
    """按 BLANK → 非正 → 饱和的顺序给像元打排除原因编码。"""
    codes = np.zeros(scaled.shape, dtype=np.uint8)
    if blank is not None:
        codes[raw == blank] = QC_BLANK
    pending = codes == QC_PARTICIPATING
    codes[pending & (scaled <= 0.0)] = QC_NONPOSITIVE
    pending &= codes == QC_PARTICIPATING
    codes[pending & (scaled >= saturation)] = QC_SATURATED
    return codes


def pack_codes(codes: np.ndarray) -> np.ndarray:
    """把 (rows, width) 的 2 bit 编码行打包为 (rows, packed_width) 字节。"""
    rows, width = codes.shape
    width_bytes = packed_width(width)
    if width % 4:
        codes = np.pad(codes, ((0, 0), (0, width_bytes * 4 - width)))
    quad = codes.reshape(rows, width_bytes, 4).astype(np.uint8)
    return (
        quad[:, :, 0]
        | (quad[:, :, 1] << 2)
        | (quad[:, :, 2] << 4)
        | (quad[:, :, 3] << 6)
    )


def unpack_codes(packed: np.ndarray) -> np.ndarray:
    """把 (rows, width_bytes) 字节解包为 (rows, width_bytes*4) 编码。"""
    rows, width_bytes = packed.shape
    codes = np.empty((rows, width_bytes * 4), dtype=np.uint8)
    codes[:, 0::4] = packed & 0b00000011
    codes[:, 1::4] = (packed >> 2) & 0b00000011
    codes[:, 2::4] = (packed >> 4) & 0b00000011
    codes[:, 3::4] = (packed >> 6) & 0b00000011
    return codes


def qc_footprint_bytes(n_frames: int, height: int, width: int) -> int:
    """质检记录连同其扩展头部的总字节数（含 FITS 2880 字节块补齐）。"""
    block = 2880
    row_bytes = packed_width(width)
    data_bytes = height * row_bytes
    padded = data_bytes + (-data_bytes) % block
    return n_frames * (QC_HEADER_BYTES + padded)


def ensure_qc_budget(n_frames: int, height: int, width: int, limit: int) -> None:
    footprint = qc_footprint_bytes(n_frames, height, width)
    if footprint > limit:
        raise PreparationError(
            f"质检记录需 {footprint} 字节，超过 {limit} 字节上限，"
            "请缩小帧数或图像尺寸",
            status_code=413,
        )



def sample_codes(
    packed: np.ndarray,
    width: int,
    max_rows: int = 64,
    max_cols: int = 64,
) -> np.ndarray:
    """从回读的打包记录中抽样取出编码矩阵，用于成品自检。

    只解包稀疏选取的行/列，避免为 4096² 记录再分配整幅解码数组。
    """
    rows, _ = packed.shape
    if rows == 0 or width == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    row_idx = np.unique(np.linspace(0, rows - 1, min(rows, max_rows)).astype(int))
    col_idx = np.unique(np.linspace(0, width - 1, min(width, max_cols)).astype(int))
    sample = np.empty((row_idx.size, col_idx.size), dtype=np.uint8)
    for r_pos, row in enumerate(row_idx):
        byte_row = packed[row]
        sample[r_pos] = (byte_row[col_idx // 4] >> ((col_idx % 4) * 2)) & 0b11
    return sample
