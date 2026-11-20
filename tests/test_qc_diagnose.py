"""质检记录与成品诊断的测试：编码、体积预算与逐帧统计。"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from app.diagnose import (
    LOW_COUNT_THRESHOLD,
    diagnose_product,
    resolve_rectangle,
)
from app.errors import PreparationError
from app.pipeline import MAX_QC_BYTES, build_master_flat
from app.qc import (
    QC_BLANK,
    QC_NONPOSITIVE,
    QC_SATURATED,
    classify_codes,
    pack_codes,
    qc_footprint_bytes,
    unpack_codes,
)


def _prepare(tmp_path, paths, saturation=5000.0, include_qc=True):
    return build_master_flat(
        paths,
        [p.name for p in paths],
        saturation,
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        include_qc=include_qc,
    )


def _qc_codes(product, index: int, width: int) -> np.ndarray:
    with fits.open(product) as hdul:
        codes = unpack_codes(np.asarray(hdul[2 + index].data))
    return codes[:, :width]


@pytest.mark.parametrize("width", [1, 3, 4, 5, 17])
def test_pack_roundtrip(width):
    rng = np.random.default_rng(width)
    codes = rng.integers(0, 4, size=(7, width)).astype(np.uint8)
    packed = pack_codes(codes)
    assert packed.shape == (7, (width + 3) // 4)
    assert np.array_equal(unpack_codes(packed)[:, :width], codes)


def test_classify_order_blank_first():
    raw = np.array([[-32768, -32768, 0, 10, 5000, 5001]], dtype=np.int32)
    scaled = raw.astype(np.float64)
    codes = classify_codes(scaled, raw, blank=-32768, saturation=5000.0)
    # BLANK 优先于非正/饱和；其次是换算后非正；最后是饱和。
    assert codes.tolist() == [[QC_BLANK, QC_BLANK, QC_NONPOSITIVE, 0, QC_SATURATED, QC_SATURATED]]


def test_qc_budget_is_49_mib_for_worst_case():
    data_bytes = 4096 * (4096 // 4)
    padded = data_bytes + (-data_bytes) % 2880
    assert qc_footprint_bytes(12, 4096, 4096) == 12 * (padded + 2880)
    assert qc_footprint_bytes(12, 4096, 4096) <= MAX_QC_BYTES
    assert MAX_QC_BYTES == 49 * 1024 * 1024


def test_qc_extensions_follow_upload_order(tmp_path, batch_factory):
    paths, raws, saturation = batch_factory(n=4, blank=-32768)
    product = _prepare(tmp_path, paths, saturation)
    with fits.open(product) as hdul:
        assert [hdu.name for hdu in hdul] == [
            "PRIMARY",
            "COUNTS",
            "QF01",
            "QF02",
            "QF03",
            "QF04",
        ]
        for index, hdu in enumerate(hdul[2:], start=1):
            assert hdu.header["QCFRAME"] == index
            assert hdu.header["NAXIS1"] == (raws[0].shape[1] + 3) // 4
            assert hdu.header["NAXIS2"] == raws[0].shape[0]
            assert hdu.data.dtype == np.uint8


def test_qc_codes_match_pipeline_rules(tmp_path, batch_factory):
    blank = -32768
    paths, raws, saturation = batch_factory(n=5, blank=blank)
    product = _prepare(tmp_path, paths, saturation)
    width = raws[0].shape[1]
    for index, raw in enumerate(raws):
        codes = _qc_codes(product, index, width)
        value = raw.astype(np.float64)
        expected = np.zeros(raw.shape, dtype=np.uint8)
        expected[raw == blank] = QC_BLANK
        open_mask = expected == 0
        expected[open_mask & (value <= 0.0)] = QC_NONPOSITIVE
        open_mask &= expected == 0
        expected[open_mask & (value >= saturation)] = QC_SATURATED
        assert np.array_equal(codes, expected), f"frame {index + 1}"


def test_qc_participation_matches_counts(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=5, blank=-32768)
    product = _prepare(tmp_path, paths, saturation)
    with fits.open(product) as hdul:
        counts = np.asarray(hdul[1].data)
        used = np.zeros(counts.shape, dtype=np.int64)
        for hdu in hdul[2:]:
            used += unpack_codes(np.asarray(hdu.data))[
                :, : counts.shape[1]
            ] == 0
        np.testing.assert_array_equal(used, counts)


def test_prepare_without_qc_keeps_two_hdus(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    product = _prepare(tmp_path, paths, saturation, include_qc=False)
    with fits.open(product) as hdul:
        assert len(hdul) == 2
        assert hdul[0].header["HASQC"] is False
        assert hdul[1].name == "COUNTS"


def _diagnose(product, x0, y0, x1, y1):
    return diagnose_product(product, x0, y0, x1, y1)


def _reference_region(raws, saturation, blank, rect):
    """按矩形全量实现：先核对参与帧数，再只统计 COUNTS < 3 的像元。"""
    x0, y0, x1, y1 = rect
    counts = np.zeros((y1 - y0, x1 - x0), dtype=np.int64)
    codes = []
    for raw in raws:
        block = raw[y0:y1, x0:x1].astype(np.int64)
        value = block.astype(np.float64)
        code = np.zeros(block.shape, dtype=np.uint8)
        if blank is not None:
            code[block == blank] = QC_BLANK
        remaining = code == 0
        code[remaining & (value <= 0.0)] = QC_NONPOSITIVE
        remaining &= code == 0
        code[remaining & (value >= saturation)] = QC_SATURATED
        codes.append(code)
        counts += code == 0
    low = counts < LOW_COUNT_THRESHOLD
    return [
        {
            "blank": int(np.count_nonzero(code[low] == QC_BLANK)),
            "nonpositive": int(np.count_nonzero(code[low] == QC_NONPOSITIVE)),
            "saturated": int(np.count_nonzero(code[low] == QC_SATURATED)),
            "participating": int(np.count_nonzero(code[low] == 0)),
        }
        for code in codes
    ], int(np.count_nonzero(low))


def test_diagnose_matches_reference(tmp_path, batch_factory):
    blank = -32768
    paths, raws, saturation = batch_factory(n=5, blank=blank)
    product = _prepare(tmp_path, paths, saturation)
    rect = (0, 0, raws[0].shape[1], raws[0].shape[0])
    result = _diagnose(product, *rect)
    expected, low_pixels = _reference_region(raws, saturation, blank, rect)
    assert result.pixel_count == raws[0].size
    assert result.low_count_pixels == low_pixels
    assert result.frame_count == len(raws)
    for frame, want in zip(result.frames, expected):
        assert frame.as_dict()["blank"] == want["blank"]
        assert frame.nonpositive == want["nonpositive"]
        assert frame.saturated == want["saturated"]
        assert frame.participating == want["participating"]


def test_diagnose_sub_rectangle_and_bounds(tmp_path, batch_factory):
    paths, raws, saturation = batch_factory(n=4, blank=-32768)
    product = _prepare(tmp_path, paths, saturation)
    height, width = raws[0].shape
    rect = (3, 5, 13, 21)
    result = _diagnose(product, *rect)
    expected, low_pixels = _reference_region(raws, saturation, -32768, rect)
    assert result.rectangle == rect
    assert result.pixel_count == (13 - 3) * (21 - 5)
    assert result.low_count_pixels == low_pixels
    for frame, want in zip(result.frames, expected):
        assert frame.nonpositive == want["nonpositive"]

    # 右边与下边不计入：x1=width, y1=height 合法。
    edge = _diagnose(product, width - 2, height - 2, width, height)
    assert edge.pixel_count == 4
    # 越界/空矩形只报错。
    with pytest.raises(PreparationError, match="越界"):
        _diagnose(product, 0, 0, width + 1, height)
    with pytest.raises(PreparationError, match="矩形非法"):
        _diagnose(product, 5, 5, 5, 9)
    with pytest.raises(PreparationError, match="不能为负"):
        _diagnose(product, -1, 0, 5, 5)


def test_diagnose_requires_qc_records(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    product = _prepare(tmp_path, paths, saturation, include_qc=False)
    with pytest.raises(PreparationError, match="质检记录"):
        _diagnose(product, 0, 0, 4, 4)


def test_diagnose_detects_counts_mismatch(tmp_path, batch_factory):
    paths, raws, saturation = batch_factory(n=4, blank=-32768)
    product = _prepare(tmp_path, paths, saturation)
    with fits.open(product) as hdul:
        counts = np.asarray(hdul[1].data).copy()
        counts[2, 2] = (int(counts[2, 2]) + 1) % (len(paths) + 1)
        hdul[1].data = counts
        broken = tmp_path / "broken.fits"
        hdul.writeto(broken)
    with pytest.raises(PreparationError, match="计数不符"):
        _diagnose(broken, 0, 0, raws[0].shape[1], raws[0].shape[0])


def test_diagnose_detects_truncated_file(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    product = _prepare(tmp_path, paths, saturation)
    blob = product.read_bytes()
    truncated = tmp_path / "truncated.fits"
    truncated.write_bytes(blob[: len(blob) - 100])
    with pytest.raises(PreparationError, match="截断"):
        _diagnose(truncated, 0, 0, 4, 4)


def test_diagnose_rejects_non_product(tmp_path, batch_factory):
    paths, _, _ = batch_factory(n=3)
    with pytest.raises(PreparationError, match="主平场成品"):
        _diagnose(paths[0], 0, 0, 4, 4)


def test_diagnose_does_not_modify_upload(tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    product = _prepare(tmp_path, paths, saturation)
    before = (product.stat().st_size, product.stat().st_mtime_ns, product.read_bytes())
    _diagnose(product, 0, 0, 6, 6)
    after = (product.stat().st_size, product.stat().st_mtime_ns, product.read_bytes())
    assert before == after


def test_resolve_rectangle_rejects_empty():
    with pytest.raises(PreparationError):
        resolve_rectangle(4, 0, 4, 5, (10, 10))
    with pytest.raises(PreparationError):
        resolve_rectangle(0, 5, 5, 5, (10, 10))
    assert resolve_rectangle(0, 0, 10, 10, (10, 10)) == (0, 0, 10, 10)
