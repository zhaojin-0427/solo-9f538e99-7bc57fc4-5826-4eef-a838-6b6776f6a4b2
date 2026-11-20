"""/prepare 与 /docs 接口测试。"""

from __future__ import annotations

import io

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from app import main
from conftest import make_flat_fits


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "products")
    (tmp_path / "work").mkdir()
    (tmp_path / "products").mkdir()
    with TestClient(main.app) as c:
        yield c


def _upload(client, paths, saturation=5000.0):
    files = [
        ("files", (p.name, p.read_bytes(), "application/fits")) for p in paths
    ]
    return client.post(
        "/prepare", data={"saturation": str(saturation)}, files=files
    )


def test_health_and_docs(client):
    assert client.get("/health").json() == {"status": "ok"}
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "主平场" in docs.text
    openapi = client.get("/openapi.json").json()
    assert "/prepare" in openapi["paths"]


def test_prepare_happy_path(client, tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=4)
    resp = _upload(client, paths, saturation)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/fits"
    assert resp.headers["x-frame-count"] == "4"
    with fits.open(io.BytesIO(resp.content)) as out:
        assert out[0].header["IMAGTYPE"] == "MASTERFLAT"
        assert out[0].header["SATURATE"] == saturation
        assert out[1].name == "COUNTS"
        assert out[1].data.shape == out[0].data.shape
    # 成品已落盘且临时上传目录被清理。
    products = list((tmp_path / "products").glob("master_flat_*.fits"))
    assert len(products) == 1
    assert not any((tmp_path / "work").glob("batch_*"))


def test_too_few_files_rejected(client, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    resp = _upload(client, paths[:2], saturation)
    assert resp.status_code == 422
    assert "帧数" in resp.json()["detail"]


def test_too_many_files_rejected(client, tmp_path):
    raw = np.ones((8, 8), dtype=np.int16) * 600
    paths = [make_flat_fits(tmp_path / f"m{i}.fits", raw) for i in range(13)]
    resp = _upload(client, paths)
    assert resp.status_code == 422
    assert "帧数" in resp.json()["detail"]


def test_bad_saturation_form_value(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in paths]
    resp = client.post(
        "/prepare", data={"saturation": "-3"}, files=files
    )
    assert resp.status_code == 422
    assert "饱和阈值" in resp.json()["detail"]


def test_truncated_upload_returns_422_and_no_product(client, tmp_path, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    blob = paths[1].read_bytes()
    files = [
        ("files", (p.name, p.read_bytes() if i != 1 else blob[:40], "application/fits"))
        for i, p in enumerate(paths)
    ]
    resp = client.post("/prepare", data={"saturation": str(saturation)}, files=files)
    assert resp.status_code == 422
    assert list((tmp_path / "products").iterdir()) == []


def test_all_dead_frames_returns_422(client, tmp_path):
    raw = np.full((12, 12), 30000, dtype=np.int16)
    paths = [make_flat_fits(tmp_path / f"d{i}.fits", raw) for i in range(3)]
    resp = _upload(client, paths, saturation=5000.0)
    assert resp.status_code == 422
    assert "有效像元" in resp.json()["detail"]


def test_missing_saturation_field_is_422(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in paths]
    resp = client.post("/prepare", files=files)
    assert resp.status_code == 422


def test_non_numeric_saturation_is_422(client, batch_factory):
    paths, _, _ = batch_factory(n=3)
    files = [("files", (p.name, p.read_bytes(), "application/fits")) for p in paths]
    resp = client.post("/prepare", data={"saturation": "abc"}, files=files)
    assert resp.status_code == 422


def _prepare_with_qc(client, paths, saturation=5000.0, include_qc=True):
    files = [
        ("files", (p.name, p.read_bytes(), "application/fits")) for p in paths
    ]
    return client.post(
        "/prepare",
        data={"saturation": str(saturation), "include_qc": "true" if include_qc else "false"},
        files=files,
    )


def test_prepare_include_qc_adds_records(client, batch_factory):
    paths, raws, saturation = batch_factory(n=4, blank=-32768)
    resp = _prepare_with_qc(client, paths, saturation)
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-qc-included"] == "1"
    with fits.open(io.BytesIO(resp.content)) as out:
        assert [hdu.name for hdu in out][:2] == ["PRIMARY", "COUNTS"]
        assert [hdu.name for hdu in out[2:]] == ["QF01", "QF02", "QF03", "QF04"]
        assert out[0].header["HASQC"] is True
    # 未勾选时不带质检记录。
    plain = _prepare_with_qc(client, paths, saturation, include_qc=False)
    assert plain.status_code == 200
    assert plain.headers["x-qc-included"] == "0"
    with fits.open(io.BytesIO(plain.content)) as out:
        assert len(out) == 2


def test_diagnose_returns_per_frame_stats(client, batch_factory):
    paths, raws, saturation = batch_factory(n=5, blank=-32768)
    prepared = _prepare_with_qc(client, paths, saturation)
    assert prepared.status_code == 200
    height, width = raws[0].shape
    resp = client.post(
        "/diagnose",
        data={"x0": "0", "y0": "0", "x1": str(width), "y1": str(height)},
        files={"file": ("product.fits", prepared.content, "application/fits")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["shape"] == [height, width]
    assert body["frame_count"] == 5
    assert body["pixel_count"] == height * width
    assert [frame["index"] for frame in body["frames"]] == [1, 2, 3, 4, 5]
    for frame in body["frames"]:
        assert set(frame) == {
            "index",
            "source",
            "blank",
            "nonpositive",
            "saturated",
            "participating",
        }


def test_diagnose_rectangle_bounds_errors(client, batch_factory):
    paths, raws, saturation = batch_factory(n=3, blank=-32768)
    prepared = _prepare_with_qc(client, paths, saturation)
    height, width = raws[0].shape
    cases = [
        {"x0": "0", "y0": "0", "x1": str(width + 1), "y1": str(height)},
        {"x0": "0", "y0": "0", "x1": str(width), "y1": str(height + 1)},
        {"x0": "-1", "y0": "0", "x1": "4", "y1": "4"},
        {"x0": "4", "y0": "4", "x1": "4", "y1": "9"},
    ]
    for data in cases:
        resp = client.post(
            "/diagnose",
            data=data,
            files={"file": ("product.fits", prepared.content, "application/fits")},
        )
        assert resp.status_code == 422, data
        assert "detail" in resp.json()


def test_diagnose_requires_qc_product(client, batch_factory):
    paths, _, saturation = batch_factory(n=3)
    plain = _prepare_with_qc(client, paths, saturation, include_qc=False)
    resp = client.post(
        "/diagnose",
        data={"x0": "0", "y0": "0", "x1": "4", "y1": "4"},
        files={"file": ("product.fits", plain.content, "application/fits")},
    )
    assert resp.status_code == 422
    assert "质检记录" in resp.json()["detail"]


def test_diagnose_truncated_upload_errors(client, batch_factory):
    paths, _, saturation = batch_factory(n=3, blank=-32768)
    prepared = _prepare_with_qc(client, paths, saturation)
    resp = client.post(
        "/diagnose",
        data={"x0": "0", "y0": "0", "x1": "4", "y1": "4"},
        files={
            "file": (
                "product.fits",
                prepared.content[: len(prepared.content) // 2],
                "application/fits",
            )
        },
    )
    assert resp.status_code == 422
    assert "截断" in resp.json()["detail"] or "质检记录" in resp.json()["detail"]
