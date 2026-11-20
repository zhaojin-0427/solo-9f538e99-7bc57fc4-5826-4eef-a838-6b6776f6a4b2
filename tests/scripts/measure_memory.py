"""在独立子进程中测量制备/诊断的峰值内存与耗时。

用法::

    python -I tests/scripts/measure_memory.py prepare <输入目录> <输出目录> <工作目录> <饱和阈值> [--qc]
    python -I tests/scripts/measure_memory.py diagnose <成品路径> <x0> <y0> <x1> <y1>

脚本不导入测试夹具，只调用应用自身的入口，便于在 256 MiB 容器内复现。
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.pipeline import build_master_flat  # noqa: E402


def _peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _prepare(args: list[str]) -> int:
    input_dir, out_dir, work_dir, sat = (
        Path(args[0]),
        Path(args[1]),
        Path(args[2]),
        float(args[3]),
    )
    include_qc = "--qc" in args[4:]
    paths = sorted(input_dir.glob("*.fits"))
    started = time.perf_counter()
    product = build_master_flat(
        paths,
        [p.name for p in paths],
        sat,
        output_dir=out_dir,
        work_dir=work_dir,
        include_qc=include_qc,
    )
    elapsed = time.perf_counter() - started
    print(
        f"RESULT mode=prepare frames={len(paths)} qc={int(include_qc)} "
        f"elapsed={elapsed:.1f}s peak_mib={_peak_mib():.1f} "
        f"size_mib={product.stat().st_size / 2 ** 20:.1f} product={product}"
    )
    return 0


def _diagnose(args: list[str]) -> int:
    from app.diagnose import diagnose_product

    path = Path(args[0])
    x0, y0, x1, y1 = (int(value) for value in args[1:5])
    started = time.perf_counter()
    result = diagnose_product(path, x0, y0, x1, y1)
    elapsed = time.perf_counter() - started
    print(
        f"RESULT mode=diagnose elapsed={elapsed:.1f}s peak_mib={_peak_mib():.1f} "
        f"pixels={result.pixel_count} low_counts={result.low_count_pixels}"
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    mode, args = argv[1], argv[2:]
    if mode == "prepare":
        return _prepare(args)
    if mode == "diagnose":
        return _diagnose(args)
    print(f"未知模式：{mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
