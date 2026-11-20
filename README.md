# 天文测光主平场制备与诊断服务

基于 **FastAPI + Astropy** 的主平场（master flat）制备接口。上传 3～12 张已扣偏置的
平场 FITS，服务按统一规则聚合，输出可直接用于平场校正的成品 FITS；可勾选附带
逐帧逐像元的质检记录，并在 `/diagnose` 中按矩形区域诊断成品。

## 处理规则

1. **输入约束**
   - 帧数 3～12 张；每张恰好一幅**无压缩、二维、16 位整数**主图（`BITPIX=16`）。
   - 宽高均不超过 4096；同批帧尺寸必须一致。
   - 同批相机（`INSTRUME`）与滤镜（`FILTER`）标识必须一致（关键字缺失则不参与比对）。
   - 禁止压缩 HDU（`CompImageHDU`）与表 HDU。
2. **物理量换算**：保留原始整数读入，自行按 `BSCALE` / `BZERO` 换算
   （`value = raw * BSCALE + BZERO`），不依赖 Astropy 的自动缩放。
3. **有效像元（排除顺序固定）**
   1. 按 `BLANK` 关键字排除缺测像元；
   2. 排除换算后非正（`value <= 0`）的像元；
   3. 排除换算后达到或超过饱和阈值（`value >= saturation`）的像元；
   4. 其余记为参与。
   饱和阈值必须为正且有限。
4. **两阶段中位数聚合**
   - 每帧的有效像元除以**该帧有效像元全图中位数**做归一；
   - 对同一位置跨帧的有效归一值取**中位数**；
   - 某位置参与帧数少于 3 时输出 `NaN`。
5. **输出 FITS**
   - 主图（PRIMARY，`float32`）：校正系数；
   - 扩展图（`COUNTS`，`uint8`）：每个像元实际参与的帧数；
   - 勾选 `include_qc` 时按上传顺序追加质检记录扩展 `QF01`…`QFnn`；
   - 头部记录帧数、饱和阈值、各帧中位数、输入清单、`HASQC` 与处理历史。
6. **整批失败语义**：文件截断、格式不符、任一帧无有效像元、写出失败均整批报错，
   成品先写临时文件、回读校验通过后才原子改名，**只提供完整成品，不留半成品**。

## 质检记录（`include_qc`）

- 每帧一幅二维 `uint8` 图像扩展，扩展名 `QF01`…`QFnn`，编号即**上传序号**。
- 每个像元用 **2 bit** 保存排除原因：`0`=参与，`1`=BLANK，`2`=换算后非正，
  `3`=饱和；同一像元只记最先命中的原因。
- 每行 4 个像元打包为 1 字节（行内低位存左侧像元），因此
  `NAXIS1 = ceil(width/4)`、`NAXIS2 = height`。
- 扩展头记录 `QCFORMAT=2BIT-REASON`、`QCORDER=BLANK,NONPOSITIVE,SATURATED`、
  `QCFRAME`（上传序号）、`QCSOURCE`、`BSCALE`/`BZERO`/`QCBLANK`/`QCMEDIAN`、`QCSAT`。
- **体积预算**：质检记录连同新增头部最多占 **49 MiB**（按 1024² 计）。最坏规模
  12 帧 × 4096² 为 `12 × (4096×1024 + 2880)` ≈ 48.05 MiB；超限直接 `413`，
  不生成半成品。

## 成品诊断（`/diagnose`）

上传 `/prepare` 勾选了质检记录的成品 FITS 与矩形像素坐标 `x0, y0, x1, y1`：

- 坐标**从 0 起算**，矩形为 `[x0, x1) × [y0, y1)`，**右边和下边不计入**；
  任何越界或空矩形都只报错。
- 诊断**不需要原曝光**；上传落盘后以只读 + memmap 打开，全程只读取与矩形相交的块，
  并校验文件大小与修改时间未变。
- 先逐像元核对「质检记录中参与帧数」与 `COUNTS`，不一致即**计数不符**报错；
- 再**只统计矩形内 `COUNTS < 3` 的像元**，按上传序号给出每帧的
  `blank` / `nonpositive` / `saturated` 排除数与 `participating` 参与数。
- 记录缺失、文件截断、计数不符、矩形越界都只报错，不输出部分结果。

返回 JSON 示例：

```json
{
  "shape": [4096, 4096],
  "rectangle": [0, 0, 64, 64],
  "frame_count": 5,
  "pixel_count": 4096,
  "low_count_pixels": 12,
  "frames": [
    {"index": 1, "source": "flat_0.fits",
     "blank": 1, "nonpositive": 9, "saturated": 1, "participating": 1}
  ]
}
```

## 内存设计（256 MiB 容器预算）

最坏情形（12 帧 × 4096×4096）实测峰值：制备 **约 194 MiB**（含质检记录）、
诊断整幅 **约 114 MiB**：

- 第一趟用 65536 桶的原始值**直方图**精确求每帧中位数，不缓存整帧有效值；
- 原始帧通过 `HDU.section` **按行块读取**，不实体化整幅 32 MiB 整数数组；
- 第二趟按**自适应行带**聚合，行带高度由 `BAND_BUDGET_BYTES`（默认 8 MiB）反推，
  行主序连续读取避免跨行随机读；
- 帧轴中位数用**原地排序 + 索引选择**，避免 `np.nanmedian` 的整份行带复制与类型提升；
- 质检记录第三趟逐帧按行块打包落盘，不常驻；
- 输出分块做字节序转换后 `tofile` **流式直写** FITS 数据块，规避整幅大端副本；
- 服务串行处理（全局锁），避免并发制备/诊断叠加内存。

## 启动方式

### Docker Compose（推荐）

```bash
docker compose up --build
```

容器内存硬限制为 256 MiB（`mem_limit` / `memswap_limit`），成品与工作目录使用命名卷。
启动后访问：

- 交互文档（可直接上传，勾选 `include_qc`）：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

### 本地直接运行

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口用法

### `POST /prepare`（`multipart/form-data`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `files` | File[] | 3～12 张平场 FITS |
| `saturation` | number | 正且有限的饱和阈值（与换算后数据同单位） |
| `include_qc` | bool | 是否附带 `QF01`… 质检记录，默认 `false` |

成功返回 `application/fits` 成品附件（响应头 `X-Frame-Count`、`X-QC-Included`）；
失败返回 `422`（输入/数据问题）、`413`（质检记录超预算）或 `500`（写出问题），
响应体 `{"detail": "..."}`。未勾选时照常下载主图 + `COUNTS` 双图 FITS。

```bash
curl -X POST http://localhost:8000/prepare \
  -F "saturation=4000" \
  -F "include_qc=true" \
  -F "files=@flat_0.fits" \
  -F "files=@flat_1.fits" \
  -F "files=@flat_2.fits" \
  -o master_flat.fits
```

### `POST /diagnose`（`multipart/form-data`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `file` | File | 带质检记录的成品 FITS |
| `x0`, `y0` | int | 矩形左上像素坐标（从 0 起算，含） |
| `x1`, `y1` | int | 矩形右下像素坐标（不含，须 ≤ 宽/高） |

```bash
curl -X POST http://localhost:8000/diagnose \
  -F "x0=0" -F "y0=0" -F "x1=64" -F "y1=64" \
  -F "file=@master_flat.fits" | python -m json.tool
```

## 测试

```bash
pip install pytest
PYTHONPATH=. python -m pytest tests/ -q
```

- `tests/test_pipeline.py`、`tests/test_api.py`、`tests/test_qc_diagnose.py`：55 个
  单元/接口测试，含与 numpy 全量对照实现的数值比对、BLANK/BSCALE/BZERO、<3 帧 NaN、
  饱和/非正过滤、质检编码与体积预算、诊断统计与失败路径。
- `tests/scripts/measure_memory.py`：`prepare`/`diagnose` 独立子进程峰值内存与耗时。
- `tests/scripts/stress_memory.py`：12 帧 4096² 端到端数值压测。
- `tests/scripts/measure_pipeline_only.py`、`matrix_sizes.py`：纯管线与多尺寸峰值测量。

## 目录结构

```
app/
  main.py        # FastAPI 路由、上传隔离、串行锁
  pipeline.py    # 校验、直方图中位数、行带聚合、质检记录、低内存 FITS 写出
  qc.py          # 2 bit 排除原因编码/打包与 49 MiB 体积预算
  diagnose.py    # 只读矩形诊断：计数核对与逐帧排除统计
  errors.py      # 整批失败异常
tests/           # 单元/接口测试与压测脚本
Dockerfile
docker-compose.yml
requirements.txt
```
