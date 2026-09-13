# interoc

Interoc measures interocular distance in images of animals, without being restricted to a fixed set of species.

## 專案介紹

Interoc 從一般照片中找出動物，量出每隻動物的**雙眼間距**，以及任兩隻動物之間的**右眼對右眼距離**。像素距離和公尺距離都會輸出。

Pipeline 分三個階段，每段都能單獨關掉。少了哪一段就少那部分結果，不會整條失敗：

| 階段 | 模型 | 產出 |
| --- | --- | --- |
| 1. 切割 | YOLO11-seg（`yolo11l-seg.pt`） | 動物的輪廓遮罩與 bbox（COCO 的 10 種動物） |
| 2. 眼睛 | ViTPose++（`usyd-community/vitpose-plus-base`） | 左右眼座標與信心分數，以及**像素**距離 |
| 3. 深度 | Depth Pro（`apple/DepthPro-hf`） | Metric 深度圖與焦距估計，以及**公尺**距離 |

- **不需要相機參數**：Depth Pro 會自己估焦距，所以沒有 EXIF 的圖片也算得出公尺數。已知真實焦距時，用 `--focal` 覆寫會更準。
- **不捏造眼睛**：分數低於 `--min-score` 的眼睛會標成虛擬眼（`observed=false`），不參與任何距離計算。虛擬眼不會被丟掉，會照樣寫進 JSON 與報告，讓你分辨「只有一顆眼睛」是側臉、出框，還是模型失效。
- **抓出左右眼重疊**：兩顆眼睛都過了門檻、卻落在幾乎同一個位置時，通常是模型把同一顆眼睛同時標成左眼和右眼。報告會把這種情況標出來。
- **像素距離一定會輸出**，可以拿來和公尺距離互相對照。

### 輸出

預設會寫入 `output/`：

```
output/
├── results.json                 完整結果（給程式讀）
├── report.md                    判讀報告（給人讀）
├── report_animals.csv           報告的「每隻動物」表
├── report_pairs.csv             報告的「任兩隻動物」表
├── <影像名>_1_masks.jpg          切割疊圖
├── <影像名>_2_eyes.jpg           眼睛疊圖
├── <影像名>_3_depth.jpg          深度圖
└── <影像名>_4_measure.jpg        距離疊圖
```

- `results.json`：每隻動物的 bbox、雙眼座標（含 `score`、`observed`、`depth_m`）、雙眼距離，以及頂層的跨物體距離 `inter_object`。`camera.focal_estimated` 記錄焦距的來源：`true` 代表焦距是模型估的，`false` 代表是使用者給的。
- `report.md`：和 `results.json` 同一份資料，但會替每筆數字下判斷。報告分兩張表：
  - **每隻動物**：位置、雙眼座標、是否為虛擬眼、雙眼距離。
  - **任兩隻動物**：右眼對右眼的距離（用 `--cross-eye` 可以改成左眼）。N 隻動物就列 N(N-1)/2 列，量不到的配對也會列出來並寫明原因。
- `report_animals.csv`、`report_pairs.csv`：上面兩張表的 CSV 版，方便在試算表裡篩選。檔案用 UTF-8 with BOM 編碼，Excel 直接開啟中文也不會亂碼。

加上 `--no-save` 時，以上檔案都不會寫出。用 `--json` 改變 JSON 路徑時，報告與 CSV 會寫在同一個目錄。

### 專案結構

```
main.py                 命令列入口
kernel/
├── core.py             pipeline 編排（InterocularCore.forward）
├── geometry.py         反投影與距離計算
├── report.py           產生 report.md 與 CSV
├── models/             segmentation / keypoint / depth 模型包裝
├── schemas/            資料結構
└── visualization/      疊圖繪製
testsample/             測試圖片（COCO）
```

## 本地運行

### 需求

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- （建議）NVIDIA GPU，驅動需支援 CUDA 13。沒有 GPU 也能在 CPU 上跑，只是比較慢。

### 步驟

```bash
# 1. 取得原始碼
git clone <repo-url> interoc
cd interoc

# 2. 安裝依賴（torch 會從 PyTorch 的 cu130 index 安裝）
uv sync

# 3. 跑 testsample/ 底下的全部圖片
uv run main.py
```

第一次執行會自動下載三份權重：YOLO、ViTPose 和 Depth Pro（約 1.9 GB）。

### 常用指令

```bash
uv run main.py path/to/cat.jpg          # 跑單張
uv run main.py imgs/ --limit 5          # 跑資料夾的前 5 張
uv run main.py --focal 1800             # 已知焦距（像素）時覆寫估計值
uv run main.py --no-depth               # 只要像素距離，不跑深度
uv run main.py --no-eyes                # 只切割（連帶不跑深度）
uv run main.py --no-save                # 只印在終端機，不寫檔
uv run main.py --outdir results/        # 換輸出目錄
uv run main.py --device cpu             # 指定裝置
uv run main.py --help                   # 所有參數
```

## Docker 部署

映像以 `ghcr.io/astral-sh/uv:python3.13-bookworm-slim` 為基底，照 `uv.lock` 安裝依賴。容器入口是 `python main.py`，所以 `docker run` 映像名稱後面的參數會直接傳給 `main.py`。

### 建置映像

```bash
docker build -t interoc .
```

### 執行

以下指令以 bash 為例：

```bash
# 跑映像內附的 testsample/，結果寫回主機的 ./output
docker run --rm --gpus all \
  -v "${PWD}/output:/app/output" \
  -v interoc-cache:/root/.cache \
  interoc

# 跑主機上自己的圖片資料夾
docker run --rm --gpus all \
  -v "${PWD}/images:/data:ro" \
  -v "${PWD}/output:/app/output" \
  -v interoc-cache:/root/.cache \
  interoc /data --limit 5

# 沒有 GPU：拿掉 --gpus all，並指定 CPU
docker run --rm \
  -v "${PWD}/output:/app/output" \
  -v interoc-cache:/root/.cache \
  interoc --device cpu
```

PowerShell 不支援用 `\` 換行，要改用反引號 `` ` ``。`${PWD}` 的寫法兩邊都能用：

```powershell
docker run --rm --gpus all `
  -v "${PWD}/output:/app/output" `
  -v interoc-cache:/root/.cache `
  interoc
```

- `--gpus all` 需要主機裝好 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。Windows 上則需要 Docker Desktop 搭配 WSL 2。
- `interoc-cache` 這個 named volume 會保存 Hugging Face 權重（ViTPose、Depth Pro），之後重跑容器就不必再下載。
- `yolo11l-seg.pt` 放在專案根目錄時，建置時會一併複製進映像。檔案不存在的話，ultralytics 會在執行時自動下載。
