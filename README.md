# VeriSight Layer-4 OCR Module

Layer-4 microservice for OCR-based expiry-date verification and text forensics.

## Project Tree

```text
Layer_4_OCR/
|-- inference/
|   |-- ocr_preprocess.py
|   |-- ocr_verification.py
|   `-- __init__.py
|-- api/
|   |-- main.py
|   `-- router.py
|-- dataset/
|   |-- raw/
|   |   |-- images/
|   |   `-- labels/
|   `-- splits/
|       |-- train/
|       |-- val/
|       `-- test/
|-- requirements.txt
`-- README.md
```

## Dataset Location

Put Layer-4 dataset inside:

```text
Layer_4_OCR/dataset/
```

Recommended usage:

- `dataset/raw/images/` -> all original images (`.jpg`, `.png`, `.webp`)
- `dataset/raw/labels/` -> metadata CSV/JSON files (expiry GT, text GT, bbox if available)
- `dataset/splits/train/`, `dataset/splits/val/`, `dataset/splits/test/` -> split manifests (`.csv`) or symlinked split image lists

If you are using the custom VeriSight dataset, copy image files into `dataset/raw/images/` and keep a single label file in `dataset/raw/labels/labels.csv`.

## Features

- Text-region detection using YOLOv8 (optional)
- OCR extraction using EasyOCR (primary) with PaddleOCR fallback (optional)
- Multi-format date extraction (DD/MM/YY, DD.MM.YYYY, MON YYYY)
- Expiry plausibility checks against order/delivery metadata
- Text texture heuristics for digitally inserted date detection
- Neutral scoring fallback when OCR fails

## Install

```bash
pip install -r requirements.txt
```

## Run API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
```

## One-Command Run

```bash
chmod +x run_project.sh
./run_project.sh run
```

Other modes:

```bash
./run_project.sh install
./run_project.sh check
./run_project.sh finetune --epochs 30 --batch 16
./run_project.sh finetune-fast --epochs 50
```

## Fine-Tune with Dataset

Manifest mode (existing behavior):

- `dataset/raw/images/*.jpg|png|webp`
- `dataset/raw/labels/labels.csv`
- `dataset/splits/train/train_manifest.csv`
- `dataset/splits/val/val_manifest.csv`
- `dataset/splits/test/test_manifest.csv`

`labels.csv` should include at least:

- `image_name`
- `manipulated_region_bbox` (example: `"[142,88,210,112]"`)

Manifests should include:

- `image_name`
- `label_row_id` (row index or ID mapping to labels CSV)

Automatic fallback mode (no manifests required):

- `dataset/CASIA2.0_revised/Tp`
- `dataset/CASIA2.0_revised/Au`
- `dataset/CASIA2.0_Groundtruth` (mask PNGs with `_gt` suffix)
- optional: `dataset/MICC-F220/groundtruthDB_220.txt`

Fallback behavior:

- CASIA tampered images are paired with masks by filename stem and converted to YOLO bboxes.
- CASIA authentic images get empty labels.
- MICC entries marked tampered in `groundtruthDB_220.txt` use full-image positive boxes (dataset has class labels but no region masks).
- Splits are generated automatically with deterministic hashing.

Start fine-tuning:

```bash
./run_project.sh finetune --epochs 30 --batch 16 --imgsz 640
```

CUDA-only fast-training example:

```bash
./run_project.sh finetune --device 0 --epochs 50 --batch 24 --imgsz 640 --workers 0 --amp --cache ram --cos-lr --no-plots
```

Accelerated preset (recommended first):

```bash
./run_project.sh finetune-fast --device 0 --epochs 50
```

Notes:

- training exits with an error if CUDA is unavailable
- `--workers 0` auto-tunes workers for faster throughput
- `--batch auto` enables Ultralytics CUDA auto-batch sizing
- `--imgsz auto` selects image size from free GPU memory (default in `finetune-fast`)
- `--amp` and `--cache ram` keep GPU training fast
- `--fast-mode` reduces expensive augmentations for higher training throughput

Dry-run dataset conversion (no training):

```bash
./run_project.sh finetune --dry-run
```

## Endpoint

- `POST /api/v1/ocr-verify`

`multipart/form-data` fields:
- `image` (required)
- `order_date` (optional)
- `delivery_date` (optional)
- `mfg_date_claimed` (optional)

Response shape:

```json
{
  "ocr_score": 62.4,
  "flags": ["expiry_date_font_inconsistency"],
  "details": {
    "extracted_text": "EXP 15/08/2024"
  },
  "processing_time_ms": 180
}
```
