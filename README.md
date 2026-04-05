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
