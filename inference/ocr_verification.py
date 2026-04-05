from __future__ import annotations

import importlib
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from inference.ocr_preprocess import BBox, OCRPreprocessor


def _optional_import(module_name: str, attr_name: Optional[str] = None):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    if attr_name is None:
        return module
    return getattr(module, attr_name, None)


easyocr = _optional_import("easyocr")
PaddleOCR = _optional_import("paddleocr", "PaddleOCR")
YOLO = _optional_import("ultralytics", "YOLO")

MONTH_MAP: Dict[str, int] = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "SEPT": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass(frozen=True)
class OCRTextCandidate:
    text: str
    confidence: float
    bbox: Optional[BBox]
    source: str


class OCRVerificationModule:
    def __init__(
        self,
        text_detector_model: str = "yolov8n.pt",
        languages: Sequence[str] = ("en",),
        use_gpu: bool = False,
        enable_yolo: bool = True,
        enable_paddle_fallback: bool = True,
    ) -> None:
        self._preprocessor = OCRPreprocessor()
        self._date_formats = (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d.%m.%Y",
            "%d-%m-%y",
            "%d/%m/%y",
            "%d.%m.%y",
            "%m-%Y",
            "%m/%Y",
            "%m.%Y",
        )

        self._yolo_model = None
        if enable_yolo and YOLO is not None:
            try:
                self._yolo_model = YOLO(text_detector_model)
            except Exception:
                self._yolo_model = None

        self._easy_reader = None
        if easyocr is not None:
            try:
                self._easy_reader = easyocr.Reader(list(languages), gpu=use_gpu, verbose=False)
            except TypeError:
                self._easy_reader = easyocr.Reader(list(languages), gpu=use_gpu)
            except Exception:
                self._easy_reader = None

        self._paddle_reader = None
        if enable_paddle_fallback and PaddleOCR is not None:
            try:
                self._paddle_reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except Exception:
                self._paddle_reader = None

    def analyze(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata = metadata or {}
        started = time.time()

        image = cv2.imread(image_path)
        if image is None:
            return {
                "score": 50.0,
                "flags": ["invalid_image", "ocr_failed_neutral_score"],
                "details": {
                    "message": "Image could not be loaded",
                    "image_path": image_path,
                },
            }

        preprocessed = self._preprocessor.preprocess(image)
        yolo_boxes = self._detect_text_regions(preprocessed.original)

        detector_mode = "yolov8" if yolo_boxes else "full_image_fallback"
        if not yolo_boxes:
            h, w = image.shape[:2]
            yolo_boxes = [(0, 0, w, h)]

        crops = self._preprocessor.crop_regions(preprocessed.original, yolo_boxes)
        candidates = self._recognize_text(crops)

        extracted_text = " ".join(candidate.text for candidate in candidates).strip()
        avg_confidence = (
            float(np.mean([candidate.confidence for candidate in candidates])) if candidates else 0.0
        )

        date_candidates = self._extract_dates_from_text(extracted_text)
        selected_expiry = self._select_expiry_date(date_candidates, metadata)

        plausibility_score, plausibility_flags, plausibility_details = self._plausibility_analysis(
            selected_expiry,
            metadata,
            has_extracted_text=bool(extracted_text),
        )

        texture_score, texture_flags, texture_details = self._text_texture_forensics(crops)

        flags: List[str] = []
        flags.extend(plausibility_flags)
        flags.extend(texture_flags)

        if not candidates:
            flags.append("ocr_failed_neutral_score")
            score = 50.0
        else:
            confidence_score = float(np.clip(avg_confidence * 100.0, 0.0, 100.0))
            score = (
                0.45 * confidence_score
                + 0.35 * plausibility_score
                + 0.20 * texture_score
            )
            score = float(np.round(np.clip(score, 0.0, 100.0), 2))

        elapsed_ms = int((time.time() - started) * 1000)

        return {
            "score": score,
            "flags": sorted(set(flags)),
            "details": {
                "detector_mode": detector_mode,
                "extracted_text": extracted_text,
                "ocr_confidence": float(np.round(avg_confidence, 4)),
                "date_candidates": [d.isoformat() for d in date_candidates],
                "selected_expiry_date": selected_expiry.isoformat() if selected_expiry else None,
                "plausibility": plausibility_details,
                "texture_forensics": texture_details,
                "detected_regions": len(crops),
                "engines": {
                    "easyocr": self._easy_reader is not None,
                    "paddleocr": self._paddle_reader is not None,
                    "yolov8": self._yolo_model is not None,
                },
                "processing_time_ms": elapsed_ms,
            },
        }

    def _detect_text_regions(self, image: np.ndarray) -> List[BBox]:
        if self._yolo_model is None:
            return []

        try:
            results = self._yolo_model.predict(source=image, conf=0.15, verbose=False)
        except Exception:
            return []

        height, width = image.shape[:2]
        boxes: List[BBox] = []

        for result in results:
            result_boxes = getattr(result, "boxes", None)
            if result_boxes is None:
                continue
            xyxy = getattr(result_boxes, "xyxy", None)
            if xyxy is None:
                continue

            array = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.array(xyxy)
            for row in array:
                if len(row) < 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in row[:4]]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))

        return self._deduplicate_boxes(boxes)

    def _deduplicate_boxes(self, boxes: Sequence[BBox], iou_threshold: float = 0.65) -> List[BBox]:
        if not boxes:
            return []

        def area(box: BBox) -> int:
            return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

        sorted_boxes = sorted(boxes, key=area, reverse=True)
        kept: List[BBox] = []

        for box in sorted_boxes:
            if all(self._iou(box, existing) < iou_threshold for existing in kept):
                kept.append(box)

        return kept

    def _iou(self, a: BBox, b: BBox) -> float:
        x_left = max(a[0], b[0])
        y_top = max(a[1], b[1])
        x_right = min(a[2], b[2])
        y_bottom = min(a[3], b[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = float((x_right - x_left) * (y_bottom - y_top))
        area_a = float((a[2] - a[0]) * (a[3] - a[1]))
        area_b = float((b[2] - b[0]) * (b[3] - b[1]))
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _recognize_text(self, crops: Sequence[Tuple[BBox, np.ndarray]]) -> List[OCRTextCandidate]:
        all_candidates: List[OCRTextCandidate] = []

        for bbox, crop in crops:
            region_candidates: List[OCRTextCandidate] = []
            if self._easy_reader is not None:
                region_candidates = self._run_easyocr(crop, bbox)
            if not region_candidates and self._paddle_reader is not None:
                region_candidates = self._run_paddleocr(crop, bbox)
            all_candidates.extend(region_candidates)

        return all_candidates

    def _run_easyocr(self, crop: np.ndarray, bbox: BBox) -> List[OCRTextCandidate]:
        if self._easy_reader is None:
            return []

        try:
            output = self._easy_reader.readtext(crop)
        except Exception:
            return []

        candidates: List[OCRTextCandidate] = []
        for item in output:
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            if not text:
                continue
            confidence = float(item[2])
            confidence = float(np.clip(confidence, 0.0, 1.0))
            candidates.append(
                OCRTextCandidate(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    source="easyocr",
                )
            )

        return candidates

    def _run_paddleocr(self, crop: np.ndarray, bbox: BBox) -> List[OCRTextCandidate]:
        if self._paddle_reader is None:
            return []

        try:
            output = self._paddle_reader.ocr(crop, cls=True)
        except Exception:
            return []

        candidates: List[OCRTextCandidate] = []
        for line in output or []:
            if not line:
                continue
            for item in line:
                if len(item) < 2:
                    continue
                result = item[1]
                if not result or len(result) < 2:
                    continue
                text = str(result[0]).strip()
                if not text:
                    continue
                confidence = float(np.clip(float(result[1]), 0.0, 1.0))
                candidates.append(
                    OCRTextCandidate(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        source="paddleocr",
                    )
                )

        return candidates

    def _extract_dates_from_text(self, text: str) -> List[date]:
        if not text:
            return []

        normalized = " ".join(text.upper().split())
        found_dates: List[date] = []

        for match in re.finditer(r"\b(\d{1,2})[\/.\-](\d{1,2})[\/.\-](\d{2,4})\b", normalized):
            day = int(match.group(1))
            month = int(match.group(2))
            year = self._normalize_year(int(match.group(3)))
            parsed = self._safe_date(year, month, day)
            if parsed is not None:
                found_dates.append(parsed)

        for match in re.finditer(r"\b(\d{1,2})[\/.\-](\d{4})\b", normalized):
            month = int(match.group(1))
            year = int(match.group(2))
            parsed = self._safe_date(year, month, 1)
            if parsed is not None:
                found_dates.append(parsed)

        for match in re.finditer(
            r"\b(?:(\d{1,2})\s+)?(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*[\s\-\/,\.]*([0-9]{2,4})\b",
            normalized,
        ):
            day = int(match.group(1)) if match.group(1) else 1
            month_token = match.group(2)
            month = MONTH_MAP[month_token]
            year = self._normalize_year(int(match.group(3)))
            parsed = self._safe_date(year, month, day)
            if parsed is not None:
                found_dates.append(parsed)

        unique_dates = sorted(set(found_dates))
        return unique_dates

    def _normalize_year(self, year: int) -> int:
        if year >= 100:
            return year
        if year <= 69:
            return 2000 + year
        return 1900 + year

    def _safe_date(self, year: int, month: int, day: int) -> Optional[date]:
        try:
            return date(year, month, day)
        except ValueError:
            return None

    def _select_expiry_date(self, candidates: Sequence[date], metadata: Dict[str, Any]) -> Optional[date]:
        if not candidates:
            return None

        delivery_date = self._parse_input_date(metadata.get("delivery_date"))
        order_date = self._parse_input_date(metadata.get("order_date"))
        reference = delivery_date or order_date

        if reference is None:
            return max(candidates)

        plausible = [d for d in candidates if d >= reference - timedelta(days=365)]
        if plausible:
            return max(plausible)

        return max(candidates)

    def _plausibility_analysis(
        self,
        expiry_date: Optional[date],
        metadata: Dict[str, Any],
        has_extracted_text: bool,
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        flags: List[str] = []
        score = 75.0 if has_extracted_text else 60.0

        order_date = self._parse_input_date(metadata.get("order_date"))
        delivery_date = self._parse_input_date(metadata.get("delivery_date"))
        mfg_date = self._parse_input_date(
            metadata.get("mfg_date_claimed")
            or metadata.get("mfg_date")
            or metadata.get("manufacturing_date")
        )

        if expiry_date is None:
            flags.append("expiry_date_not_found")
            score -= 25

        days_past_delivery: Optional[int] = None
        if expiry_date is not None and delivery_date is not None:
            days_past_delivery = (delivery_date - expiry_date).days
            if days_past_delivery > 180:
                flags.append("date_in_past_relative_to_delivery")
                score -= 55
            elif days_past_delivery > 30:
                flags.append("expiry_near_or_before_delivery")
                score -= 20

        if expiry_date is not None and mfg_date is not None and expiry_date < mfg_date:
            flags.append("expiry_before_mfg_date")
            score -= 35

        if expiry_date is not None and order_date is not None:
            days_after_order = (expiry_date - order_date).days
            if days_after_order > 3650:
                flags.append("expiry_too_far_future")
                score -= 15

        score = float(np.round(np.clip(score, 0.0, 100.0), 2))

        details = {
            "plausibility_score": score,
            "order_date": order_date.isoformat() if order_date else None,
            "delivery_date": delivery_date.isoformat() if delivery_date else None,
            "mfg_date": mfg_date.isoformat() if mfg_date else None,
            "days_past_delivery": days_past_delivery,
        }

        return score, flags, details

    def _text_texture_forensics(
        self,
        crops: Sequence[Tuple[BBox, np.ndarray]],
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        if not crops:
            return (
                50.0,
                ["text_region_not_found"],
                {
                    "texture_score": 50.0,
                    "edge_density": None,
                    "intensity_std": None,
                    "laplacian_variance": None,
                    "gradient_strength": None,
                },
            )

        selected_bbox, selected_crop = max(crops, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))

        gray = cv2.cvtColor(selected_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)

        edge_density = float(np.mean(edges > 0))
        intensity_std = float(np.std(gray))
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_strength = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))

        suspicion = 0.0
        if edge_density > 0.22 and intensity_std < 26:
            suspicion += 35
        if intensity_std < 18:
            suspicion += 25
        if laplacian_variance < 15:
            suspicion += 20
        if gradient_strength < 18:
            suspicion += 20

        texture_score = float(np.round(np.clip(100.0 - suspicion, 0.0, 100.0), 2))

        flags: List[str] = []
        if texture_score < 45:
            flags.append("expiry_date_font_inconsistency")
        elif texture_score < 60:
            flags.append("text_texture_anomaly")

        details = {
            "texture_score": texture_score,
            "selected_bbox": list(selected_bbox),
            "edge_density": float(np.round(edge_density, 6)),
            "intensity_std": float(np.round(intensity_std, 4)),
            "laplacian_variance": float(np.round(laplacian_variance, 4)),
            "gradient_strength": float(np.round(gradient_strength, 4)),
        }

        return texture_score, flags, details

    def _parse_input_date(self, raw_value: Any) -> Optional[date]:
        if raw_value is None:
            return None
        if isinstance(raw_value, datetime):
            return raw_value.date()
        if isinstance(raw_value, date):
            return raw_value
        if not isinstance(raw_value, str):
            return None

        candidate = raw_value.strip()
        if not candidate:
            return None

        normalized = candidate.replace("T", " ").replace("Z", "")
        for date_format in self._date_formats:
            try:
                return datetime.strptime(normalized, date_format).date()
            except ValueError:
                continue

        try:
            return datetime.fromisoformat(candidate).date()
        except ValueError:
            return None
