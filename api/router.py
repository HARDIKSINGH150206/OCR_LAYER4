import logging
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from inference.ocr_verification import OCRVerificationModule

LOGGER = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

_detector: OCRVerificationModule | None = None


def get_detector() -> OCRVerificationModule:
    global _detector
    if _detector is None:
        _detector = OCRVerificationModule()
    return _detector


@router.post("/api/v1/ocr-verify")
async def ocr_verify(
    image: UploadFile = File(...),
    order_date: str | None = Form(None),
    delivery_date: str | None = Form(None),
    mfg_date_claimed: str | None = Form(None),
):
    start = time.perf_counter()

    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a valid image.")

    suffix = Path(image.filename or "input.jpg").suffix or ".jpg"

    metadata = {
        "order_date": order_date,
        "delivery_date": delivery_date,
        "mfg_date_claimed": mfg_date_claimed,
    }
    metadata = {key: value for key, value in metadata.items() if value is not None and value != ""}

    try:
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = Path(temp_file.name)

        detector = get_detector()
        result = detector.analyze(str(temp_path), metadata=metadata)

        processing_time_ms = int((time.perf_counter() - start) * 1000)
        return {
            "ocr_score": result["score"],
            "flags": result["flags"],
            "details": result["details"],
            "processing_time_ms": processing_time_ms,
        }
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Layer 4 OCR inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        try:
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        except Exception:
            LOGGER.warning("Failed to cleanup temporary file", exc_info=True)
