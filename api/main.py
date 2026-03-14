import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.router import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="VeriSight Layer-4 OCR Service", version="1.0.0")
app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "layer4-ocr"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    LOGGER.exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
