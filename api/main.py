import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from api.router import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="VeriSight Layer-4 OCR Service", version="1.0.0")
app.include_router(router)

# Setup static files and templates
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main OCR tester frontend"""
    html_file = TEMPLATES_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file), media_type="text/html")
    return {"error": "Frontend not found"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "layer4-ocr"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    LOGGER.exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
