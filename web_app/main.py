from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from inference import EfficientNetFoldEnsemble

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Skin Cancer Detection API")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_ensemble: EfficientNetFoldEnsemble | None = None


def get_ensemble() -> EfficientNetFoldEnsemble:
    global _ensemble
    if _ensemble is None:
        _ensemble = EfficientNetFoldEnsemble()
    return _ensemble


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    ens = get_ensemble()
    if not ens.checkpoint_paths():
        raise HTTPException(
            status_code=503,
            detail=(
                "No model checkpoints found. Train Phase 2 (Notebook 06) or set "
                "SKINCD_MODEL_DIR to a folder containing phase2_*_fold*.pt files."
            ),
        )

    try:
        p_mal, meta = ens.predict_malignant_prob(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    is_malignant = p_mal >= 0.5
    confidence = round(max(p_mal, 1.0 - p_mal), 4)

    return {
        "filename": file.filename,
        "prediction": "Malignant" if is_malignant else "Benign",
        "confidence": confidence,
        "raw_score": round(p_mal, 6),
        "model": "EfficientNet-B4-NS (Phase 2, K-fold logit average)",
        "branch": meta["branch"],
        "ensemble_folds": meta["ensemble_folds"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
