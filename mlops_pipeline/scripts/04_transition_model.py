#!/usr/bin/env python3
# FastAPI service for Fruits+Freshness classifier (multiclass)
# - Load model from MLflow Model Registry alias "Production"
# - Accept image file (multipart/form-data)
# - Return top-1 + top-k probabilities

import io
import json
import os

import mlflow.pyfunc
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ======= CONFIG (ปรับได้) ======= #
MODEL_NAME = "fruits-freshness-classifier"
MODEL_URI = f"models:/{MODEL_NAME}@Production"
IMG_SIZE = (160, 160)
CLASSES_JSON = "processed_metadata/class_indices.json"
TOPK = 5

# ======= APP ======= #
app = FastAPI(title="Fruits Freshness API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= LOAD MODEL & CLASSES ======= #
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Cannot load model from {MODEL_URI}: {e}")

if os.path.exists(CLASSES_JSON):
    with open(CLASSES_JSON, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
else:
    n_classes = 10
    idx_to_class = {i: f"class_{i}" for i in range(n_classes)}


# ======= UTILS ======= #
def preprocess_pil_image(file_bytes: bytes) -> np.ndarray:
    """convert uploaded bytes -> model-ready numpy array shape (1,H,W,3) scaled 0..1"""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    img = img.resize(IMG_SIZE)
    arr = (np.asarray(img, dtype="float32") / 255.0)[None, ...]
    return arr


def predict_array(x: np.ndarray) -> np.ndarray:
    """call pyfunc model.predict; return prob array (N, C)"""
    preds = model.predict(x)
    if not isinstance(preds, np.ndarray):
        try:
            preds = preds.to_numpy()
        except Exception:
            preds = np.array(preds)
    return preds


# ======= ENDPOINTS ======= #
@app.get("/ping")
def ping():
    return {"status": "ok", "model": MODEL_NAME, "alias": "Production"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    content = await file.read()
    x = preprocess_pil_image(content)
    probs = predict_array(x)
    if probs.ndim == 1:
        probs = probs[None, :]  # (C,) -> (1,C)

    p = probs[0]
    top1_idx = int(np.argmax(p))
    top1_label = idx_to_class.get(top1_idx, str(top1_idx))
    top1_conf = float(p[top1_idx])

    # ======== 🔍 OOD detection heuristic ======== #

    def softmax_entropy(prob):
        # ป้องกัน log(0)
        return float(-np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0))))

    entropy = softmax_entropy(p)
    top2_conf = float(np.partition(p, -2)[-2])
    margin = top1_conf - top2_conf

    THRESH_PROB = 0.99
    THRESH_ENT = 0.25
    THRESH_MAR = 0.001

    if (top1_conf < THRESH_PROB) or (entropy > THRESH_ENT) or (margin < THRESH_MAR):
        return {
            "filename": file.filename,
            "prediction": None,
            "ood": True,
            "message": "Unknown or unrelated image (not recognized as dataset class)",
            "signals": {
                "top1_conf": top1_conf,
                "entropy": entropy,
                "margin": margin
            }
        }
    # ============================================ #

    # top-k predictions
    topk_idx = np.argsort(p)[::-1][:TOPK]
    topk = [
        {"label": idx_to_class.get(int(i), str(int(i))), "prob": float(p[int(i)])}
        for i in topk_idx
    ]

    return {
        "filename": file.filename,
        "prediction": {"label": top1_label, "prob": top1_conf},
        "topk": topk,
        "ood": False
    }
