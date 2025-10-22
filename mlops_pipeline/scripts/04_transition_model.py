#!/usr/bin/env python3
# FastAPI service for Fruits+Freshness classifier (multiclass)
# - Load model from MLflow Model Registry alias "Production"
# - Accept image file (multipart/form-data)
# - Return top-1 + top-k probabilities

import io, json, os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow.pyfunc

# ======= CONFIG (ปรับได้) ======= #
MODEL_NAME = "fruits-freshness-classifier"      # ให้ตรงกับตอน register
MODEL_URI  = f"models:/{MODEL_NAME}@Production" # ใช้ alias แทน stage
IMG_SIZE   = (224, 224)                         # ให้ตรงกับตอนเทรน
CLASSES_JSON = "processed_metadata/class_indices.json"  # ไฟล์ที่สร้างจากขั้น Preprocessing
TOPK = 5

# ======= APP ======= #
app = FastAPI(title="Fruits Freshness API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ======= LOAD MODEL & CLASSES ======= #
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Cannot load model from {MODEL_URI}: {e}")

if os.path.exists(CLASSES_JSON):
    with open(CLASSES_JSON, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    # สร้าง idx_to_class ตามลำดับ index
    idx_to_class = {v: k for k, v in class_to_idx.items()}
else:
    # ถ้าไม่มีไฟล์ mapping จะทำ label เป็น "class_0..N-1"
    # (แนะนำให้มีไฟล์นี้เพื่อชื่อคลาสสวย ๆ)
    try:
        # พยายามเดาจำนวนคลาสจากชั้นสุดท้ายของโมเดล (บางกรณี pyfunc ไม่บอกได้)
        # กำหนด fallback ไว้สัก 10 คลาส
        n_classes = 10
    except Exception:
        n_classes = 10
    idx_to_class = {i: f"class_{i}" for i in range(n_classes)}

# ======= UTILS ======= #
def preprocess_pil_image(file_bytes: bytes) -> np.ndarray:
    """convert uploaded bytes -> model-ready numpy array shape (1,H,W,3) scaled 0..1"""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    img = img.resize(IMG_SIZE)
    arr = (np.asarray(img, dtype="float32") / 255.0)[None, ...]  # (1,H,W,3)
    return arr

def predict_array(x: np.ndarray) -> np.ndarray:
    """call pyfunc model.predict; return prob array (N, C)"""
    preds = model.predict(x)  # pyfunc: รองรับ numpy ndarray สำหรับ TF/Keras
    # ถ้าได้ pandas DataFrame/Series กลับมา ให้แปลงเป็น np.array
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

    # top-k
    topk_idx = np.argsort(p)[::-1][:TOPK]
    topk = [
        {"label": idx_to_class.get(int(i), str(int(i))), "prob": float(p[int(i)])}
        for i in topk_idx
    ]

    return {
        "filename": file.filename,
        "prediction": {"label": top1_label, "prob": top1_conf},
        "topk": topk
    }
