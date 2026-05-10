"""
PneumoScan AI — FastAPI backend.

Endpoint:
    POST /predict   (multipart/form-data, field: "file")
    -> { "prediction": "PNEUMONIA" | "NORMAL", "confidence": 0.0-1.0 }

Place your trained model file in this folder:
    - model.h5    (Keras / TensorFlow)   -> auto-loaded
    - model.pth   (PyTorch state_dict)   -> auto-loaded

Run:
    pip install -r requirements.txt
    python app.py
"""
from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

app = FastAPI(title="PneumoScan AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Model loading (lazy) --------
_tf_model = None
_torch_model = None
_torch = None


def _load_tf():
    """Load the Keras DenseNet201 model trained by train_model.py.

    Looks for (in order): best_model.keras, model.keras, model.h5
    Class indices match Keras flow_from_directory (alphabetical):
        0 -> NORMAL, 1 -> PNEUMONIA  (sigmoid output)
    """
    global _tf_model
    if _tf_model is not None:
        return _tf_model
    here = os.path.dirname(__file__)
    for name in ("best_model.keras", "model.keras", "model.h5"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            import tensorflow as tf  # type: ignore
            _tf_model = tf.keras.models.load_model(path)
            print(f"[PneumoScan] Loaded TensorFlow model: {path}")
            return _tf_model
    return None


def _load_torch():
    global _torch_model, _torch
    if _torch_model is not None:
        return _torch_model
    path = os.path.join(os.path.dirname(__file__), "model.pth")
    if not os.path.exists(path):
        return None
    import torch  # type: ignore
    _torch = torch
    # Adjust architecture below to match your trained model.
    from torchvision import models  # type: ignore
    model = models.densenet201(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _torch_model = model
    print(f"[PneumoScan] Loaded PyTorch model: {path}")
    return _torch_model


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def predict_image(image_bytes: bytes) -> tuple[str, float]:
    arr = preprocess(image_bytes)

    tf_model = _load_tf()
    if tf_model is not None:
        x = np.expand_dims(arr, 0)  # arr already resized to 224x224 and normalized /255
        prob = float(tf_model.predict(x, verbose=0).ravel()[0])
        label = CLASS_NAMES[1] if prob > 0.3 else CLASS_NAMES[0]
        confidence = prob if label == CLASS_NAMES[1] else 1 - prob
        return label, confidence

    torch_model = _load_torch()
    if torch_model is not None:
        import torch  # type: ignore
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = torch_model(x)
            prob = float(torch.sigmoid(logits).ravel()[0])
        label = CLASS_NAMES[1] if prob >= 0.5 else CLASS_NAMES[0]
        confidence = prob if label == CLASS_NAMES[1] else 1 - prob
        return label, confidence

    raise HTTPException(
        status_code=503,
        detail="No model file found. Place model.h5 or model.pth in /backend.",
    )


@app.get("/")
def health():
    return {"status": "ok", "service": "PneumoScan AI"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    data = await file.read()
    label, conf = predict_image(data)
    return {"prediction": label, "confidence": round(conf, 4)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
