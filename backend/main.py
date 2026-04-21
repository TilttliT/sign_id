import os
import tempfile
import base64
import random
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import sys
import os

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from signature_model.inference import SignatureVerifier
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import SignatureVerifier: {e}. Using mock mode.")
    ML_AVAILABLE = False

app = FastAPI(title="Signature Recognition Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = "../checkpoints/signature_model.pth"

verifier = None
if ML_AVAILABLE and os.path.exists(CHECKPOINT_PATH):
    try:
        verifier = SignatureVerifier(CHECKPOINT_PATH, device="cpu")
        print("✅ ML model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        verifier = None
else:
    if not ML_AVAILABLE:
        print("⚠️ SignatureVerifier not available, running in mock mode")
    elif not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️ Checkpoint not found at {CHECKPOINT_PATH}, running in mock mode")

def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

async def mock_verify(image1_b64: str, image2_b64: str, threshold: float) -> dict:
    match = random.random() > 0.3
    confidence = random.uniform(0.5, 0.99) if match else random.uniform(0.01, 0.5)
    return {
        "match": match,
        "confidence": confidence,
        "applied_threshold": threshold,
        "processing_time_ms": random.randint(10, 100)
    }

async def mock_identify(image_b64: str, threshold: float) -> dict:
    if random.random() > 0.4:
        return {
            "person_id": "ivanov",
            "person_name": "Иван Иванов",
            "confidence": random.uniform(0.7, 0.99),
            "applied_threshold": threshold,
            "is_unknown": False
        }
    else:
        return {
            "person_id": None,
            "person_name": None,
            "confidence": random.uniform(0.1, 0.5),
            "applied_threshold": threshold,
            "is_unknown": True,
            "message": "Подпись не распознана"
        }

@app.get("/")
async def root():
    return {"message": "Signature Recognition API", "status": "running"}
@app.post("/verify")
async def verify(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    threshold: float = Form(...)
):
    img1_bytes = await image1.read()
    img2_bytes = await image2.read()

    if verifier is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp2:
            tmp1.write(img1_bytes)
            tmp2.write(img2_bytes)
            tmp1.flush()
            tmp2.flush()
            try:
                is_match, similarity = verifier.verify(tmp1.name, tmp2.name)
                # Преобразуем numpy типы в стандартные Python
                is_match = bool(is_match)
                similarity = float(similarity)
                model_threshold = getattr(verifier, "threshold", threshold)
                applied = float(model_threshold)
                result = {
                    "match": is_match,
                    "confidence": similarity,
                    "applied_threshold": applied
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"ML model error: {str(e)}")
            finally:
                os.unlink(tmp1.name)
                os.unlink(tmp2.name)
    else:
        # мок-режим
        img1_b64 = image_bytes_to_base64(img1_bytes)
        img2_b64 = image_bytes_to_base64(img2_bytes)
        result = await mock_verify(img1_b64, img2_b64, threshold)

    return result

@app.post("/identify")
async def identify(
    image: UploadFile = File(...),
    threshold: float = Form(...)
):

    img_bytes = await image.read()
    img_b64 = image_bytes_to_base64(img_bytes)
    result = await mock_identify(img_b64, threshold)
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}