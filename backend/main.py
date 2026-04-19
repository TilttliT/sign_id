from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64

import random
import httpx
import os

app = FastAPI(title="Signature Recognition Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # заменить на конкретный адрес фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ML_VERIFY_URL = os.getenv("ML_VERIFY_URL", "http://localhost:8001/verify")
ML_IDENTIFY_URL = os.getenv("ML_IDENTIFY_URL", "http://localhost:8001/identify")
USE_REAL_ML = False


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')


def preprocess_image(image_bytes: bytes) -> bytes:
    return image_bytes


async def mock_verify(image1_b64: str, image2_b64: str, threshold: float) -> dict:
    # Случайный ответ
    match = random.random() > 0.3
    confidence = random.uniform(0.5, 0.99) if match else random.uniform(0.01, 0.5)
    return {
        "match": match,
        "confidence": confidence,
        "applied_threshold": threshold,
        "processing_time_ms": random.randint(10, 100)
    }


async def mock_identify(image_b64: str, threshold: float) -> dict:
    # Случайный ответ
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

    img1_bytes = preprocess_image(img1_bytes)
    img2_bytes = preprocess_image(img2_bytes)

    img1_b64 = image_bytes_to_base64(img1_bytes)
    img2_b64 = image_bytes_to_base64(img2_bytes)

    if USE_REAL_ML:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                payload = {
                    "image1_base64": img1_b64,
                    "image2_base64": img2_b64,
                    "user_threshold": threshold,
                    "mode": "verify"
                }
                response = await client.post(ML_VERIFY_URL, json=payload)
                response.raise_for_status()
                result = response.json()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"ML service error: {str(e)}")
    else:
        result = await mock_verify(img1_b64, img2_b64, threshold)

    return result


@app.post("/identify")
async def identify(
        image: UploadFile = File(...),
        threshold: float = Form(...)
):
    img_bytes = await image.read()
    img_bytes = preprocess_image(img_bytes)
    img_b64 = image_bytes_to_base64(img_bytes)

    if USE_REAL_ML:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                payload = {
                    "image_base64": img_b64,
                    "user_threshold": threshold,
                    "mode": "identify"
                }
                response = await client.post(ML_IDENTIFY_URL, json=payload)
                response.raise_for_status()
                result = response.json()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"ML service error: {str(e)}")
    else:
        result = await mock_identify(img_b64, threshold)

    return result


@app.get("/health")
async def health():
    return {"status": "ok"}