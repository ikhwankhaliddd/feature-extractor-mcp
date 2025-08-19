import uuid
import requests
import numpy as np
from models import IdentityRequest, IdentityResponse
from typing import List
from io import BytesIO
from PIL import Image, ImageFile
import os
from dotenv import load_dotenv
import cv2
import tempfile

from openai import OpenAI

# External: InsightFace for embeddings
from insightface.app import FaceAnalysis


# External: VikingDB for storage
from volcengine.viking_db import *

load_dotenv()
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === CONFIG ===
SEED16_API_URL = "https://api.bytedance.com/seed16"
VIKINGDB_API_URL = "api-vikingdb.mlp.ap-mya.byteplus.com"
SEED16_API_KEY = os.getenv("SEED16_API_KEY")
VIKINGDB_API_KEY = os.getenv("VIKINGDB_API_KEY")
SEED_1_6_MODEL_ID = os.getenv("SEED_1_6_MODEL_ID")
AK = os.getenv("AK")
SK = os.getenv("SK")

# Init InsightFace
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    # Remove allowed_modules to enable all modules (detection + recognition)
)
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

client = OpenAI(
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    api_key=SEED16_API_KEY,
)

# Init VikingDB
vikingdb_service = VikingDBService(host=VIKINGDB_API_URL, region="ap-southeast-1")

vikingdb_service.set_ak(AK)
vikingdb_service.set_sk(SK)

collection = vikingdb_service.get_collection("Khaleed_Identity_Context_Collection")


async def extract_identity(request: IdentityRequest) -> IdentityResponse:
    # 1. Download image
    image = download_image(request.image_url)

    # 2. Get face embedding
    face_embedding, confidence = get_face_embedding(
        image, request.metadata.get("face_bbox")
    )

    # 3. Get identity description from Seed 1.6
    identity_description = call_seed16_identity_description(request.image_url)

    # 4. Get context embedding (semantic)
    # context_embedding = call_seed16_text_embedding(identity_description)

    # 5. Store in VikingDB
    store_in_vikingdb(
        img_url=request.image_url,
        face_bbox=request.metadata.get("face_bbox"),
        source="feature-extractor-mcp",
        confidence=confidence,
        style_tags=[""],
        created_at="",
        version="1.0",
        description=identity_description,
        img=request.tos_image_url,
    )

    return IdentityResponse(
        identity_description=identity_description,
        face_embedding=face_embedding,
        confidence=confidence,
    )


# --- Helper functions ---


def download_image(url: str) -> str:  # Return file path instead of PIL Image
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "image/*,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        img_bytes = resp.content

        if not img_bytes:
            raise ValueError("No image data received")

        # Check if we got HTML error page instead of image
        if img_bytes.startswith(b"<!DOCTYPE") or img_bytes.startswith(b"<html"):
            raise ValueError("Received HTML content instead of image data")

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_file.write(img_bytes)
        temp_file.close()

        return temp_file.name  # Return the file path

    except requests.RequestException as e:
        raise ValueError(f"Failed to download image: {str(e)}")


def get_face_embedding(
    image_path: str, bbox: List[int]
):  # Accept file path instead of PIL Image
    # Use cv2.imread() directly - this gives the format InsightFace expects
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert BGR to RGB for InsightFace
    img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Debug: Print image info
    print(f"Image shape: {img_np.shape}")
    print(f"Image dtype: {img_np.dtype}")
    print(f"Array flags: C_CONTIGUOUS={img_np.flags['C_CONTIGUOUS']}")

    # Always detect faces on the full image first
    faces = face_app.get(img_np)  # Now this gets the cv2.imread() format!

    print(f"Number of faces detected: {len(faces)}")

    # Clean up temporary file
    try:
        os.unlink(image_path)
    except:
        pass  # Ignore cleanup errors

    if not faces:
        raise ValueError("No face detected in the image.")

    # Debug: Print face info
    for i, face in enumerate(faces):
        print(f"Face {i}: bbox={face.bbox}, score={face.det_score}")

    # If bbox is provided, filter faces that are within the bbox
    if bbox:
        x, y, w, h = bbox
        bbox_faces = []

        for face in faces:
            # Check if face center is within bbox
            face_bbox = face.bbox  # [x1, y1, x2, y2]
            face_center_x = (face_bbox[0] + face_bbox[2]) / 2
            face_center_y = (face_bbox[1] + face_bbox[3]) / 2

            if x <= face_center_x <= x + w and y <= face_center_y <= y + h:
                bbox_faces.append(face)

        if not bbox_faces:
            raise ValueError(f"No face detected within the specified bbox {bbox}.")

        faces = bbox_faces

    # Get the face with highest confidence
    main_face = max(faces, key=lambda f: f.det_score)
    return main_face.embedding.tolist(), float(main_face.det_score)


def call_seed16_text_embedding(text: str) -> List[float]:
    payload = {"text": text}
    headers = {"Authorization": f"Bearer {SEED16_API_KEY}"}
    r = requests.post(f"{SEED16_API_URL}/embed", json=payload, headers=headers)
    r.raise_for_status()
    return r.json().get("embedding", [])


def store_in_vikingdb(
    img_url: str,
    face_bbox: List[int],
    source: str,
    confidence: float,
    style_tags: List[str],
    created_at: str,
    version: str,
    description: str,
    img: str,
):
    payload = {
        "id": uuid.uuid4().__str__(),
        "image_url": str(img_url),
        "face_bbox": face_bbox,
        "source": source,
        "confidence": confidence,
        "style_tags": style_tags,
        "created_at": created_at,
        "version": version,
        "description": description,
        "image": img,
    }
    data = Data(payload)
    datas = []
    datas.append(data)
    collection.upsert_data(datas)


def call_seed16_identity_description(image_url: str) -> str:

    response = client.chat.completions.create(
        model=SEED_1_6_MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_url}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "You are an expert visual analyst. Given an image, you describe the identity of the main person while keeping the description factual and visually observable. Avoid assumptions about personality or backstory.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image and describe the main subject's physical appearance:\n"
                            "- Gender & estimated age range\n"
                            "- Skin tone\n"
                            "- Hair color & style\n"
                            "- Clothing & colors\n"
                            "- Accessories\n"
                            "- Distinctive features\n"
                            "Return as a single clear prompt. (<= 120 characters)"
                        ),
                    }
                ],
            },
        ],
    )
    return response.choices[0].message.content
