from pydantic import BaseModel, HttpUrl
from typing import List, Optional


class IdentityRequest(BaseModel):
    image_url: HttpUrl
    tos_image_url: str
    metadata: dict


class IdentityResponse(BaseModel):
    identity_description: str
    face_embedding: List[float]
    confidence: float
