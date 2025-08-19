from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Any


class IdentityRequest(BaseModel):
    image_url: HttpUrl
    tos_image_url: str
    metadata: Optional[Any]


class IdentityResponse(BaseModel):
    identity_description: str
    face_embedding: List[float]
    confidence: float
