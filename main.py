from fastapi import FastAPI, HTTPException
from models import IdentityRequest, IdentityResponse
from service import extract_identity

app = FastAPI(title="Feature Extractor MCP")


@app.post("/extract-feature", response_model=IdentityResponse)
async def extract_identity_endpoint(request: IdentityRequest):
    try:
        result = await extract_identity(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
