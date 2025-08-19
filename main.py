from fastapi import FastAPI, HTTPException
from models import IdentityRequest, IdentityResponse
from service import extract_identity
import uvicorn
import json

app = FastAPI(title="Feature Extractor MCP")


@app.post("/extract-feature", response_model=IdentityResponse)
async def extract_identity_endpoint(request: IdentityRequest):
    if isinstance(request.metadata, str):
        try:
            json_metadata = json.loads(request.metadata)
            request.metadata = json_metadata
        except json.JSONDecodeError:
            pass  # leave as string if invalid JSON

    try:
        result = await extract_identity(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Feature Extractor MCP Agent is running on port 8004.")
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
