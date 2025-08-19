from fastapi import FastAPI, HTTPException
from models import IdentityRequest, IdentityResponse
from service import extract_identity
import uvicorn
import json

app = FastAPI(title="Feature Extractor MCP")


@app.post("/extract-feature", response_model=IdentityResponse)
async def extract_identity_endpoint(request: IdentityRequest):
    # Work on a clean dict version
    req_dict = request.model_dump()

    if isinstance(req_dict["metadata"], str):
        try:
            req_dict["metadata"] = json.loads(req_dict["metadata"])
        except json.JSONDecodeError:
            pass  # leave as string if invalid JSON

    try:
        result = extract_identity(
            req_dict["image_url"],
            req_dict["metadata"]["face_bbox"],
            req_dict["tos_image_url"],
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Feature Extractor MCP Agent is running on port 8004.")
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
