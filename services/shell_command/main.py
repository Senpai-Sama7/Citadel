from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os

app = FastAPI(title="Shell Command Service")

# --- Simple API Key middleware (applies to all routes except health/docs) ---
API_KEY = os.getenv("API_KEY", "")
from starlette.responses import JSONResponse

@app.middleware("http")
async def _require_api_key(request, call_next):
    # allow unauthenticated access to health and docs
    if request.url.path in {"/health", "/docs", "/openapi.json"} or request.url.path.startswith("/docs"):
        return await call_next(request)
    key = request.headers.get("X-API-Key")
    if not API_KEY or key != API_KEY:
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return await call_next(request)



class ShellCommand(BaseModel):
    command: str

@app.post("/execute")
def execute_shell_command(cmd: ShellCommand):
    try:
        result = subprocess.run(cmd.command, shell=True, capture_output=True, text=True, check=True)
        return {"stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail={"stdout": e.stdout, "stderr": e.stderr})

@app.get("/health")
def health():
    return {"status": "ok"}
