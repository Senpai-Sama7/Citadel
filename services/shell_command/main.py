#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shell Command Service (Refactored)
----------------------------------

This service provides a secure endpoint for executing shell commands.
It is protected by API key authentication.
"""

import os
import logging
import subprocess
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# --- Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
API_KEY = os.getenv("API_KEY")

# --- Logging ---
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Pre-flight Checks ---
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set.")

# --- Models ---
class ShellCommand(BaseModel):
    command: str

# --- Service Initialization ---
app = FastAPI(
    title="Shell Command Service",
    description="Executes shell commands.",
    version="2.0.0"
)

# --- Security ---
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(key: str = Security(api_key_header)):
    if key == API_KEY:
        return key
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# --- API Endpoints ---
@app.post("/execute", summary="Execute a shell command")
def execute_shell_command(cmd: ShellCommand, api_key: str = Security(get_api_key)):
    """
    Executes a shell command and returns the output.
    """
    try:
        result = subprocess.run(
            cmd.command, shell=True, capture_output=True, text=True, check=True, timeout=120
        )
        return {"stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail={"stdout": e.stdout, "stderr": e.stderr})
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Command timed out after 120 seconds.")
    except Exception as e:
        logger.error(f"Error executing shell command: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/health", summary="Health check endpoint")
def health():
    """Provides a basic health check of the service."""
    return {"status": "ok"}
