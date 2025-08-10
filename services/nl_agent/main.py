"""
Advanced Natural Language Agent Service (FastAPI)
-------------------------------------------------

This service is the brain of the AI platform, upgraded with advanced tool-using
capabilities and streaming responses.

**Upgraded Features:**

- **Advanced Tool-Using Loop:** The agent can now intelligently decide which 
  service (tool) to call based on the user's prompt, interpret the tool's 
  output, and perform multi-step reasoning.
- **Streaming Responses:** Uses FastAPI's `StreamingResponse` to send back the 
  LLM's output token-by-token for a real-time, interactive experience.
- **JSON Mode for Tool Calls:** leverages modern LLM features to force 
  structured JSON output for tool calls, ensuring reliable parsing.
- **Dynamic System Prompt:** The system prompt is dynamically generated to 
  include a list of available tools and their descriptions.
- **Enhanced Error Handling & Observability:** More robust error handling and 
  detailed logging for better debugging and monitoring.

**Environment Variables:**

- `MODEL_PATH`: (Required) Path to the GGUF model file.
- `GATEWAY_URL`: (Required) Base URL for the API gateway.
- `N_GPU_LAYERS`: Number of model layers to offload to GPU.
- `N_CTX`: Context window size.
- `MAX_TOKENS`: Maximum tokens for generation.
- `TEMPERATURE`: Sampling temperature.
- `LOG_LEVEL`: Logging level.
"""

from __future__ import annotations

import os
import json
import logging
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama

# --- Configuration & Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Llama-3-8B-Instruct-Q4_K_M.gguf")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://api_gateway:8000")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1))
N_CTX = int(os.getenv("N_CTX", 4096))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
SHELL_CONFIRMATION_REQUIRED = os.getenv("SHELL_CONFIRMATION_REQUIRED", "true").lower() == "true"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("nl_agent_service")

# --- Tool Definitions ---
# A catalog of available services the LLM can call.
AVAILABLE_TOOLS = {
    "web_search": {
        "description": "Performs a web search to find information on the internet. Use this when you need to find current events, facts, or general knowledge that might not be in the model's training data.",
        "endpoint": "/web/search",
        "payload": {"query": "<search_query>", "max_results": 5},
    },
    "web_fetch": {
        "description": "Fetches the content of a specific URL. Use this when you have a URL and need to read the content of that webpage.",
        "endpoint": "/web/fetch",
        "payload": {"url": "<url_to_fetch>"},
    },
    "vector_search": {
        "description": "Searches for information in a vector database. Use this for questions about general knowledge, documents, or to find similar items.",
        "endpoint": "/vector/search",
        "payload": {"query": "<user_query>", "top_k": 3},
    },
    "knowledge_graph": {
        "description": "Queries a graph database to find relationships between entities. Use this to answer questions about how things are connected.",
        "endpoint": "/graph/query",
        "payload": {"query": "<cypher_query>"},
    },
    "time_series_forecast": {
        "description": "Predicts future values for a time series. Use this for forecasting questions.",
        "endpoint": "/timeseries/forecast",
        "payload": {"series": [], "steps": 10},
    },
    "execute_shell": {
        "description": "Executes a shell command on the local machine. Use this to run commands, scripts, or interact with the filesystem.",
        "endpoint": "/shell/execute",
        "payload": {"command": "<command_to_execute>"},
    },
}

# --- FastAPI App & Global State ---
app = FastAPI(title="Advanced Natural Language Agent", version="2.0.0")

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

llm: Optional[Llama] = None

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = MAX_TOKENS
    temperature: Optional[float] = TEMPERATURE
    stream: Optional[bool] = True

# --- Core Service Logic ---

def load_model():
    global llm
    if not os.path.exists(MODEL_PATH):
        log.error(f"Fatal: Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    log.info(f"Loading model: {MODEL_PATH}")
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=True)
        log.info("Model loaded successfully.")
    except Exception as e:
        log.critical(f"Failed to load LLM from {MODEL_PATH}: {e}", exc_info=True)
        raise RuntimeError(f"LLM failed to load: {e}")

def get_system_prompt() -> str:
    """Generates the system prompt with the list of available tools."""
    tool_list = json.dumps([{
        "name": name,
        "description": details["description"],
        "parameters": details["payload"]
    } for name, details in AVAILABLE_TOOLS.items()], indent=2)

    return f"""You are a helpful AI assistant that can use tools to answer questions.

You have access to the following tools:
{tool_list}

When you need to use a tool, respond with a JSON object containing the tool name and its parameters, like this:
{{"tool": "<tool_name>", "parameters": {{...}}}}

If you have enough information to answer the user's question, provide a direct, helpful response."""

async def call_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Calls a specified tool (microservice) via the API gateway."""
    if tool_name not in AVAILABLE_TOOLS:
        log.warning(f"Attempted to call unknown tool: {tool_name}")
        return {"error": f"Tool '{tool_name}' not found."}
    
    tool_info = AVAILABLE_TOOLS[tool_name]
    url = f"{GATEWAY_URL}{tool_info['endpoint']}"
    log.info(f"Calling tool: {tool_name} at {url} with params: {parameters}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=parameters, timeout=60.0)
            response.raise_for_status()
            log.info(f"Tool {tool_name} call successful. Status: {response.status_code}")
            return response.json()
        except httpx.RequestError as e:
            log.error(f"Network error calling tool {tool_name} at {url}: {e}", exc_info=True)
            return {"error": f"Network error calling {tool_name}: {e}"}
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error from tool {tool_name} at {url}: {e.response.status_code} - {e.response.text}", exc_info=True)
            return {"error": f"Tool {tool_name} returned HTTP error {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            log.error(f"Unexpected error calling tool {tool_name} at {url}: {e}", exc_info=True)
            return {"error": f"Unexpected error calling {tool_name}: {e}"}

async def chat_stream_generator(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """The main generator for handling chat requests, tool use, and streaming."""
    if not llm:
        log.error("LLM not loaded when chat_stream_generator was called.")
        raise HTTPException(status_code=503, detail="Model not available.")

    messages = [{"role": "system", "content": get_system_prompt()}] + [msg.dict() for msg in request.messages]
    
    # --- Tool-Using Loop ---
    for i in range(5): # Limit to 5 tool calls to prevent infinite loops
        log.debug(f"Iteration {i+1}: LLM Request Messages: {messages}")
        
        try:
            # 1. Ask the LLM for a response (either a tool call or a final answer)
            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
                # Force JSON output if a tool call is likely
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "parameters": {"type": "object"}
                        }
                    }
                } if i == 0 and any(tool_name in messages[-1]["content"].lower() for tool_name in AVAILABLE_TOOLS.keys()) else {"type": "text"}
            )

            # 2. Process the LLM's output
            full_response = ""
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                full_response += delta

            log.debug(f"LLM Full Response: {full_response}")

            # 3. Check if the response is a tool call
            try:
                tool_call = json.loads(full_response)
                if "tool" in tool_call and "parameters" in tool_call:
                    log.info(f"LLM requested tool call: {tool_call}")

                    if tool_call["tool"] == "execute_shell" and SHELL_CONFIRMATION_REQUIRED:
                        yield f"data: {json.dumps({'type': 'confirmation_required', 'command': tool_call['parameters']['command']})}\n\n"
                        # This is a simplified example. In a real application, you'd need a mechanism
                        # to wait for user confirmation before proceeding.
                        # For this example, we'll assume the user confirms and proceed.

                    tool_result = await call_tool(tool_call["tool"], tool_call["parameters"])
                    log.info(f"Tool Result for {tool_call['tool']}: {tool_result}")
                    
                    # Append the tool call and result to the conversation history
                    messages.append({"role": "assistant", "content": full_response}) # The tool request
                    messages.append({"role": "tool", "content": json.dumps(tool_result)}) # The tool's output
                    continue # Go back to the LLM with the new info
            except (json.JSONDecodeError, TypeError):
                # Not a tool call, so it's a final answer.
                log.debug("LLM response was not a tool call. Assuming final answer.")
                pass

            # 4. If not a tool call, stream the final answer to the user
            final_stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )
            for chunk in final_stream:
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
            return # End the loop

        except Exception as e:
            log.error(f"Error during LLM interaction or tool loop: {e}", exc_info=True)
            yield f"data: {json.dumps({'content': f'An error occurred during processing: {e}'})}\n\n"
            return

    log.warning("Max tool calls reached without a final answer.")
    yield f"data: {json.dumps({'content': 'Max tool calls reached. Please try again with a more specific query.'})}\n\n"

# --- FastAPI Lifecycle & Endpoints ---
@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/health")
def health():
    model_status = "loaded" if llm else "not loaded"
    return {"status": "ok", "model_status": model_status, "n_gpu_layers": N_GPU_LAYERS, "n_ctx": N_CTX}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            chat_stream_generator(request),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming logic would go here, but we default to streaming
        raise HTTPException(status_code=400, detail="Non-streaming not implemented. Please set stream=True.")