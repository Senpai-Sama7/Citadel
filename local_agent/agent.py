#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local Super Agent - Core Logic
------------------------------

This module contains the core AI logic for the Local Super Agent, including
LLM interaction, tool definitions, and the main chat processing loop.
"""

import os
import json
import logging
import subprocess
import httpx
import sqlite3
import re
from typing import List, Dict, Any, Optional, Callable

from llama_cpp import Llama
from git import Repo, InvalidGitRepositoryError

# --- Configuration ---
CONFIG_PATH = os.getenv("CITADEL_CONFIG_PATH", "local_agent/config.json")

def load_config():
    """Loads the configuration from the specified path."""
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {CONFIG_PATH}")
        # Provide a default configuration
        return {
            "model_path": "models/Llama-3-8B-Instruct-Q4_K_M.gguf",
            "n_gpu_layers": -1,
            "n_ctx": 4096,
            "max_tokens": 1024,
            "temperature": 0.3,
            "log_level": "INFO",
            "autonomous_mode": False,
            "gateway_url": "http://localhost:8010"
        }

config = load_config()

LOG_LEVEL = config.get("log_level", "INFO").upper()
MODEL_PATH = config.get("model_path")
N_GPU_LAYERS = config.get("n_gpu_layers", -1)
N_CTX = config.get("n_ctx", 4096)
MAX_TOKENS = config.get("max_tokens", 1024)
TEMPERATURE = config.get("temperature", 0.3)
AUTONOMOUS_MODE = config.get("autonomous_mode", False)
GATEWAY_URL = config.get("gateway_url", "http://localhost:8010")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("local_super_agent_core")

# --- Persistent Memory (SQLite) ---
DB_PATH = "local_agent/memory.db"

def init_db():
    """Initializes the SQLite database for persistent memory."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                fact TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        log.error(f"Database error: {e}")

init_db()

# --- Tool Definitions ---
class Tool:
    """A class to represent a tool that the agent can use."""
    def __init__(self, name: str, description: str, func: Callable, requires_confirmation: bool = False):
        self.name = name
        self.description = description
        self.func = func
        self.requires_confirmation = requires_confirmation
        self.parameters = {k: "<...>" for k in func.__code__.co_varnames}

    def to_dict(self):
        """Returns a dictionary representation of the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

def _run_shell_command(command: str) -> str:
    """
    Executes a shell command and returns the output.
    Note: This function uses shell=True, which can be a security risk.
    Ensure that the command is sanitized before execution.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.CalledProcessError as e:
        return f"ERROR:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}\nExit Code: {e.returncode}"
    except Exception as e:
        return f"ERROR: {str(e)}"

def _read_file(path: str) -> str:
    """Reads the content of a file."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"ERROR: File not found at {path}"
    except Exception as e:
        return f"ERROR: Could not read file: {str(e)}"

def _write_file(path: str, content: str) -> str:
    """Writes content to a file."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"ERROR: Could not write to file: {str(e)}"

def _list_directory(path: str) -> str:
    """Lists the contents of a directory."""
    try:
        return "\n".join(os.listdir(path))
    except FileNotFoundError:
        return f"ERROR: Directory not found at {path}"
    except Exception as e:
        return f"ERROR: Could not list directory: {str(e)}"

def _search_file_content(pattern: str, path: str = ".") -> str:
    """Searches for a regular expression pattern within files."""
    matches = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if re.search(pattern, line):
                            matches.append(f"{file_path}:{i+1}: {line.strip()}")
            except Exception:
                continue
    return "\n".join(matches) if matches else "No matches found."

def _glob_files(pattern: str, path: str = ".") -> str:
    """Finds files matching a glob pattern."""
    import glob
    try:
        return "\n".join(glob.glob(os.path.join(path, pattern), recursive=True))
    except Exception as e:
        return f"ERROR: Could not glob files: {str(e)}"

def _install_package(package_name: str) -> str:
    """Installs a Python package using pip."""
    return _run_shell_command(f"pip install {package_name}")

def _run_tests(test_path: str = ".") -> str:
    """Runs pytest tests."""
    return _run_shell_command(f"pytest {test_path}")

def _git_command(command: str) -> str:
    """Executes a Git command."""
    try:
        repo = Repo(os.getcwd())
        if command.startswith("commit"):
            parts = command.split(" ", 1)
            if len(parts) > 1:
                return _run_shell_command(f'git commit -m "{parts[1].replace("`", "")}"')
            else:
                return "ERROR: Git commit command requires a message."
        return _run_shell_command(f"git {command}")
    except InvalidGitRepositoryError:
        return "ERROR: Not a git repository."
    except Exception as e:
        return f"ERROR: Git command failed: {str(e)}"

async def _call_remote_service(service_prefix: str, endpoint: str, payload: Dict[str, Any]) -> str:
    """A helper function to call a remote service."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{GATEWAY_URL}/{service_prefix}/{endpoint}", json=payload, timeout=60.0)
            response.raise_for_status()
            return json.dumps(response.json())
    except httpx.RequestError as e:
        return f"ERROR: Network error calling {service_prefix}/{endpoint}: {e}"
    except httpx.HTTPStatusError as e:
        return f"ERROR: {service_prefix}/{endpoint} returned HTTP error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"ERROR: Unexpected error calling {service_prefix}/{endpoint}: {e}"

async def _web_search(query: str, max_results: int = 5) -> str:
    """Performs a web search."""
    return await _call_remote_service("web", "search", {"query": query, "max_results": max_results})

async def _web_fetch(url: str) -> str:
    """Fetches the content of a URL."""
    return await _call_remote_service("web", "fetch", {"url": url})

def _save_memory(fact: str) -> str:
    """Saves a fact to the agent's persistent memory."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO memory (fact) VALUES (?)", (fact,))
        conn.commit()
        conn.close()
        return "Fact saved to memory."
    except sqlite3.Error as e:
        return f"ERROR: Could not save fact to memory: {e}"

def _retrieve_memory(query: str) -> str:
    """Retrieves facts from the agent's persistent memory."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT fact FROM memory WHERE fact LIKE ?", (f"%{query}%",))
        results = cursor.fetchall()
        conn.close()
        return "\n".join([row[0] for row in results]) if results else "No relevant facts found in memory."
    except sqlite3.Error as e:
        return f"ERROR: Could not retrieve facts from memory: {e}"

def _restart_agent() -> str:
    """Signals that the agent needs to be restarted."""
    return "RESTART_SIGNAL: Agent needs to be restarted to apply changes. Please restart the application."

async def _vector_search(query: str, top_k: int = 3) -> str:
    """Performs a vector search."""
    return await _call_remote_service("vector", "search", {"query": query, "top_k": top_k})

async def _knowledge_graph_query(query: str) -> str:
    """Queries the knowledge graph."""
    return await _call_remote_service("knowledge", "query", {"query": query})

async def _time_series_forecast(series: List[float], steps: int = 10) -> str:
    """Forecasts a time series."""
    return await _call_remote_service("time", "forecast", {"series": series, "steps": steps})

async def _causal_inference(data: Dict[str, Any]) -> str:
    """Performs causal inference."""
    return await _call_remote_service("causal", "infer", {"data": data})

async def _multi_modal_process(image_url: str, text: str) -> str:
    """Processes multi-modal data."""
    return await _call_remote_service("multi", "process", {"image_url": image_url, "text": text})

async def _hierarchical_classification(text: str) -> str:
    """Performs hierarchical classification."""
    return await _call_remote_service("hier", "classify", {"text": text})

async def _rule_engine_evaluate(data: Dict[str, Any]) -> str:
    """Evaluates data against rules."""
    return await _call_remote_service("rule", "evaluate", {"data": data})

async def _orchestrator_publish(event_type: str, data: Dict[str, Any]) -> str:
    """Publishes an event to the orchestrator."""
    return await _call_remote_service("orch", "publish", {"type": event_type, "data": data})

AVAILABLE_TOOLS = [
    Tool("execute_shell", "Executes a shell command on the local machine. Use this for any command-line operations.", _run_shell_command, requires_confirmation=True),
    Tool("read_file", "Reads the content of a file.", _read_file),
    Tool("write_file", "Writes content to a file. Use with caution, especially for its own source code.", _write_file, requires_confirmation=True),
    Tool("list_directory", "Lists the contents of a directory.", _list_directory),
    Tool("search_file_content", "Searches for a regular expression pattern within the content of files in a specified directory.", _search_file_content),
    Tool("glob_files", "Finds files matching specific glob patterns.", _glob_files),
    Tool("install_package", "Installs a Python package using pip.", _install_package, requires_confirmation=True),
    Tool("run_tests", "Runs pytest tests in a specified path.", _run_tests, requires_confirmation=True),
    Tool("git_command", "Executes a Git command (e.g., 'git status', 'git add .', 'git commit -m \"msg\"').", _git_command, requires_confirmation=True),
    Tool("web_search", "Performs a web search to find information on the internet.", _web_search),
    Tool("web_fetch", "Fetches the content of a specific URL.", _web_fetch),
    Tool("vector_search", "Searches for information in a vector database.", _vector_search),
    Tool("knowledge_graph_query", "Queries a graph database to find relationships between entities.", _knowledge_graph_query),
    Tool("time_series_forecast", "Predicts future values for a time series.", _time_series_forecast),
    Tool("causal_inference", "Performs causal inference on provided data.", _causal_inference),
    Tool("multi_modal_process", "Processes multi-modal data (e.g., image and text).", _multi_modal_process),
    Tool("hierarchical_classification", "Performs hierarchical classification on text.", _hierarchical_classification),
    Tool("rule_engine_evaluate", "Evaluates data against defined rules.", _rule_engine_evaluate),
    Tool("orchestrator_publish", "Publishes an event to the orchestrator.", _orchestrator_publish),
    Tool("save_memory", "Saves a specific fact or piece of information to the agent's persistent memory.", _save_memory),
    Tool("retrieve_memory", "Retrieves relevant facts from the agent's persistent memory based on a query.", _retrieve_memory),
    Tool("restart_agent", "Signals that the agent needs to be restarted to apply changes (e.g., after code modifications).", _restart_agent, requires_confirmation=True),
]

class Agent:
    """The core of the Local Super Agent."""
    def __init__(self):
        self.llm: Optional[Llama] = None
        self._load_model()

    def _load_model(self):
        """Loads the Llama model."""
        if not os.path.exists(MODEL_PATH):
            log.error(f"Fatal: Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        log.info(f"Loading model: {MODEL_PATH}")
        try:
            self.llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=True)
            log.info("Model loaded successfully.")
        except Exception as e:
            log.critical(f"Failed to load LLM from {MODEL_PATH}: {e}", exc_info=True)
            raise RuntimeError(f"LLM failed to load: {e}")

    def get_system_prompt(self) -> str:
        """Generates the system prompt for the agent."""
        tool_list_str = json.dumps([tool.to_dict() for tool in AVAILABLE_TOOLS], indent=2)

        return f"""You are a super-intelligent, autonomous AI agent with full access to this computer and a suite of specialized microservices. Your goal is to solve the user's request by creating a plan and then executing it. You can also modify your own source code and architecture.

**Your Project Structure:**
Your own code is located in the `local_agent/` directory.
- `local_agent/gui.py`: The graphical user interface.
- `local_agent/agent.py`: Your core logic (this file).
- `local_agent/config.json`: Your configuration settings.
- `local_agent/requirements.txt`: Your Python dependencies.
- `local_agent/memory.db`: Your persistent memory database.

**Phase 1: Planning**
First, think step-by-step to create a plan to address the user's request. The plan should be a sequence of tool calls. Consider the following:
- **Analyze:** What information do you need? What tools can provide it?
- **Break Down:** Can the task be broken into smaller, manageable steps?
- **Prioritize:** What's the most logical order of operations?
- **Self-Correction:** If a previous attempt failed, how can you adjust your plan?
- **Reflection:** After a task, consider what you learned and if you should save it to memory.

**Phase 2: Execution**
Execute the plan by calling the necessary tools. You have access to the following tools:
{tool_list_str}

When you need to use a tool, respond with a JSON object containing the tool name and its parameters, like this:
{{"tool": "<tool_name>", "parameters": {{...}}}}

If a tool requires confirmation (marked with `requires_confirmation: true`), you will receive a `CONFIRMATION_REQUIRED` signal. You must wait for user approval before proceeding.

After each tool call, you will get the result. Use this result to inform your next step.

**Important Considerations:**
- **Safety:** Be extremely cautious with `execute_shell`, `write_file` (especially on your own code), `install_package`, `run_tests`, `git_command`, and `restart_agent`. Always consider the potential impact.
- **Self-Modification:** When modifying your own code (`local_agent/agent.py` or `local_agent/gui.py`), plan carefully. Read the file, make precise changes, and then write the file back. After modifying your own code, you MUST use the `restart_agent` tool to apply changes.
- **Git:** Use `git_command` to manage your codebase (e.g., `git status`, `git add .`, `git commit`).
- **Memory:** Use `save_memory` to store important facts or insights you gain. Use `retrieve_memory` to recall past information.
- **Remote Services:** Tools like `web_search`, `web_fetch`, `vector_search`, `knowledge_graph_query`, `time_series_forecast`, `causal_inference`, `multi_modal_process`, `hierarchical_classification`, `rule_engine_evaluate`, and `orchestrator_publish` communicate with Dockerized microservices. Ensure these services are running via `docker-compose` for these tools to function.
- **Final Answer:** If you have enough information or have completed the task, provide a direct, helpful response to the user.
"""

    async def process_chat(self, messages: List[Dict[str, Any]], send_response: Callable[[str], None], send_confirmation_request: Callable[[str, Dict[str, Any]], bool]) -> None:
        """Processes a chat message from the user."""
        full_messages = [{"role": "system", "content": self.get_system_prompt()}] + messages

        for i in range(10): # Limit iterations to prevent infinite loops
            log.debug(f"Iteration {i+1}: LLM Request Messages: {full_messages}")

            try:
                stream = self.llm.create_chat_completion(
                    messages=full_messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stream=True,
                    response_format={"type": "json_object"} if i == 0 or (full_messages and full_messages[-1].get("role") == "tool") else {"type": "text"}
                )

                full_response_content = ""
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    full_response_content += delta

                log.debug(f"LLM Full Response: {full_response_content}")

                try:
                    tool_call = json.loads(full_response_content)
                    if "tool" in tool_call and "parameters" in tool_call:
                        tool_name = tool_call["tool"]
                        parameters = tool_call["parameters"]
                        log.info(f"LLM requested tool call: {tool_call}")

                        tool_obj = next((t for t in AVAILABLE_TOOLS if t.name == tool_name), None)
                        if not tool_obj:
                            tool_result = f"ERROR: Tool '{tool_name}' not found."
                            log.warning(tool_result)
                        else:
                            if tool_obj.requires_confirmation and not AUTONOMOUS_MODE:
                                confirmed = await send_confirmation_request(tool_name, parameters)
                                if not confirmed:
                                    tool_result = f"User denied execution of {tool_name}."
                                    log.info(tool_result)
                                else:
                                    tool_result = tool_obj.func(**parameters)
                            else:
                                tool_result = tool_obj.func(**parameters)

                        full_messages.append({"role": "assistant", "content": full_response_content})
                        full_messages.append({"role": "tool", "content": str(tool_result)})
                        continue

                except (json.JSONDecodeError, TypeError):
                    send_response(full_response_content)
                    return

            except Exception as e:
                log.error(f"Error during LLM interaction or tool loop: {e}", exc_info=True)
                send_response(f"An error occurred during processing: {e}")
                return

        send_response("Max iterations reached without a final answer. Please try again with a more specific query.")
