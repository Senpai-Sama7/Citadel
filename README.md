# The Ciadel

This project now features a powerful, self-contained AI agent designed to run directly on your local computer with extensive capabilities. It replaces the previous microservices architecture for a more integrated and user-friendly experience.

## Key Features:

*   **Desktop GUI:** A retro-futuristic, liquid-glass like neo-skeuomorphic graphical interface for intuitive interaction.
*   **Full Local Access:** The agent has direct access to your computer's file system and can execute shell commands.
*   **Self-Modification:** The agent can read, analyze, and modify its own source code and underlying architecture.
*   **Deep Codebase Understanding:** Equipped with tools to search file content, glob files, and interact with Git repositories.
*   **Enhanced Intelligence:** Utilizes a "Plan-and-Execute" methodology for more deliberate and effective problem-solving.
*   **Persistent Memory:** Stores facts and insights across sessions using a local SQLite database.
*   **Web Interaction:** Can perform web searches and fetch content from URLs.
*   **Development Tools:** Includes capabilities for installing Python packages and running tests.
*   **Safety Controls:** Features a configurable "Autonomous Mode" with a default to require user confirmation for sensitive actions (e.g., executing commands, modifying files).

## Architecture Overview:

The core of the system is the `local_agent` directory, which contains:

*   `local_agent/gui.py`: The desktop graphical user interface.
*   `local_agent/agent.py`: The core AI logic, including the Large Language Model (LLM) integration, tool definitions, and the chat processing loop.
*   `local_agent/config.json`: Configuration settings for the agent (e.g., model path, safety mode).
*   `local_agent/requirements.txt`: Python dependencies required for the agent.
*   `local_agent/memory.db`: The SQLite database for persistent memory.

## Getting Started:

### Prerequisites:

*   **Python 3.10+:** Ensure Python is installed on your system.
*   **Git:** (Optional, but recommended for agent's Git capabilities) Ensure Git is installed.

### Setup & Running:

1.  **Open your Terminal/Command Prompt** and navigate to the project's root directory (`full_ai_platform`).

2.  **Install Dependencies:**
    ```bash
    pip install -r local_agent/requirements.txt
    ```

3.  **Run the Agent GUI:**
    ```bash
    python local_agent/gui.py
    ```
    The desktop application window will appear. Keep the terminal open as long as the agent is running.

### (Optional) Create a One-Click Executable:

To run the agent without the command line:

1.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```

2.  **Build the Executable:**
    ```bash
    pyinstaller --onefile --windowed --name LocalSuperAgent local_agent/gui.py
    ```
    This will create a `dist` folder in your project root, containing the `LocalSuperAgent` executable. You can then launch the agent by double-clicking this file.

## Safety & Autonomous Mode:

By default, the agent operates in **Safe Mode**, requiring your explicit confirmation for sensitive actions (e.g., running shell commands, modifying files). This is highly recommended.

To enable **Autonomous Mode** (where the agent acts without confirmation), edit `local_agent/config.json` and change `"autonomous_mode": false` to `"autonomous_mode": true`. **WARNING: This mode is extremely dangerous and can lead to unintended and potentially destructive changes to your system.**

## Previous Microservices Architecture:

The `backend/` and `services/` directories contain the previous Docker-based microservices architecture. While still present, the primary focus and development are now on the `local_agent` for direct local interaction.
