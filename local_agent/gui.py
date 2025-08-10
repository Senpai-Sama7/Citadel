#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local Super Agent - Desktop GUI
---------------------------------

A standalone desktop application with a retro-futuristic, liquid-glass UI.
"""

import sys
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
    QHBoxLayout,
)
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from qfluentwidgets import setTheme, Theme, MicaWindow, PrimaryPushButton, PushButton

from local_agent.agent import Agent

# --- Configuration ---
CONFIG_PATH = "local_agent/config.json"

def load_config():
    """Loads the configuration from the specified path."""
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {CONFIG_PATH}")
        return {}

config = load_config()

# --- Agent Logic ---
class AgentWorker(QThread):
    """A worker thread for running the agent's processing logic."""
    response_ready = pyqtSignal(str)
    confirmation_required = pyqtSignal(str, dict)
    confirmation_response = pyqtSignal(bool)

    def __init__(self, agent_instance: Agent, messages: List[Dict[str, Any]]):
        super().__init__()
        self.agent = agent_instance
        self.messages = messages
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.confirmed = False

    def run(self):
        """Runs the agent's chat processing."""
        asyncio.run(self.agent.process_chat(
            self.messages,
            self._send_response_callback,
            self._send_confirmation_request_callback
        ))

    def _send_response_callback(self, text: str):
        """Sends a response from the agent to the main thread."""
        self.response_ready.emit(text)

    async def _send_confirmation_request_callback(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """Sends a confirmation request to the main thread."""
        self.mutex.lock()
        try:
            self.confirmation_required.emit(tool_name, parameters)
            self.condition.wait(self.mutex)
            return self.confirmed
        finally:
            self.mutex.unlock()

    def set_confirmation_response(self, response: bool):
        """Sets the confirmation response from the user."""
        self.mutex.lock()
        try:
            self.confirmed = response
            self.condition.wakeOne()
        finally:
            self.mutex.unlock()

class MainWindow(MicaWindow):
    """The main window of the application."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Super Agent")
        self.setWindowIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))

        # --- UI Elements ---
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")

        self.send_button = PrimaryPushButton("Send")

        # --- Layout ---
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.chat_history)
        main_layout.addLayout(input_layout)

        # --- Agent Instance ---
        self.agent = Agent()
        self.current_worker: Optional[AgentWorker] = None

        # --- Styling ---
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                color: #E0E0E0;
                font-size: 16px;
            }
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 15px;
                color: #E0E0E0;
            }
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 10px;
                color: #E0E0E0;
            }
            .confirmation-buttons QPushButton {
                margin-left: 10px;
            }
        """)

        # --- Connections ---
        self.send_button.clicked.connect(self.send_message)
        self.input_field.returnPressed.connect(self.send_message)

    def send_message(self):
        """Sends a message from the user to the agent."""
        message_text = self.input_field.text()
        if not message_text:
            return

        self.input_field.clear()
        self.update_chat(f'<p style="color: #87CEEB;"><b>You:</b> {message_text}</p>')

        self.current_worker = AgentWorker(self.agent, [{'role': 'user', 'content': message_text}])
        self.current_worker.response_ready.connect(self.update_chat)
        self.current_worker.confirmation_required.connect(self.show_confirmation)
        self.current_worker.start()

    def update_chat(self, text):
        """Updates the chat history with a new message."""
        self.chat_history.append(text)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def show_confirmation(self, tool_name: str, parameters: Dict[str, Any]):
        """Shows a confirmation dialog for a tool execution."""
        confirmation_text = f'<p style="color: #FFD700;"><b>Agent wants to run:</b></p>'
        confirmation_text += f'<pre style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 10px;"><code>Tool: {tool_name}\nParameters: {json.dumps(parameters, indent=2)}</code></pre>'
        confirmation_text += f'<p style="color: #FFD700;">Allow this action?</p>'

        self.update_chat(confirmation_text)

        allow_btn = PrimaryPushButton("Allow")
        deny_btn = PushButton("Deny")

        allow_btn.clicked.connect(lambda: self._handle_confirmation_response(True))
        deny_btn.clicked.connect(lambda: self._handle_confirmation_response(False))

        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.addWidget(allow_btn)
        button_layout.addWidget(deny_btn)
        button_layout.addStretch(1)
        button_container.setObjectName("confirmation_buttons")

        cursor = self.chat_history.textCursor()
        cursor.insertBlock()
        cursor.insertWidget(button_container)

    def _handle_confirmation_response(self, allowed: bool):
        """Handles the user's confirmation response."""
        if self.current_worker:
            self.current_worker.set_confirmation_response(allowed)
            self.update_chat(f'<p style="color: #ADFF2F;"><b>User:</b> {"Allowed" if allowed else "Denied"}</p>')

if __name__ == "__main__":
    setTheme(Theme.DARK)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
