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
import subprocess
import httpx
import os
import asyncio
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt6.QtGui import QColor
from qfluentwidgets import setTheme, Theme, MicaWindow, PrimaryPushButton, PushButton, PrimaryPushButton, PushButton, PrimaryPushButton, PushButton

from local_agent.agent import Agent, AVAILABLE_TOOLS # Import the Agent and tools

# --- Configuration ---
CONFIG_PATH = "local_agent/config.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()

# --- Agent Logic ---
class AgentWorker(QThread):
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
        asyncio.run(self.agent.process_chat(
            self.messages,
            self._send_response_callback,
            self._send_confirmation_request_callback
        ))

    def _send_response_callback(self, text: str):
        self.response_ready.emit(text)

    async def _send_confirmation_request_callback(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        self.mutex.lock()
        try:
            self.confirmation_required.emit(tool_name, parameters)
            self.condition.wait(self.mutex) # Wait for user confirmation
            return self.confirmed
        finally:
            self.mutex.unlock()

    def set_confirmation_response(self, response: bool):
        self.mutex.lock()
        try:
            self.confirmed = response
            self.condition.wakeOne() # Wake up the waiting thread
        finally:
            self.mutex.unlock()

class MainWindow(MicaWindow):
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
                color: #E0E0E0; /* Light gray for text */
                font-size: 16px;
            }
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.05); /* Very subtle translucent white */
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px; /* More rounded corners */
                padding: 15px;
                color: #E0E0E0;
            }
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.1); /* Slightly more opaque for input */
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 10px;
                color: #E0E0E0;
            }
            /* Fluent Design buttons will be styled by qfluentwidgets */
            .confirmation-buttons QPushButton {
                margin-left: 10px;
            }
        """)

        # --- Connections ---
        self.send_button.clicked.connect(self.send_message)
        self.input_field.returnPressed.connect(self.send_message)

    def send_message(self):
        message_text = self.input_field.text()
        if not message_text: return

        self.input_field.clear()
        self.update_chat(f"<p style=\"color: #87CEEB;\"><b>You:</b> {message_text}</p>") # User message in light blue

        # Start agent processing in a new thread
        self.current_worker = AgentWorker(self.agent, [{'role': 'user', 'content': message_text}])
        self.current_worker.response_ready.connect(self.update_chat)
        self.current_worker.confirmation_required.connect(self.show_confirmation)
        self.current_worker.start()

    def update_chat(self, text):
        self.chat_history.append(text)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def show_confirmation(self, tool_name: str, parameters: Dict[str, Any]):
        confirmation_text = f"<p style=\"color: #FFD700;\"><b>Agent wants to run:</b></p>"
        confirmation_text += f"<pre style=\"background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 10px;\"><code>Tool: {tool_name}\nParameters: {json.dumps(parameters, indent=2)}</code></pre>"
        confirmation_text += f"<p style=\"color: #FFD700;\">Allow this action?</p>"

        self.update_chat(confirmation_text)

        # Add confirmation buttons
        allow_btn = PrimaryPushButton("Allow")
        deny_btn = PushButton("Deny")

        allow_btn.clicked.connect(lambda: self._handle_confirmation_response(True))
        deny_btn.clicked.connect(lambda: self._handle_confirmation_response(False))

        # Create a temporary widget to hold buttons and add to layout
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.addWidget(allow_btn)
        button_layout.addWidget(deny_btn)
        button_layout.addStretch(1) # Push buttons to left
        button_container.setObjectName("confirmation_buttons") # For styling

        # Add the button container to the chat history (as a block element)
        self.chat_history.document().addBlock().setWidget(button_container)

    def _handle_confirmation_response(self, allowed: bool):
        if self.current_worker:
            self.current_worker.set_confirmation_response(allowed)
            # Remove confirmation buttons from UI after response
            # This is a bit tricky with QTextEdit. A simpler approach for now:
            # You might need to clear and re-append messages or manage a custom widget list.
            # For this example, we'll just append the user's choice.
            self.update_chat(f"<p style=\"color: #ADFF2F;\"><b>User:</b> {'Allowed' if allowed else 'Denied'}</p>")


if __name__ == "__main__":
    setTheme(Theme.DARK)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())