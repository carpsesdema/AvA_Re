# app/ui/dialogs/personality_dialog.py
import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton,
    QDialogButtonBox, QLabel, QScrollArea, QWidget
)
from PySide6.QtGui import QFont

try:
    # Assuming constants are in utils
    from utils import constants
except ImportError:
    logging.getLogger(__name__).warning("PersonalityDialog: utils.constants not found, using fallback fonts.")
    constants = type("constants", (object,), {"CHAT_FONT_FAMILY": "Segoe UI", "CHAT_FONT_SIZE": 10})  # type: ignore

logger = logging.getLogger(__name__)


class EditPersonalityDialog(QDialog):
    """
    A dialog for editing the AI's personality (system prompt).
    """

    def __init__(self, current_prompt: Optional[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit AI Persona (System Prompt)")
        self.setMinimumSize(500, 400)  # Decent default size

        self._prompt_text_edit: Optional[QTextEdit] = None
        self._initial_prompt = current_prompt or ""  # Store initial to detect changes

        self._init_ui()
        if self._prompt_text_edit:
            self._prompt_text_edit.setPlainText(self._initial_prompt)

        logger.debug("EditPersonalityDialog initialized.")

    def _init_ui(self):
        """Initializes the UI elements of the dialog."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # --- Title Label ---
        title_label = QLabel("Customize AI Personality")
        title_font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"),
                           getattr(constants, 'CHAT_FONT_SIZE', 10) + 2, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # --- Description/Instructions ---
        description_label = QLabel(
            "Edit the system prompt below to change how the chat AI behaves. "
            "This prompt guides its responses, tone, and knowledge focus. "
            "Clear the text to revert to the default persona for the selected model."
        )
        description_label.setWordWrap(True)
        description_label.setStyleSheet("color: #8B949E;")  # Secondary text color
        layout.addWidget(description_label)

        # --- Prompt Text Edit Area ---
        self._prompt_text_edit = QTextEdit()
        self._prompt_text_edit.setObjectName("PersonalityPromptTextEdit")
        self._prompt_text_edit.setAcceptRichText(False)
        self._prompt_text_edit.setPlaceholderText(
            "Enter system prompt here... (e.g., 'You are a helpful assistant specializing in Python programming.')")
        font_size = getattr(constants, 'CHAT_FONT_SIZE', 10)
        text_edit_font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), font_size)
        self._prompt_text_edit.setFont(text_edit_font)
        self._prompt_text_edit.setMinimumHeight(150)  # Ensure enough space for editing
        layout.addWidget(self._prompt_text_edit, 1)  # Takes up available vertical space

        # --- Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Reset)
        self.button_box.button(QDialogButtonBox.StandardButton.Reset).setText("Use Default")  # type: ignore

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        reset_button = self.button_box.button(QDialogButtonBox.StandardButton.Reset)
        if reset_button:
            reset_button.clicked.connect(self._reset_to_default_placeholder)  # Placeholder action

        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def _reset_to_default_placeholder(self):
        """
        Clears the text edit, signifying that the default persona for the
        currently selected chat LLM should be used. The actual default prompt
        is managed by BackendConfigManager or ChatManager.
        """
        if self._prompt_text_edit:
            self._prompt_text_edit.clear()
        logger.info("Personality prompt reset to default (cleared in dialog).")
        # Optionally, could show a message or disable reset if already default.

    def get_prompt_text(self) -> str:
        """Returns the edited prompt text."""
        if self._prompt_text_edit:
            # If text is empty, it implies using the default.
            # The component receiving this (BackendConfigManager via ChatManager)
            # will interpret an empty string as "use default for the model".
            return self._prompt_text_edit.toPlainText().strip()
        return ""

    def accept(self):
        """Handles the OK button click."""
        logger.info("EditPersonalityDialog accepted.")
        # Optionally, add validation here if needed before accepting.
        super().accept()

    def reject(self):
        """Handles the Cancel button click."""
        logger.info("EditPersonalityDialog rejected (cancelled).")
        super().reject()