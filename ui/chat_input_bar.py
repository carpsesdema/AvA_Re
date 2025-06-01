# app/ui/chat_input_bar.py
import logging
import os
from typing import Optional, List, Dict, Any  # Added List, Dict, Any for potential image data

from PySide6.QtCore import QSize, Slot, QTimer, Signal as pyqtSignal, Qt  # Added Qt for KeyPressEvent
from PySide6.QtGui import QFont, QIcon, QKeyEvent  # Added QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSizePolicy
)

try:
    from utils import constants
    # Assuming MultilineInputWidget is in the same 'ui' package
    from .multiline_input_widget import MultilineInputWidget
except ImportError as e_cib:
    logging.getLogger(__name__).critical(f"Critical import error in ChatInputBar: {e_cib}", exc_info=True)
    # Fallback types for type hinting
    constants = type("constants", (object,), {"CHAT_FONT_FAMILY": "Arial", "CHAT_FONT_SIZE": 10})  # type: ignore
    MultilineInputWidget = type("MultilineInputWidget", (QWidget,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class ChatInputBar(QWidget):
    """
    A widget providing a multiline text input for chat messages and a send button.
    Handles Enter/Shift+Enter for sending or newlines.
    Emits sendMessageRequested when the user intends to send a message.
    """
    # Emitted when the user clicks send or presses Enter (without Shift)
    # The signal will carry the text and any attached image data (though image handling is TBD here)
    sendMessageRequested = pyqtSignal(str, list)  # text_content, image_data_list

    ACTION_BUTTON_SIZE = QSize(30, 30)  # Slightly larger for better clickability

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatInputBar")  # For styling

        self._multiline_input: Optional[MultilineInputWidget] = None
        self._send_button: Optional[QPushButton] = None

        # State flags to manage send button enabling and send action
        self._is_sending_blocked_momentarily: bool = False  # Prevents rapid multi-sends
        self._is_globally_busy: bool = False  # Reflects if backend or other app parts are busy
        self._is_explicitly_disabled_by_app: bool = False  # If app logic disables input

        self._init_ui()
        self._connect_internal_signals()
        self._update_send_button_state()  # Initial state check

        logger.info("ChatInputBar initialized.")

    def _init_ui(self):
        """Initializes the UI components of the input bar."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)  # Adjusted margins
        main_layout.setSpacing(6)  # Adjusted spacing

        self._multiline_input = MultilineInputWidget(self)
        self._multiline_input.setObjectName("ChatMultilineInput")  # For specific styling
        main_layout.addWidget(self._multiline_input, 1)  # Input takes most space

        self._send_button = QPushButton("Send")  # Text can be replaced by an icon too
        self._send_button.setObjectName("ChatSendButton")
        font_size = getattr(constants, 'CHAT_FONT_SIZE', 10)
        send_button_font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), font_size)
        self._send_button.setFont(send_button_font)
        self._send_button.setToolTip("Send message (Enter)\nAdd newline (Shift+Enter)")
        self._send_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)  # Expanding vertically

        # Set fixed width and allow height to expand with MultilineInputWidget
        # self._send_button.setMinimumHeight(self.ACTION_BUTTON_SIZE.height())
        # A better approach is to let the layout manage heights, or match MultilineInputWidget's dynamic height.
        # For now, let its vertical size policy be Expanding.
        # self._send_button.setFixedWidth(60) # Example fixed width

        # Attempt to load a send icon
        try:
            assets_path = getattr(constants, 'ASSETS_PATH', './assets')
            send_icon_path = os.path.join(assets_path, "send_icon.png")  # Example icon name
            if os.path.exists(send_icon_path):
                self._send_button.setIcon(QIcon(send_icon_path))
                self._send_button.setText("")  # Clear text if icon is present
                self._send_button.setFixedSize(self.ACTION_BUTTON_SIZE)  # Fixed size for icon button
                self._send_button.setIconSize(
                    QSize(self.ACTION_BUTTON_SIZE.width() - 10, self.ACTION_BUTTON_SIZE.height() - 10))
            else:
                logger.warning(f"Send icon not found at {send_icon_path}, using text button.")
                self._send_button.setMinimumSize(60, self.ACTION_BUTTON_SIZE.height())


        except Exception as e_icon:
            logger.error(f"Error setting send button icon: {e_icon}")
            self._send_button.setMinimumSize(60, self.ACTION_BUTTON_SIZE.height())

        main_layout.addWidget(self._send_button)
        self.setLayout(main_layout)

    def _connect_internal_signals(self):
        """Connects signals from internal widgets to handler slots."""
        if self._multiline_input:
            # MultilineInputWidget should emit sendMessageRequested on Enter (handled in its keyPressEvent)
            self._multiline_input.sendMessageRequested.connect(self._trigger_send_message_action)
            self._multiline_input.textChanged.connect(self._update_send_button_state)

        if self._send_button:
            self._send_button.clicked.connect(self._trigger_send_message_action)

    @Slot()
    def _trigger_send_message_action(self):
        """Slot to handle send action from button click or Enter key."""
        if self._is_sending_blocked_momentarily: return
        if self._is_explicitly_disabled_by_app or self._is_globally_busy: return

        text_to_send = self.get_text()
        # TODO: Implement image data attachment if MultilineInputWidget supports it
        image_data_list: List[Dict[str, Any]] = []  # Placeholder
        # if hasattr(self._multiline_input, 'get_attached_image_data'):
        #    image_data_list = self._multiline_input.get_attached_image_data()

        if not text_to_send and not image_data_list:  # Nothing to send
            return

        self._is_sending_blocked_momentarily = True  # Prevent rapid re-sends
        self.sendMessageRequested.emit(text_to_send, image_data_list)

        self.clear_text()  # Clear input after sending
        # Re-enable sending after a short delay
        QTimer.singleShot(100, lambda: setattr(self, '_is_sending_blocked_momentarily', False))

    @Slot(bool)
    def set_globally_busy_state(self, is_busy: bool):
        """
        Slot to be connected to an EventBus signal indicating overall application busy state
        (e.g., LLM processing).
        """
        if self._is_globally_busy == is_busy: return
        self._is_globally_busy = is_busy
        self._update_send_button_state()
        if self._multiline_input:
            # Input field itself is disabled only if explicitly disabled by app,
            # or if globally busy (to prevent typing during critical ops).
            self._multiline_input.set_enabled(not self._is_explicitly_disabled_by_app and not self._is_globally_busy)

    @Slot()
    def _update_send_button_state(self):
        """Updates the enabled state of the send button based on current conditions."""
        if self._send_button and self._multiline_input:
            can_send = (not self._is_explicitly_disabled_by_app and
                        not self._is_globally_busy and
                        not self._is_sending_blocked_momentarily and
                        bool(self._multiline_input.get_text().strip()))  # Check if text edit has content
            # Add 'or self._multiline_input.has_images()' if image sending is implemented
            self._send_button.setEnabled(can_send)

    def get_text(self) -> str:
        """Returns the current text from the input widget."""
        return self._multiline_input.get_text() if self._multiline_input else ""

    def clear_text(self):
        """Clears the text from the input widget."""
        if self._multiline_input:
            self._multiline_input.clear_text()
        # _update_send_button_state will be called automatically due to textChanged signal

    def set_input_focus(self):  # Renamed for clarity
        """Sets focus to the text input widget."""
        if self._multiline_input:
            self._multiline_input.set_focus()

    def set_enabled_state(self, enabled: bool):  # Renamed for clarity
        """
        Sets the overall enabled state of the input bar.
        If disabled, neither typing nor sending is allowed.
        """
        self._is_explicitly_disabled_by_app = not enabled
        # Update actual widget enabled states based on all flags
        self._update_send_button_state()
        if self._multiline_input:
            self._multiline_input.set_enabled(not self._is_explicitly_disabled_by_app and not self._is_globally_busy)
        self.setEnabled(enabled)  # Enable/disable the container widget itself

    # --- Image Data Handling (Placeholder) ---
    def get_attached_image_data(self) -> List[Dict[str, Any]]:
        """
        Placeholder for retrieving attached image data.
        This would be implemented if MultilineInputWidget supports image attachments.
        """
        # Example if MultilineInputWidget had such a method:
        # if hasattr(self._multiline_input, 'get_images'):
        #     return self._multiline_input.get_images()
        return []

    # Override keyPressEvent if needed for global shortcuts within this bar,
    # but MultilineInputWidget handles Enter/Shift+Enter locally.
    # def keyPressEvent(self, event: QKeyEvent):
    #     super().keyPressEvent(event)