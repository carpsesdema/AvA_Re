# app/ui/chat_terminal.py
import datetime
import logging
from typing import Optional, List  # Added Dict, Any

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QKeyEvent, QTextCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit

try:
    from core.event_bus import EventBus
    # ChatManager is optional for now, terminal might interact primarily via EventBus
    from core.chat_manager import ChatManager  # For type hinting
    from models.chat_message import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from utils import constants
except ImportError as e_ct:
    logging.getLogger(__name__).critical(f"ChatTerminal: Critical import error: {e_ct}", exc_info=True)
    EventBus = type("EventBus", (object,), {})  # type: ignore
    ChatManager = type("ChatManager", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "model", "system", "error"
    constants = type("constants", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class ChatTerminal(QWidget):
    """
    A terminal-like interface for interacting with the chat system.
    Displays messages in a text area and provides a command-line input.
    Can also display messages from the main chat flow if connected to appropriate signals.
    """

    # Emitted when the user submits text from the terminal's input line.
    # The boolean indicates if it's likely a command (starts with /) or a chat message.
    terminalInputSubmitted = Signal(str, bool)  # input_text, is_command_attempt

    def __init__(self,
                 event_bus: EventBus,
                 # ChatManager can be optional; if provided, it could be used for direct interactions
                 # or this terminal can be fully decoupled and only use EventBus.
                 chat_manager: Optional[ChatManager] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatTerminalWidget")
        self.setWindowTitle(f"{getattr(constants, 'APP_NAME', 'AvA')} - Chat Terminal")

        if not isinstance(event_bus, EventBus):  # type: ignore
            raise TypeError("ChatTerminal requires a valid EventBus.")

        self._event_bus = event_bus
        self._chat_manager = chat_manager  # Store if provided

        self._display_area: Optional[QTextEdit] = None
        self._input_line: Optional[QLineEdit] = None

        self._command_history: List[str] = []
        self._history_index: int = -1  # -1 means current input is new, not from history

        self._init_ui()
        self._connect_signals()

        # To make the terminal display general application messages or specific session chat,
        # it needs to listen to appropriate EventBus signals.
        # For example, to mirror the active chat session:
        if self._chat_manager and hasattr(self._chat_manager, 'get_chat_session_manager'):
            session_manager = self._chat_manager.get_chat_session_manager()
            if session_manager:
                session_manager.newMessageAddedToActiveSession.connect(self._display_chat_message_from_session)
                session_manager.activeSessionChanged.connect(self._handle_active_session_changed)
                session_manager.activeSessionCleared.connect(self._handle_active_session_cleared)
        else:  # Fallback or if only for system messages
            self._event_bus.uiStatusUpdateGlobal.connect(self._display_global_status_update)

        logger.info("ChatTerminal initialized.")
        self.add_message_to_display(SYSTEM_ROLE, "Chat Terminal Initialized. Type '/help' for commands.")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)  # Added some margins
        layout.setSpacing(6)

        self._display_area = QTextEdit(self)
        self._display_area.setObjectName("TerminalDisplayArea")
        self._display_area.setReadOnly(True)
        font_family = getattr(constants, 'CHAT_FONT_FAMILY', "Monospace")  # Monospace is good for terminals
        font_size = getattr(constants, 'CHAT_FONT_SIZE', 10)
        self._display_area.setFont(QFont(font_family, font_size))
        # Using a slightly different style for terminal display
        self._display_area.setStyleSheet(
            "QTextEdit#TerminalDisplayArea {"
            "  background-color: #0c0c0c;"  # Very dark background
            "  color: #00E000;"  # Classic green terminal text
            "  border: 1px solid #222222;"
            "  font-family: 'Consolas', 'Courier New', Monospace;"  # Ensure monospace
            "  padding: 5px;"
            "}"
        )

        self._input_line = QLineEdit(self)
        self._input_line.setObjectName("TerminalInputLine")
        self._input_line.setFont(QFont(font_family, font_size))
        self._input_line.setPlaceholderText("Enter command (e.g., /help) or message...")
        self._input_line.setStyleSheet(
            "QLineEdit#TerminalInputLine {"
            "  background-color: #1a1a1a;"
            "  color: #00FF00;"  # Bright green input text
            "  border: 1px solid #333333;"
            "  padding: 4px;"
            "  font-family: 'Consolas', 'Courier New', Monospace;"
            "}"
            "QLineEdit#TerminalInputLine:focus {"
            "  border: 1px solid #00E000;"  # Green border on focus
            "}"
        )

        layout.addWidget(self._display_area, 1)
        layout.addWidget(self._input_line)
        self.setLayout(layout)
        self.setMinimumSize(600, 350)  # Slightly smaller min size

    def _connect_signals(self):
        if self._input_line:
            self._input_line.returnPressed.connect(self._process_input_line_submission)
        # Key press events for history are handled in self.keyPressEvent override for the QLineEdit

    @Slot()
    def _process_input_line_submission(self):
        if not self._input_line: return
        input_text = self._input_line.text().strip()
        if not input_text: return

        # Echo user input to the terminal display itself
        self.add_message_to_display(USER_ROLE, f"{input_text}")  # Simple echo, prefix added by add_message_to_display

        if input_text and (not self._command_history or input_text != self._command_history[-1]):
            self._command_history.append(input_text)
        self._history_index = len(self._command_history)  # Reset history index for next up/down navigation

        is_command = input_text.startswith("/")

        # Emit signal for ChatManager or other listeners to process
        self.terminalInputSubmitted.emit(input_text, is_command)

        if is_command:
            self._handle_local_terminal_command(input_text)  # Handle some basic local commands

        self._input_line.clear()

    def _handle_local_terminal_command(self, command_text: str):
        """Handles basic commands local to the terminal window itself."""
        command_parts = command_text.lower().split()
        cmd = command_parts[0]

        if cmd == "/help":
            self.add_message_to_display(SYSTEM_ROLE,
                                        "Local Terminal Commands:\n"
                                        "  /help          - Show this help message.\n"
                                        "  /clear         - Clear the terminal display.\n"
                                        "  /history       - Show command history.\n"
                                        "Any other input is treated as a message to the main chat system."
                                        )
        elif cmd == "/clear":
            if self._display_area: self._display_area.clear()
            self.add_message_to_display(SYSTEM_ROLE, "Terminal display cleared.")
        elif cmd == "/history":
            if self._command_history:
                hist_text = "Command History:\n" + "\n".join(
                    [f"  {i + 1:2d}. {cmd_item}" for i, cmd_item in enumerate(self._command_history)]
                )
                self.add_message_to_display(SYSTEM_ROLE, hist_text)
            else:
                self.add_message_to_display(SYSTEM_ROLE, "No command history recorded yet.")
        # else: command not handled locally, already emitted via terminalInputSubmitted

    def add_message_to_display(self, role: str, text: str, timestamp_str: Optional[str] = None):
        """Appends a formatted message to the terminal display area."""
        if not self._display_area: return

        now_str = timestamp_str or datetime.now().strftime("%H:%M:%S")

        # Determine color and prefix based on role
        role_colors = {
            USER_ROLE: "#39D353",  # Green
            MODEL_ROLE: "#00E0E0",  # Cyan/Aqua for AI
            SYSTEM_ROLE: "#9E9E9E",  # Grey
            ERROR_ROLE: "#F85149",  # Red
            "DEBUG": "#FFB74D"  # Orange for debug/internal if needed
        }
        role_prefixes = {
            USER_ROLE: f"👤 [{now_str}] User:",
            MODEL_ROLE: f"🤖 [{now_str}] AvA:",
            SYSTEM_ROLE: f"⚙️ [{now_str}] System:",
            ERROR_ROLE: f"⚠️ [{now_str}] Error:",
            "DEBUG": f"🐞 [{now_str}] Debug:"
        }

        color = role_colors.get(role, "#00E000")  # Default to terminal green
        prefix = role_prefixes.get(role, f"💬 [{now_str}] {role}:")

        # Escape HTML in the text part to prevent injection if text comes from untrusted source
        escaped_text = text.replace("<", "<").replace(">", ">")

        # Construct HTML for the line
        # Using a table-like structure with no borders for alignment
        formatted_line = (
            f'<div style="line-height: 1.3;">'
            f'<span style="color: {color}; font-weight: bold; white-space: pre;">{prefix: <18}</span>'  # Fixed width for prefix
            f'<span style="color: {color}; white-space: pre-wrap;">{escaped_text}</span>'
            f'</div>'
        )
        self._display_area.append(formatted_line)
        self._display_area.moveCursor(QTextCursor.MoveOperation.End)  # Scroll to bottom

    @Slot(str, str, object)  # project_id, session_id, ChatMessage object
    def _display_chat_message_from_session(self, project_id: str, session_id: str,
                                           message: ChatMessage):  # type: ignore
        """Displays a ChatMessage from the main chat session if this terminal is mirroring it."""
        # This assumes ChatManager is configured to know which project/session this terminal instance cares about.
        # For simplicity, if this slot is connected, we display the message.
        # A more advanced terminal might have a "current_context_to_mirror" state.
        if isinstance(message, ChatMessage):  # type: ignore
            self.add_message_to_display(message.role, message.text, message.timestamp)

    @Slot(str, str, list)  # project_id, session_id, history_list
    def _handle_active_session_changed(self, project_id: str, session_id: str,
                                       history: List[ChatMessage]):  # type: ignore
        if self._display_area: self._display_area.clear()
        self.add_message_to_display(SYSTEM_ROLE, f"Terminal now viewing session: P:{project_id[:8]}/S:{session_id[:8]}")
        for msg in history:
            if isinstance(msg, ChatMessage):  # type: ignore
                self.add_message_to_display(msg.role, msg.text, msg.timestamp)

    @Slot(str, str)  # old_project_id, old_session_id
    def _handle_active_session_cleared(self, project_id: str, session_id: str):
        if self._display_area: self._display_area.clear()
        self.add_message_to_display(SYSTEM_ROLE,
                                    f"Active session P:{project_id[:8]}/S:{session_id[:8]} cleared. Terminal unlinked.")

    @Slot(str, str, bool, int)  # message, color_hex, is_temporary, duration_ms
    def _display_global_status_update(self, message: str, color_hex: str, is_temporary: bool, duration_ms: int):
        """Displays global status updates in the terminal if not mirroring a specific session."""
        # This could be made conditional, e.g., only if not actively mirroring a chat.
        # For now, let's assume it always shows them.
        self.add_message_to_display(SYSTEM_ROLE, f"Status: {message}")

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key presses for command history in the input line if it has focus."""
        if self._input_line and self._input_line.hasFocus():
            key = event.key()
            if key == Qt.Key.Key_Up:
                if self._command_history:
                    self._history_index -= 1
                    if self._history_index < 0: self._history_index = len(self._command_history) - 1
                    if 0 <= self._history_index < len(self._command_history):
                        self._input_line.setText(self._command_history[self._history_index])
                        self._input_line.selectAll()  # Optional: select text for easy overwrite
                event.accept()
                return
            elif key == Qt.Key.Key_Down:
                if self._command_history:
                    self._history_index += 1
                    if self._history_index >= len(self._command_history):
                        self._history_index = -1  # Indicates new input after cycling through
                        self._input_line.clear()
                    elif 0 <= self._history_index < len(self._command_history):
                        self._input_line.setText(self._command_history[self._history_index])
                        self._input_line.selectAll()
                event.accept()
                return
        super().keyPressEvent(event)  # Pass to parent if not handled

    def show_terminal(self):
        """Shows the terminal window and sets focus to input."""
        self.show()
        self.activateWindow()  # Bring to front
        self.raise_()
        if self._input_line:
            QTimer.singleShot(0, self._input_line.setFocus)  # Ensure focus after shown