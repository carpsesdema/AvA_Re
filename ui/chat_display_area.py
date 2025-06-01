# app/ui/chat_display_area.py
import logging
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Slot, QPoint, Signal, QModelIndex
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListView, QAbstractItemView, QMenu, QApplication
)

try:
    # Corrected paths assuming these are in the same 'ui' package or accessible
    from .chat_item_delegate import ChatItemDelegate
    from .chat_list_model import ChatListModel, ChatMessageRole  # ChatMessageRole is used here
    # Models are in models
    from models.chat_message import ChatMessage, SYSTEM_ROLE, ERROR_ROLE
    # Enums for message state
    from models.message_enums import MessageLoadingState
except ImportError as e_cda:
    logging.getLogger(__name__).critical(f"Critical import error in ChatDisplayArea: {e_cda}", exc_info=True)
    # Fallback types for type hinting
    ChatItemDelegate = type("ChatItemDelegate", (QAbstractItemView.itemDelegate,), {})  # type: ignore
    ChatListModel = type("ChatListModel", (QAbstractItemView.model,), {})  # type: ignore
    ChatMessageRole = Qt.ItemDataRole.UserRole + 100  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    SYSTEM_ROLE, ERROR_ROLE = "system", "error"
    MessageLoadingState = type("MessageLoadingState", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class ChatDisplayArea(QWidget):
    """
    A widget that displays chat messages using a QListView,
    custom model (ChatListModel), and a custom delegate (ChatItemDelegate).
    It handles adding messages, scrolling, and context menus for messages.
    """
    # Emitted when message text is copied from the context menu
    textCopied = Signal(str, str)  # message_text, status_color_hex

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatDisplayAreaWidget")  # For styling

        self.chat_list_view: Optional[QListView] = None
        self.chat_list_model: Optional[ChatListModel] = None
        self.chat_item_delegate: Optional[ChatItemDelegate] = None

        # Stores the project and session ID this display area is currently showing.
        # This helps in filtering events if multiple ChatDisplayAreas were ever used (though unlikely for now).
        self._current_project_id: Optional[str] = None
        self._current_session_id: Optional[str] = None

        self._init_ui()
        logger.info("ChatDisplayArea initialized.")

    def _init_ui(self):
        """Initializes the UI components of the chat display area."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)  # No external margins for the layout itself
        outer_layout.setSpacing(0)

        self.chat_list_view = QListView(self)
        self.chat_list_view.setObjectName("ChatListView")  # For styling the view
        self.chat_list_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Usually chat display isn't directly focusable
        self.chat_list_view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)  # Allow text selection from bubbles
        self.chat_list_view.setResizeMode(QListView.ResizeMode.Adjust)  # Adjust items on resize
        self.chat_list_view.setUniformItemSizes(False)  # Bubbles have varying heights
        self.chat_list_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)  # Smooth scrolling
        self.chat_list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # No horizontal scroll
        self.chat_list_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chat_list_view.setWordWrap(True)  # Ensure delegate's text document can wrap

        # Each ChatDisplayArea instance gets its own model and delegate
        self.chat_list_model = ChatListModel(parent=self)
        self.chat_list_view.setModel(self.chat_list_model)

        self.chat_item_delegate = ChatItemDelegate(parent=self)
        self.chat_item_delegate.setView(self.chat_list_view)  # Pass view reference to delegate for updates
        self.chat_list_view.setItemDelegate(self.chat_item_delegate)

        # Context menu for copying message text
        self.chat_list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list_view.customContextMenuRequested.connect(self._show_message_context_menu)

        outer_layout.addWidget(self.chat_list_view)
        self.setLayout(outer_layout)

    def set_current_context(self, project_id: str, session_id: str):
        """
        Informs the display area which project/session it is currently displaying.
        This is important if events need to be filtered for this specific context.
        """
        logger.debug(f"CDA: Current context set to P:{project_id}/S:{session_id}")
        self._current_project_id = project_id
        self._current_session_id = session_id
        # Typically, after context is set, history is loaded, which is handled by load_history_into_model.

    # --- Model Interaction Methods ---
    # These methods are typically called by MainWindow in response to EventBus signals
    # from ChatSessionManager or other core components.

    @Slot(str, str, ChatMessage)  # type: ignore
    def add_message_to_display(self, project_id: str, session_id: str, message: ChatMessage):  # type: ignore
        """Adds a single message to the display if it matches the current context."""
        if self.chat_list_model and project_id == self._current_project_id and session_id == self._current_session_id:
            self.chat_list_model.addMessage(message)
            self._scroll_to_bottom_if_needed()  # Scroll after adding a new message
        else:
            logger.debug(
                f"CDA: Ignored add_message for P:{project_id}/S:{session_id} (current: P:{self._current_project_id}/S:{self._current_session_id})")

    @Slot(str, str, str, str)  # project_id, session_id, request_id (maps to message_id), chunk_text
    def append_chunk_to_message_display(self, project_id: str, session_id: str, message_id: str, chunk_text: str):
        """Appends a chunk of text to an existing message in the display."""
        if self.chat_list_model and project_id == self._current_project_id and session_id == self._current_session_id:
            success = self.chat_list_model.append_chunk_to_message_by_id(message_id, chunk_text)
            if success:
                self._scroll_to_bottom_if_needed(is_streaming=True)
        # else: logger.debug("CDA: Ignored append_chunk for non-active context.")

    @Slot(str, str, str, ChatMessage,
          bool)  # type: ignore # project_id, session_id, message_id, final_message_obj, is_error
    def finalize_message_display(self, project_id: str, session_id: str, message_id: str,
                                 final_message_obj: Optional[ChatMessage], is_error: bool):  # type: ignore
        """Finalizes a message in the display (e.g., sets content and completed/error state)."""
        if self.chat_list_model and project_id == self._current_project_id and session_id == self._current_session_id:
            self.chat_list_model.finalize_message_by_id(message_id, final_message_obj, is_error)
            self._scroll_to_bottom_if_needed()  # Ensure final state is visible
        # else: logger.debug("CDA: Ignored finalize_message for non-active context.")

    @Slot(str, str)  # project_id, session_id
    def clear_display(self, project_id: str, session_id: str):
        """Clears all messages from the display if it matches the current context."""
        if self.chat_list_model and project_id == self._current_project_id and session_id == self._current_session_id:
            logger.info(f"CDA: Clearing model for active P:{project_id}/S:{session_id}")
            self.chat_list_model.clearMessages()
            if self.chat_item_delegate: self.chat_item_delegate.clearCache()
        else:
            logger.debug(
                f"CDA: Ignored clear_model for P:{project_id}/S:{session_id} (current: P:{self._current_project_id}/S:{self._current_session_id})")

    @Slot(str, str, list)  # project_id, session_id, history_list[ChatMessage]
    def load_history_into_display(self, project_id: str, session_id: str, history: List[ChatMessage]):  # type: ignore
        """Loads a complete chat history into the display, replacing current content."""
        # This method is called when switching sessions or loading a project.
        self.set_current_context(project_id, session_id)  # Ensure current context is updated
        if self.chat_list_model:
            logger.info(f"CDA: Loading history ({len(history)} msgs) for P:{project_id}/S:{session_id}")
            self.chat_list_model.loadHistory(history)
            if self.chat_item_delegate: self.chat_item_delegate.clearCache()
            self._scroll_to_bottom_delayed()  # Scroll after loading full history, with a slight delay for rendering

    # --- Scrolling ---
    def _scroll_to_bottom_if_needed(self, is_streaming: bool = False):
        """Scrolls to the bottom if user is already near the bottom or not streaming."""
        if not self.chat_list_view or not self.chat_list_model or self.chat_list_model.rowCount() == 0:
            return

        v_scrollbar = self.chat_list_view.verticalScrollBar()
        if v_scrollbar:
            should_scroll_now = True
            if is_streaming:
                # Only auto-scroll during streaming if user hasn't scrolled up significantly
                scroll_threshold = v_scrollbar.pageStep() // 2  # Heuristic: half a page
                if v_scrollbar.value() < v_scrollbar.maximum() - scroll_threshold:
                    should_scroll_now = False

            if should_scroll_now:
                # QTimer.singleShot(0, self.chat_list_view.scrollToBottom) # Scroll on next event loop iteration
                self.chat_list_view.scrollToBottom()  # Try immediate scroll

    def _scroll_to_bottom_delayed(self):
        """Ensures scrolling to bottom after items have likely rendered."""
        if self.chat_list_view:
            QTimer.singleShot(50, self.chat_list_view.scrollToBottom)  # 50ms delay

    # --- Context Menu ---
    @Slot(QPoint)
    def _show_message_context_menu(self, pos: QPoint):
        """Shows a context menu for a chat message item (e.g., to copy text)."""
        if not self.chat_list_view or not self.chat_list_model: return

        index = self.chat_list_view.indexAt(pos)
        if not index.isValid(): return

        message: Optional[ChatMessage] = self.chat_list_model.data(index, ChatMessageRole)  # type: ignore
        if isinstance(message, ChatMessage) and message.text and message.text.strip() and \
                message.role not in [SYSTEM_ROLE,
                                     ERROR_ROLE]:  # Don't offer copy for system/error messages without useful text
            context_menu = QMenu(self)
            copy_action = context_menu.addAction("Copy Message Text")
            # Lambda captures current message text for the action
            copy_action.triggered.connect(
                lambda checked=False, msg_text=message.text: self._copy_message_text_to_clipboard(msg_text))
            context_menu.exec(self.chat_list_view.mapToGlobal(pos))

    def _copy_message_text_to_clipboard(self, text_to_copy: str):
        """Copies the given text to the system clipboard."""
        try:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(text_to_copy)
                # Emit signal for UI feedback (e.g., status bar message)
                self.textCopied.emit("Message text copied!", "#98c379")  # Green color for success
                logger.info("Message text copied to clipboard.")
        except Exception as e:
            logger.error(f"Error copying text to clipboard: {e}", exc_info=True)
            self.textCopied.emit(f"Error copying: {str(e)[:50]}...", "#e06c75")  # Red color for error

    # --- Getters ---
    def get_model(self) -> Optional[ChatListModel]:
        return self.chat_list_model

    def get_current_context(self) -> Tuple[Optional[str], Optional[str]]:
        return self._current_project_id, self._current_session_id