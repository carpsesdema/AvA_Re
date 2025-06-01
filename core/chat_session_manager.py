# app/core/chat_session_manager.py
import logging
from typing import List, Optional, Tuple, Any

from PySide6.QtCore import QObject, Signal, Slot

try:
    # Corrected path for models
    from models.chat_message import ChatMessage
    from models.message_enums import MessageLoadingState  # For type hinting if message objects have it
    from core.event_bus import EventBus  # For emitting app-wide notifications if needed by other modules
except ImportError as e_csm:
    logging.getLogger(__name__).critical(f"ChatSessionManager: Critical import error: {e_csm}", exc_info=True)
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MessageLoadingState = type("MessageLoadingState", (object,), {})  # type: ignore
    EventBus = type("EventBus", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class ChatSessionManager(QObject):
    """
    Manages the state of the currently active chat session.
    This includes the current project ID, session ID, and the message history
    for that active session. It emits signals when the active session's
    data or state changes, allowing UI components to react.
    """

    # Emitted when the active session (project_id, session_id) changes.
    # The `initial_history` is provided for the UI to load.
    activeSessionChanged = Signal(str, str, list)  # project_id, session_id, initial_history_list[ChatMessage]

    # Emitted when the active session is cleared (e.g., no session is active).
    # Sends the IDs of the previously active session.
    activeSessionCleared = Signal(str, str)  # old_project_id, old_session_id

    # Emitted when a new message is added to the *currently active* session.
    newMessageAddedToActiveSession = Signal(str, str, object)  # project_id, session_id, ChatMessage object

    # Emitted when an existing message in the *active session* is updated.
    # The message_id helps the UI pinpoint which item to refresh.
    messageInActiveSessionUpdated = Signal(str, str, str,
                                           object)  # project_id, session_id, message_id, updated ChatMessage object

    # Emitted when the entire history of the *active session* is replaced or significantly refreshed.
    activeSessionHistoryRefreshed = Signal(str, str, list)  # project_id, session_id, full_new_history_list[ChatMessage]

    def __init__(self, event_bus: EventBus, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._event_bus = event_bus  # Retain for potential future use, though direct signals are preferred from CSM
        self._current_project_id: Optional[str] = None
        self._current_session_id: Optional[str] = None
        self._current_chat_history: List[ChatMessage] = []  # type: ignore

        logger.info("ChatSessionManager initialized.")

    def set_active_session(self, project_id: str, session_id: str, history: List[ChatMessage]):  # type: ignore
        """
        Sets the new active chat session and loads its history.
        Emits activeSessionChanged if the session truly changes, or
        activeSessionHistoryRefreshed if it's the same session but history is updated.
        """
        if not project_id or not session_id:
            logger.error("CSM: Attempted to set active session with invalid project_id or session_id.")
            return

        logger.info(f"CSM: Setting active session to P:{project_id}/S:{session_id}. History items: {len(history)}")

        if self._current_project_id == project_id and self._current_session_id == session_id:
            logger.debug(f"CSM: Session P:{project_id}/S:{session_id} is already active. Refreshing history.")
            self._current_chat_history = list(history)  # type: ignore
            self.activeSessionHistoryRefreshed.emit(project_id, session_id, self._current_chat_history)
        else:
            old_pid, old_sid = self._current_project_id, self._current_session_id

            self._current_project_id = project_id
            self._current_session_id = session_id
            self._current_chat_history = list(history)  # type: ignore

            if old_pid and old_sid:  # If there was a previously active session
                self.activeSessionCleared.emit(old_pid, old_sid)

            self.activeSessionChanged.emit(project_id, session_id, self._current_chat_history)

    def clear_active_session(self):
        """Clears the currently active session data and emits activeSessionCleared."""
        if self._current_project_id and self._current_session_id:
            pid, sid = self._current_project_id, self._current_session_id
            self._current_project_id = None
            self._current_session_id = None
            self._current_chat_history = []
            logger.info(f"CSM: Active session P:{pid}/S:{sid} cleared.")
            self.activeSessionCleared.emit(pid, sid)
        else:
            logger.debug("CSM: No active session to clear.")

    def add_message(self, message: ChatMessage) -> bool:  # type: ignore
        """
        Adds a message to the current active session's history.
        Returns True if successful, False otherwise.
        """
        if not self._current_project_id or not self._current_session_id:
            logger.warning("CSM: Cannot add message, no active session.")
            return False
        if not isinstance(message, ChatMessage):  # type: ignore
            logger.error(f"CSM: Attempted to add invalid message type: {type(message)}")
            return False

        self._current_chat_history.append(message)
        logger.debug(
            f"CSM: Added message (ID: {message.id}) to P:{self._current_project_id}/S:{self._current_session_id}")
        self.newMessageAddedToActiveSession.emit(self._current_project_id, self._current_session_id, message)
        return True

    def update_message_in_history(self, updated_message: ChatMessage) -> bool:  # type: ignore
        """
        Updates an existing message in the history by its ID.
        Emits messageInActiveSessionUpdated if successful.
        Returns True if successful, False otherwise.
        """
        if not self._current_project_id or not self._current_session_id:
            logger.warning("CSM: Cannot update message, no active session.")
            return False
        if not hasattr(updated_message, 'id') or not updated_message.id:  # type: ignore
            logger.error("CSM: Message to update has no ID attribute or ID is empty.")
            return False

        for i, msg in enumerate(self._current_chat_history):
            if hasattr(msg, 'id') and msg.id == updated_message.id:  # type: ignore
                self._current_chat_history[i] = updated_message
                logger.debug(
                    f"CSM: Updated message (ID: {updated_message.id}) in P:{self._current_project_id}/S:{self._current_session_id}")
                self.messageInActiveSessionUpdated.emit(self._current_project_id, self._current_session_id,
                                                        updated_message.id, updated_message)  # type: ignore
                return True
        logger.warning(
            f"CSM: Message with ID {updated_message.id} not found in active session P:{self._current_project_id}/S:{self._current_session_id} for update.")  # type: ignore
        return False

    def get_message_by_id(self, message_id: str) -> Optional[ChatMessage]:  # type: ignore
        """Retrieves a message from the current history by its ID."""
        if not self._current_chat_history or not message_id:
            return None
        for msg in self._current_chat_history:
            if hasattr(msg, 'id') and msg.id == message_id:  # type: ignore
                return msg
        return None

    def get_current_chat_history(self) -> List[ChatMessage]:  # type: ignore
        """Returns a *copy* of the current active chat history to prevent external modification."""
        return list(self._current_chat_history)

    def get_current_project_id(self) -> Optional[str]:
        return self._current_project_id

    def get_current_session_id(self) -> Optional[str]:
        return self._current_session_id

    def get_active_session_context(self) -> Optional[Tuple[str, str]]:
        """Returns (project_id, session_id) if a session is active, else None."""
        if self._current_project_id and self._current_session_id:
            return self._current_project_id, self._current_session_id
        return None