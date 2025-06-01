# app/ui/chat_list_model.py
import logging
from typing import List, Optional, Any

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, QObject

try:
    # Corrected paths for models and enums
    from models.chat_message import ChatMessage
    from models.message_enums import MessageLoadingState
except ImportError:
    # Fallback classes for type hinting if core imports fail,
    # allows the file to be parsed but will likely fail at runtime.
    logging.getLogger(__name__).warning(
        "ChatListModel: Could not import ChatMessage or MessageLoadingState from models. Using fallbacks."
    )


    class ChatMessage:  # type: ignore
        def __init__(self, id=None, role=None, parts=None, loading_state=None, timestamp=None, metadata=None):
            self.id = id
            self.role = role
            self.parts = parts if parts is not None else []
            self.loading_state = loading_state
            self.timestamp = timestamp
            self.metadata = metadata if metadata is not None else {}

        @property
        def text(self): return str(self.parts[0]) if self.parts and isinstance(self.parts[0], str) else ""


    from enum import Enum, auto


    class MessageLoadingState(Enum):  # type: ignore
        IDLE = auto()
        LOADING = auto()
        COMPLETED = auto()
        ERROR = auto()

logger = logging.getLogger(__name__)

# Custom Roles for QAbstractListModel
ChatMessageRole = Qt.ItemDataRole.UserRole + 1  # Stores the actual ChatMessage object
LoadingStatusRole = Qt.ItemDataRole.UserRole + 2  # Stores the MessageLoadingState enum


class ChatListModel(QAbstractListModel):
    """
    A Qt model for managing and displaying a list of ChatMessage objects
    in a QListView. It handles adding, updating, and clearing messages,
    and notifies views of these changes.
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._messages: List[ChatMessage] = []
        logger.info("ChatListModel initialized.")

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Returns the number of messages in the model."""
        return 0 if parent.isValid() else len(self._messages)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Provides data for a given index and role."""
        if not index.isValid() or not (0 <= index.row() < len(self._messages)):
            return None

        message = self._messages[index.row()]

        if role == ChatMessageRole:
            return message  # Return the full ChatMessage object for the delegate
        elif role == Qt.ItemDataRole.DisplayRole:
            # Basic text for roles like ToolTipRole or simple list views (not typically used by delegate)
            text_preview = message.text[:50] + "..." if len(message.text) > 50 else message.text
            return f"[{message.role}] {text_preview}"
        elif role == LoadingStatusRole:
            return message.loading_state

        # Add other roles if needed by the delegate, e.g., Qt.ToolTipRole
        # elif role == Qt.ToolTipRole:
        #     return f"ID: {message.id}\nTimestamp: {message.timestamp}"

        return None

    def addMessage(self, message: ChatMessage):
        """Adds a new message to the end of the list."""
        if not isinstance(message, ChatMessage):
            logger.error(f"Attempted to add invalid type to ChatListModel: {type(message)}")
            return

        row_to_insert = len(self._messages)
        self.beginInsertRows(QModelIndex(), row_to_insert, row_to_insert)
        self._messages.append(message)
        self.endInsertRows()
        logger.debug(f"ChatListModel: Added message ID {message.id}, new count {len(self._messages)}")

    def append_chunk_to_message_by_id(self, message_id: str, chunk: str) -> bool:
        """Appends a text chunk to an existing message identified by its ID."""
        if not isinstance(message_id, str) or not message_id or not isinstance(chunk, str):
            logger.warning(
                f"ChatListModel: Invalid arguments for append_chunk. ID: {message_id}, Chunk: {chunk[:20]}...")
            return False

        row = self.find_message_row_by_id(message_id)
        if row is None:
            logger.warning(f"ChatListModel: Message with ID '{message_id}' not found to append chunk.")
            return False

        message_to_update = self._messages[row]

        # Ensure parts list exists and handle text part
        if not hasattr(message_to_update, 'parts') or not isinstance(message_to_update.parts, list):
            message_to_update.parts = []  # Initialize if missing

        text_part_index = -1
        current_text = ""
        for i, part_content in enumerate(message_to_update.parts):
            if isinstance(part_content, str):  # Simple text part
                current_text = part_content
                text_part_index = i
                break
            elif isinstance(part_content, dict) and part_content.get("type") == "text":  # Text part in dict
                current_text = part_content.get("text", "")
                text_part_index = i
                break

        updated_text = current_text + chunk

        if text_part_index != -1:  # Update existing text part
            if isinstance(message_to_update.parts[text_part_index], str):
                message_to_update.parts[text_part_index] = updated_text
            elif isinstance(message_to_update.parts[text_part_index], dict):
                message_to_update.parts[text_part_index]["text"] = updated_text
        else:  # No existing text part, add a new one (typically at the beginning for display)
            message_to_update.parts.insert(0, updated_text)

        # Update loading state if it's still IDLE (first chunk for a loading message)
        if message_to_update.loading_state == MessageLoadingState.IDLE:
            message_to_update.loading_state = MessageLoadingState.LOADING

        model_idx = self.index(row, 0)
        # Notify view that data (ChatMessageRole for content, LoadingStatusRole for potential state change) has changed.
        self.dataChanged.emit(model_idx, model_idx, [ChatMessageRole, LoadingStatusRole, Qt.ItemDataRole.DisplayRole])
        return True

    def finalize_message_by_id(self, message_id: str, final_message_obj: Optional[ChatMessage] = None,
                               is_error: bool = False):
        """
        Finalizes a message, typically updating its content and setting its loading state
        to COMPLETED or ERROR. If final_message_obj is provided, it replaces the existing one.
        """
        row = self.find_message_row_by_id(message_id)
        if row is None:
            logger.warning(f"ChatListModel: Message with ID '{message_id}' not found to finalize.")
            return

        message_to_finalize = self._messages[row]

        if final_message_obj and isinstance(final_message_obj, ChatMessage):
            # Replace the entire message object, ensuring the ID matches for consistency
            # The final_message_obj should ideally have the same ID as the placeholder it's replacing.
            if final_message_obj.id != message_id:
                logger.warning(
                    f"ChatListModel: Finalizing message ID '{message_id}' with object having different ID '{final_message_obj.id}'. Using ID from final_message_obj.")
            self._messages[row] = final_message_obj
            message_to_finalize = self._messages[row]  # Update reference

        message_to_finalize.loading_state = MessageLoadingState.ERROR if is_error else MessageLoadingState.COMPLETED
        if hasattr(message_to_finalize, 'metadata') and message_to_finalize.metadata is not None:
            message_to_finalize.metadata.pop("is_streaming", None)  # Remove streaming indicator

        model_idx = self.index(row, 0)
        self.dataChanged.emit(model_idx, model_idx, [ChatMessageRole, LoadingStatusRole, Qt.ItemDataRole.DisplayRole])
        logger.debug(
            f"ChatListModel: Finalized message ID {message_id}. State: {message_to_finalize.loading_state.name}")

    def update_message_loading_state_by_id(self, message_id: str, new_state: MessageLoadingState) -> bool:
        """Updates only the loading state of a message, primarily for visual indicators."""
        if not isinstance(message_id, str) or not message_id: return False
        row = self.find_message_row_by_id(message_id)
        if row is not None:
            message = self._messages[row]
            if message.loading_state != new_state:
                message.loading_state = new_state
                model_index = self.index(row, 0)
                self.dataChanged.emit(model_index, model_index, [LoadingStatusRole])
                logger.debug(f"ChatListModel: Updated loading state for message ID {message_id} to {new_state.name}")
                return True
        else:
            logger.warning(f"ChatListModel: Message ID '{message_id}' not found for state update to {new_state.name}.")
        return False

    def find_message_row_by_id(self, message_id: str) -> Optional[int]:
        """Finds the row index of a message given its ID."""
        if not isinstance(message_id, str) or not message_id: return None
        for row_num, msg in enumerate(self._messages):
            if msg.id == message_id:
                return row_num
        return None

    def loadHistory(self, history: List[ChatMessage]):
        """Replaces the entire message list with a new history and notifies the view."""
        self.beginResetModel()
        self._messages = list(history) if history else []
        self.endResetModel()
        logger.info(f"ChatListModel: Loaded history with {len(self._messages)} messages.")

    def clearMessages(self):
        """Clears all messages from the model and notifies the view."""
        if not self._messages: return  # Nothing to clear
        self.beginResetModel()
        self._messages = []
        self.endResetModel()
        logger.info("ChatListModel: All messages cleared.")

    def getMessage(self, row: int) -> Optional[ChatMessage]:
        """Retrieves a message at a specific row index."""
        if 0 <= row < len(self._messages):
            return self._messages[row]
        logger.warning(f"ChatListModel: Attempted to get message at invalid row: {row}")
        return None

    def getAllMessages(self) -> List[ChatMessage]:
        """Returns a copy of all messages currently in the model."""
        return list(self._messages)  # Return a copy