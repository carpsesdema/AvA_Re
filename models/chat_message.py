# app/models/chat_message.py
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Union, Dict, Any

try:
    # Attempt to import MessageLoadingState from its new location
    from .message_enums import MessageLoadingState
except ImportError:
    # Fallback if the import fails (e.g., during initial setup or if files are moved)
    from enum import Enum, auto
    class MessageLoadingState(Enum): # type: ignore
        IDLE = auto()
        LOADING = auto()
        COMPLETED = auto()
        ERROR = auto()
    import logging
    logging.getLogger(__name__).warning(
        "ChatMessage: Could not import MessageLoadingState from .message_enums, using fallback."
    )

# Define role constants, often used with ChatMessage
USER_ROLE = "user"
MODEL_ROLE = "model"
SYSTEM_ROLE = "system"
ERROR_ROLE = "error"

@dataclass
class ChatMessage:
    role: str
    parts: List[Union[str, Dict[str, Any]]] # Can be simple text or complex parts (e.g., for images)
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict) # Ensure metadata is always a dict
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    loading_state: MessageLoadingState = MessageLoadingState.IDLE

    def __post_init__(self):
        # Ensure metadata is a dictionary if it was somehow passed as None
        if self.metadata is None:
            self.metadata = {}

    @property
    def text(self) -> str:
        """
        Extracts and concatenates all text content from the 'parts' list.
        Handles parts that are strings or dictionaries with a 'text' key.
        """
        text_parts_list = []
        for part in self.parts:
            if isinstance(part, str):
                text_parts_list.append(part)
            elif isinstance(part, dict) and part.get("type") == "text": # Common for multimodal messages
                text_parts_list.append(part.get("text", ""))
        return "".join(text_parts_list).strip()

    @property
    def has_images(self) -> bool:
        """Checks if any part in the message is an image."""
        return any(isinstance(part, dict) and part.get("type") == "image" for part in self.parts)

    @property
    def image_parts(self) -> List[Dict[str, Any]]:
        """Returns a list of all image parts in the message."""
        return [part for part in self.parts if isinstance(part, dict) and part.get("type") == "image"]

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ChatMessage to a dictionary, ensuring enums are serializable."""
        data = asdict(self)
        if isinstance(self.loading_state, MessageLoadingState):
            data['loading_state'] = self.loading_state.name
        elif self.loading_state is not None: # Should not happen with type hints, but defensive
            data['loading_state'] = str(self.loading_state)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Creates a ChatMessage from a dictionary."""
        # Convert loading_state string back to enum
        loading_state_str = data.get('loading_state')
        if isinstance(loading_state_str, str):
            try:
                data['loading_state'] = MessageLoadingState[loading_state_str]
            except KeyError:
                data['loading_state'] = MessageLoadingState.IDLE # Fallback
        elif loading_state_str is None : # If it was None or missing from dict
             data['loading_state'] = MessageLoadingState.IDLE


        # Ensure metadata is a dict
        if 'metadata' not in data or data['metadata'] is None:
            data['metadata'] = {}


        return cls(**data)