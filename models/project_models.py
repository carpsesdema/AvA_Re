# app/models/project_models.py
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    # Assuming ChatMessage is in the same 'models' package
    from .chat_message import ChatMessage
except ImportError:
    # Fallback for type hinting if ChatMessage is not available at this point
    # This might happen if files are being created out of order or in a flat structure temporarily
    import logging
    logging.getLogger(__name__).warning(
        "ProjectModels: Could not import ChatMessage from .chat_message. Using a fallback type."
    )
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore


@dataclass
class Project:
    id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_session_id: Optional[str] = None
    # Add any other project-specific metadata fields here
    # Example: last_accessed: Optional[str] = None
    # Example: project_path_on_disk: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        return cls(**data)


@dataclass
class ChatSession:
    id: str
    project_id: str # Link back to the parent project
    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    message_history: List[ChatMessage] = field(default_factory=list) # type: ignore
    # Add any other session-specific metadata fields here
    # Example: last_modified: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    # Example: summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure ChatMessage objects are correctly serialized using their own to_dict method
        data['message_history'] = [
            msg.to_dict() if hasattr(msg, 'to_dict') else msg
            for msg in self.message_history
        ]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        messages_data = data.get('message_history', [])
        messages = []
        for msg_data in messages_data:
            if isinstance(msg_data, dict) and hasattr(ChatMessage, 'from_dict'):
                try:
                    messages.append(ChatMessage.from_dict(msg_data))
                except Exception as e:
                    # Log error and potentially skip malformed message
                    import logging
                    logging.getLogger(__name__).error(f"Error creating ChatMessage from dict: {msg_data}, error: {e}")
            elif isinstance(msg_data, ChatMessage): # If it's already a ChatMessage instance
                 messages.append(msg_data) # type: ignore
            # else: skip or handle malformed message data

        session_data = {k: v for k, v in data.items() if k != 'message_history'}
        session_data['message_history'] = messages
        return cls(**session_data)