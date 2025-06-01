# app/models/message_enums.py
from enum import Enum, auto

class MessageLoadingState(Enum):
    IDLE = auto()
    LOADING = auto()
    COMPLETED = auto()
    ERROR = auto()

class ApplicationMode(Enum):
    IDLE = auto()
    NORMAL_CHAT_PROCESSING = auto()

# You can add other message-related or state-related enums here
# if they are general enough to be considered part of the 'models' package.