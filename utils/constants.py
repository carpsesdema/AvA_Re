# utils/constants.py
import logging
import os
import sys

logger = logging.getLogger(__name__)

APP_NAME = "AvA: PySide6 Rebuild"
APP_VERSION = "1.0.0"

DEFAULT_CHAT_BACKEND_ID = "gemini_chat_default"
DEFAULT_GEMINI_CHAT_MODEL = "gemini-2.5-pro-preview-05-06"
DEFAULT_OLLAMA_CHAT_MODEL = "llama3:latest"
DEFAULT_GPT_CHAT_MODEL = "gpt-4o"

GENERATOR_BACKEND_ID = "ollama_generator_default"
DEFAULT_OLLAMA_GENERATOR_MODEL = "qwen2.5-coder:14b"

# Core application settings
CHAT_FONT_FAMILY = "Consolas" # Preferred font for chat
CHAT_FONT_SIZE = 11         # Default font size

# --- UI Theme Colors (Example - these can be expanded or moved to a theme manager) ---
THEME_BACKGROUND_DARK = "#0D1117"
THEME_BACKGROUND_MEDIUM = "#161B22"
THEME_BACKGROUND_LIGHT = "#21262D"

THEME_TEXT_PRIMARY = "#C9D1D9"
THEME_TEXT_SECONDARY = "#8B949E"
THEME_TEXT_ACCENT_GREEN = "#39D353"
THEME_TEXT_ACCENT_GREEN_HOVER = "#5EE878"
THEME_TEXT_ERROR = "#F85149"

THEME_ACCENT_GREEN = THEME_TEXT_ACCENT_GREEN
THEME_ACCENT_GREEN_LIGHT = "#2EA043"
THEME_BORDER_COLOR = "#30363D"

USER_BUBBLE_COLOR_HEX = THEME_BACKGROUND_MEDIUM
USER_TEXT_COLOR_HEX = THEME_ACCENT_GREEN

AI_BUBBLE_COLOR_HEX = THEME_BACKGROUND_MEDIUM
AI_TEXT_COLOR_HEX = THEME_TEXT_PRIMARY

SYSTEM_BUBBLE_COLOR_HEX = THEME_BACKGROUND_LIGHT
SYSTEM_TEXT_COLOR_HEX = THEME_TEXT_SECONDARY

ERROR_BUBBLE_COLOR_HEX = THEME_TEXT_ERROR
ERROR_TEXT_COLOR_HEX = "#FFFFFF"

BUBBLE_BORDER_COLOR_HEX = THEME_BORDER_COLOR
TIMESTAMP_COLOR_HEX = THEME_TEXT_SECONDARY
CODE_BLOCK_BG_COLOR_HEX = "#010409"

BUTTON_BG_COLOR = THEME_BACKGROUND_LIGHT
BUTTON_TEXT_COLOR = THEME_TEXT_PRIMARY
BUTTON_BORDER_COLOR = THEME_BORDER_COLOR
BUTTON_HOVER_BG_COLOR = "#2f353c"
BUTTON_PRESSED_BG_COLOR = "#272b30"

BUTTON_ACCENT_BG_COLOR = THEME_ACCENT_GREEN
BUTTON_ACCENT_TEXT_COLOR = THEME_BACKGROUND_DARK
BUTTON_ACCENT_HOVER_BG_COLOR = THEME_TEXT_ACCENT_GREEN_HOVER

SCROLLBAR_BG_COLOR = THEME_BACKGROUND_DARK
SCROLLBAR_HANDLE_COLOR = "#30363D"
SCROLLBAR_HANDLE_HOVER_COLOR = "#3C424A"

INPUT_BG_COLOR = THEME_BACKGROUND_MEDIUM
INPUT_TEXT_COLOR = THEME_TEXT_PRIMARY
INPUT_BORDER_COLOR = THEME_BORDER_COLOR
INPUT_FOCUS_BORDER_COLOR = THEME_ACCENT_GREEN

# --- File Paths & Directories ---
# Determine base directory
if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(sys.executable)
else:
    # Assuming this constants.py is in a 'utils' subdirectory of the project root
    APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USER_DATA_DIR_NAME = ".ava_pys6_data_p1" # Name for the user-specific data folder
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), USER_DATA_DIR_NAME)

ASSETS_DIR_NAME = "assets"
ASSETS_PATH = os.path.join(APP_BASE_DIR, ASSETS_DIR_NAME) # Path to assets folder

STYLESHEET_FILENAME = "style.qss"
BUBBLE_STYLESHEET_FILENAME = "bubble_style.qss"
UI_DIR_NAME = "ui" # Assuming ui files are in an 'ui' subdirectory of app
# Corrected path assuming constants.py is in utils/, and ui/ is in app/
UI_DIR_PATH = os.path.join(APP_BASE_DIR, UI_DIR_NAME)


STYLE_PATHS_TO_CHECK = [ # Paths to search for the main stylesheet
    os.path.join(UI_DIR_PATH, STYLESHEET_FILENAME), # Preferred: app/ui/style.qss
    os.path.join(APP_BASE_DIR, STYLESHEET_FILENAME) # Fallback: project_root/style.qss
]
BUBBLE_STYLESHEET_PATH = os.path.join(UI_DIR_PATH, BUBBLE_STYLESHEET_FILENAME)

LOADING_GIF_FILENAME = "loading.gif" # In assets folder
APP_ICON_FILENAME = "Synchat.ico" # In assets folder

# --- RAG Specific Constants ---
RAG_COLLECTIONS_DIR_NAME = "rag_collections"
RAG_COLLECTIONS_PATH = os.path.join(USER_DATA_DIR, RAG_COLLECTIONS_DIR_NAME)
GLOBAL_COLLECTION_ID = "global_knowledge" # Default/global RAG collection name

RAG_NUM_RESULTS = 5          # Number of results to fetch from RAG
RAG_CHUNK_SIZE = 1000        # Character size for RAG chunks
RAG_CHUNK_OVERLAP = 150      # Character overlap for RAG chunks
RAG_MAX_FILE_SIZE_MB = 50    # Max file size in MB for RAG processing
MAX_SCAN_DEPTH = 5           # Max directory depth for RAG scanning

ALLOWED_TEXT_EXTENSIONS = {  # File extensions considered for RAG text processing
    '.txt', '.md', '.markdown', '.rst',
    '.py', '.js', '.ts', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env',
    '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.swift', '.php', '.rb',
    '.pdf', '.docx', # Note: .pdf and .docx require special handling
}

DEFAULT_IGNORED_DIRS = { # Directories ignored during RAG scanning
    '.git', '.idea', '__pycache__', 'venv', 'node_modules', 'build', 'dist',
    '.pytest_cache', '.vscode', '.env', '.DS_Store', 'logs',
}

# --- Logging Configuration ---
LOG_LEVEL = "INFO"  # Default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE_NAME = "ava_pys6_phase1.log" # Log file name within USER_DATA_DIR
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)-8s - [%(name)s:%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- Ensure USER_DATA_DIR exists ---
try:
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    logger.debug(f"User data directory ensured at: {USER_DATA_DIR}")
except OSError as e:
    logger.critical(f"CRITICAL: Error creating user data directory in constants.py: {e}", exc_info=True)
    # Depending on how critical this is, you might raise an exception or exit
    # For now, just log it critically.