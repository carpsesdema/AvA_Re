# app/core/config.py
import logging
import os
import sys
from typing import Optional

try:
    import dotenv
except ImportError:
    dotenv = None
    logging.getLogger(__name__).warning(
        "python-dotenv library not found. .env file will not be loaded. pip install python-dotenv")

logger = logging.getLogger(__name__)

# --- Path Setup & Environment Variable Configuration ---
# This section should run as early as possible, before other modules import libraries
# that might use these environment variables (like sentence-transformers).

if getattr(sys, 'frozen', False):
    # Application is running in a bundled PyInstaller executable
    _APP_EXECUTABLE_DIR = os.path.dirname(sys.executable)
    _MEIPASS_DIR = getattr(sys, '_MEIPASS', _APP_EXECUTABLE_DIR) # sys._MEIPASS is the temp dir for bundled files
    APP_BASE_DIR_RESOLVED = _MEIPASS_DIR
    logger.info(f"Running bundled. APP_BASE_DIR resolved to MEIPASS: {APP_BASE_DIR_RESOLVED}")
else:
    # Running from source
    # Assuming this config.py is in app/core/
    APP_BASE_DIR_RESOLVED = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logger.info(f"Running from source. APP_BASE_DIR resolved to: {APP_BASE_DIR_RESOLVED}")

# Define USER_DATA_DIR here as it's needed for cache paths
# This mirrors the logic that might be in utils.constants but ensures it's available early.
_USER_DATA_DIR_NAME_CONFIG = ".ava_pys6_data_p1" # Consistent with constants.py
USER_DATA_DIR_CONFIG = os.path.join(os.path.expanduser("~"), _USER_DATA_DIR_NAME_CONFIG)

try:
    os.makedirs(USER_DATA_DIR_CONFIG, exist_ok=True)

    # Configure cache directories for Hugging Face libraries
    # These need to be set BEFORE sentence_transformers or transformers are imported.
    _HF_CACHE_BASE_DIR = os.path.join(USER_DATA_DIR_CONFIG, "huggingface_cache")
    os.makedirs(_HF_CACHE_BASE_DIR, exist_ok=True)

    _TRANSFORMERS_CACHE_DIR = os.path.join(_HF_CACHE_BASE_DIR, "transformers")
    _SENTENCE_TRANSFORMERS_HOME_DIR = os.path.join(_HF_CACHE_BASE_DIR, "sentence_transformers")
    _HF_ASSETS_CACHE_DIR = os.path.join(_HF_CACHE_BASE_DIR, "assets")
    _HF_HUB_CACHE_DIR = os.path.join(_HF_CACHE_BASE_DIR, "hub")


    os.environ['TRANSFORMERS_CACHE'] = _TRANSFORMERS_CACHE_DIR
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = _SENTENCE_TRANSFORMERS_HOME_DIR
    # HF_HOME is a general cache directory for Hugging Face Hub related files (models, datasets)
    os.environ['HF_HOME'] = _HF_HUB_CACHE_DIR # Often includes models, datasets, etc.
    # Some older versions might use HF_ASSETS_CACHE
    os.environ['HF_ASSETS_CACHE'] = _HF_ASSETS_CACHE_DIR


    os.makedirs(_TRANSFORMERS_CACHE_DIR, exist_ok=True)
    os.makedirs(_SENTENCE_TRANSFORMERS_HOME_DIR, exist_ok=True)
    os.makedirs(_HF_ASSETS_CACHE_DIR, exist_ok=True)
    os.makedirs(_HF_HUB_CACHE_DIR, exist_ok=True)

    logger.info(f"Hugging Face cache directories configured within: {USER_DATA_DIR_CONFIG}")
    logger.info(f"  TRANSFORMERS_CACHE: {_TRANSFORMERS_CACHE_DIR}")
    logger.info(f"  SENTENCE_TRANSFORMERS_HOME: {_SENTENCE_TRANSFORMERS_HOME_DIR}")
    logger.info(f"  HF_HOME (Hub cache): {_HF_HUB_CACHE_DIR}")

except Exception as e_cache_setup:
    logger.error(f"Failed to set up Hugging Face cache directories: {e_cache_setup}", exc_info=True)

# --- End Path Setup & Environment Variable Configuration ---


# Determine the base directory for .env loading
# If frozen, sys.executable is the path to the .exe. The .env might be alongside it or in project root.
# If not frozen, it's relative to this config.py file.
if getattr(sys, 'frozen', False):
    # For a bundled app, assume .env might be next to the executable or not used (env vars preferred)
    _DOTENV_BASE_DIR = os.path.dirname(sys.executable)
else:
    # config.py is in app/core/ so project root is three levels up
    _DOTENV_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DOTENV_PATH = os.path.join(_DOTENV_BASE_DIR, '.env')


def load_config() -> dict:
    config = {}

    if dotenv and os.path.exists(_DOTENV_PATH):
        logger.info(f"Loading configuration from: {_DOTENV_PATH}")
        dotenv.load_dotenv(dotenv_path=_DOTENV_PATH)
        config['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")
        config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
        if not config.get('GEMINI_API_KEY'):
            logger.warning("GEMINI_API_KEY not found or empty in .env file.")
        if not config.get('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY not found or empty in .env file.")
    else:
        if not dotenv:
            logger.info(
                ".env file support disabled (python-dotenv not installed). Checking system environment variables.")
        elif not os.path.exists(_DOTENV_PATH): # Check specifically if .env is missing
             logger.info(f".env file not found at expected path '{_DOTENV_PATH}'. Checking system environment variables.")
        else: # Other reason it wasn't loaded (e.g. permission issue, though unlikely)
            logger.warning(f"Could not load .env file from {_DOTENV_PATH}. Checking system environment variables.")


        config['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")
        config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
        if not config.get('GEMINI_API_KEY'):
            logger.warning("GEMINI_API_KEY not found in system environment variables.")
        if not config.get('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY not found in system environment variables.")
    return config


APP_CONFIG = load_config()


def get_gemini_api_key() -> Optional[str]:
    return APP_CONFIG.get("GEMINI_API_KEY")


def get_openai_api_key() -> Optional[str]:
    return APP_CONFIG.get("OPENAI_API_KEY")

# You can add other configuration-related functions or classes here if needed.
# For example, functions to get specific settings with defaults.