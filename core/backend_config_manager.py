# app/core/backend_config_manager.py
import logging
from typing import Optional, Dict, List, Any  # Changed Tuple to Any for config values

from PySide6.QtCore import QObject, Slot, Signal

try:
    from core.event_bus import EventBus
    from llm.backend_coordinator import BackendCoordinator
    from utils import constants
    from llm import prompts as llm_prompts  # For default system prompts
    from core.config import get_gemini_api_key, get_openai_api_key  # For API keys
except ImportError as e_bcm:
    logging.getLogger(__name__).critical(f"BackendConfigManager: Critical import error: {e_bcm}", exc_info=True)
    EventBus = type("EventBus", (object,), {})  # type: ignore
    BackendCoordinator = type("BackendCoordinator", (object,), {})  # type: ignore
    constants = type("constants", (object,), {})  # type: ignore
    llm_prompts = type("llm_prompts", (object,), {})  # type: ignore


    def get_gemini_api_key() -> Optional[str]:
        return None


    def get_openai_api_key() -> Optional[str]:
        return None


    raise

logger = logging.getLogger(__name__)


class BackendConfigManager(QObject):
    """
    Manages the configuration and selection of active LLM backends for various tasks.
    It listens to UI requests for backend/model changes, configures them via
    the BackendCoordinator, and keeps track of the current active settings.
    It also manages settings like temperature and system prompts per purpose.
    """

    # Emitted when the effective configuration for a purpose (chat, specialized) changes.
    # This could be due to model selection, system prompt update, or config status change.
    activeConfigurationForPurposeChanged = Signal(str)  # purpose_key (e.g., "chat", "specialized")

    def __init__(self,
                 event_bus: EventBus,
                 backend_coordinator: BackendCoordinator,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(event_bus, EventBus):  # type: ignore
            raise TypeError("BackendConfigManager requires a valid EventBus.")
        if not isinstance(backend_coordinator, BackendCoordinator):  # type: ignore
            raise TypeError("BackendConfigManager requires a valid BackendCoordinator.")

        self._event_bus = event_bus
        self._backend_coordinator = backend_coordinator

        # Stores the desired active configuration for different "purposes".
        # 'is_actually_configured' reflects the status from BackendCoordinator.
        self._purpose_configs: Dict[str, Dict[str, Any]] = {
            "chat": {
                "backend_id": constants.DEFAULT_CHAT_BACKEND_ID,  # type: ignore
                "model_name": None,  # To be determined based on backend_id
                "system_prompt": "You are Ava, a bubbly, enthusiastic, and incredibly helpful AI assistant!",
                "temperature": 0.7,
                "is_actually_configured": False
            },
            "specialized_coder": {  # More specific purpose for coding tasks
                "backend_id": constants.GENERATOR_BACKEND_ID,  # type: ignore
                "model_name": None,  # To be determined
                "system_prompt": getattr(llm_prompts, 'ENHANCED_CODING_SYSTEM_PROMPT',
                                         "You are an expert Python coding assistant."),
                "temperature": 0.1,  # Lower temp for coding
                "is_actually_configured": False
            }
            # Add more purposes like "summarization", "translation" with their own defaults
        }

        self._connect_event_bus_handlers()
        self._initialize_and_attempt_configure_all_purposes()

        logger.info("BackendConfigManager initialized.")

    def _connect_event_bus_handlers(self):
        # Listen to UI-driven changes for backend/model/persona
        self._event_bus.chatLlmSelectionChanged.connect(
            lambda be_id, mdl_name: self.set_active_backend_for_purpose("chat", be_id, mdl_name)
        )
        self._event_bus.specializedLlmSelectionChanged.connect(
            lambda be_id, mdl_name: self.set_active_backend_for_purpose("specialized_coder", be_id, mdl_name)
        )
        self._event_bus.chatLlmPersonalitySubmitted.connect(self._handle_chat_personality_update)

        # Listen to BackendCoordinator's confirmation of configuration attempts
        self._event_bus.backendConfigurationChanged.connect(self._handle_backend_coordinator_config_update)

    def _initialize_and_attempt_configure_all_purposes(self):
        """
        Sets default model names based on backend IDs for each purpose
        and initiates their configuration.
        """
        logger.info("BCM: Initializing and attempting to configure default backends for all purposes...")
        for purpose, config in self._purpose_configs.items():
            backend_id = config["backend_id"]
            model_name = config.get("model_name")  # Might be None
            system_prompt = config.get("system_prompt")

            if not model_name:  # Determine default model if not explicitly set
                if backend_id == constants.DEFAULT_CHAT_BACKEND_ID:  # type: ignore
                    model_name = constants.DEFAULT_GEMINI_CHAT_MODEL  # type: ignore
                elif backend_id == "ollama_chat_default":
                    model_name = constants.DEFAULT_OLLAMA_CHAT_MODEL  # type: ignore
                elif backend_id == "gpt_chat_default":
                    model_name = constants.DEFAULT_GPT_CHAT_MODEL  # type: ignore
                elif backend_id == constants.GENERATOR_BACKEND_ID:  # type: ignore
                    model_name = constants.DEFAULT_OLLAMA_GENERATOR_MODEL  # type: ignore
                else:
                    logger.warning(
                        f"BCM: No default model mapping for backend_id '{backend_id}' for purpose '{purpose}'. Using None.")

                config["model_name"] = model_name  # Update the stored config

            if backend_id and model_name:
                self._initiate_backend_configuration(purpose, backend_id, model_name, system_prompt)
            else:
                logger.error(f"BCM: Cannot initiate config for purpose '{purpose}': missing backend_id or model_name.")

    def _initiate_backend_configuration(self,
                                        purpose_key: str,  # "chat" or "specialized_coder"
                                        backend_id: str,
                                        model_name: str,
                                        system_prompt: Optional[str]) -> bool:
        """
        Shared logic to prepare and trigger backend configuration via BackendCoordinator.
        """
        logger.info(
            f"BCM: Initiating configuration for Purpose='{purpose_key}', Backend='{backend_id}', Model='{model_name}'")
        # Emit a general status update that configuration is in progress
        self._event_bus.uiStatusUpdateGlobal.emit(
            f"Configuring {purpose_key.replace('_', ' ').title()} LLM: {model_name[:30]}...", "#e5c07b", True, 6000)

        api_key_to_use: Optional[str] = None
        if "gemini" in backend_id.lower():
            api_key_to_use = get_gemini_api_key()
        elif "gpt" in backend_id.lower():
            api_key_to_use = get_openai_api_key()

        if ("gemini" in backend_id.lower() or "gpt" in backend_id.lower()) and not api_key_to_use:
            err_msg = f"{backend_id.split('_')[0].upper()} API Key missing for {purpose_key}."
            logger.error(f"BCM: {err_msg}")
            # Directly update our status and notify others, as BackendCoordinator won't be called
            self._purpose_configs[purpose_key]["is_actually_configured"] = False
            self.activeConfigurationForPurposeChanged.emit(purpose_key)
            self._event_bus.uiStatusUpdateGlobal.emit(f"{err_msg} Check .env/settings.", "#FF6B6B", True, 7000)
            # Also emit the generic backendConfigurationChanged so UI elements like LeftPanel can update their lists/status for this backend_id
            self._event_bus.backendConfigurationChanged.emit(backend_id, model_name, False,
                                                             self._backend_coordinator.get_available_models_for_backend(
                                                                 backend_id)  # Send current known models
                                                             )
            return False

        # BackendCoordinator.configure_backend returns True if the call was accepted for processing.
        # The actual success/failure is reported via the backendConfigurationChanged signal.
        config_attempt_accepted = self._backend_coordinator.configure_backend(
            backend_id=backend_id, api_key=api_key_to_use,
            model_name=model_name, system_prompt=system_prompt
        )

        if not config_attempt_accepted:
            last_error = self._backend_coordinator.get_last_error_for_backend(backend_id)
            logger.error(
                f"BCM: BackendCoordinator rejected initial configuration for {backend_id} ({purpose_key}). Error: {last_error}")
            self._purpose_configs[purpose_key]["is_actually_configured"] = False
            self.activeConfigurationForPurposeChanged.emit(purpose_key)
            self._event_bus.uiStatusUpdateGlobal.emit(
                f"Config Error ({purpose_key}, {backend_id}): {last_error or 'Setup issue'}", "#FF6B6B", True, 7000)
            self._event_bus.backendConfigurationChanged.emit(backend_id, model_name, False,
                                                             self._backend_coordinator.get_available_models_for_backend(
                                                                 backend_id))
            return False

        return True  # Configuration attempt was initiated

    @Slot(str, str, bool, list)
    def _handle_backend_coordinator_config_update(self, backend_id_from_event: str, model_name_from_event: str,
                                                  is_configured_status: bool, available_models: List[str]):
        """
        Handles the backendConfigurationChanged event from BackendCoordinator.
        Updates the 'is_actually_configured' status for any purpose using this backend/model.
        """
        logger.info(
            f"BCM: Received config update from BackendCoordinator: BE='{backend_id_from_event}', Model='{model_name_from_event}', Configured='{is_configured_status}'")

        config_changed_for_any_purpose = False
        for purpose, current_config_for_purpose in self._purpose_configs.items():
            if current_config_for_purpose["backend_id"] == backend_id_from_event and \
                    current_config_for_purpose["model_name"] == model_name_from_event:

                if current_config_for_purpose["is_actually_configured"] != is_configured_status:
                    current_config_for_purpose["is_actually_configured"] = is_configured_status
                    config_changed_for_any_purpose = True
                    logger.info(
                        f"BCM: Status for purpose '{purpose}' (Backend: {backend_id_from_event}, Model: {model_name_from_event}) updated to Configured={is_configured_status}.")
                    self.activeConfigurationForPurposeChanged.emit(purpose)

                    # Emit a user-friendly global status message
                    ui_purpose_name = purpose.replace("_", " ").title()
                    model_display_name = model_name_from_event.split('/')[-1][:30]  # Shorten for display
                    if is_configured_status:
                        self._event_bus.uiStatusUpdateGlobal.emit(
                            f"{ui_purpose_name} LLM ({model_display_name}) ready.", "#98c379", True, 3000)
                    else:
                        last_error = self._backend_coordinator.get_last_error_for_backend(
                            backend_id_from_event) or "config failed"
                        self._event_bus.uiStatusUpdateGlobal.emit(
                            f"Error: {ui_purpose_name} LLM ({model_display_name}) - {last_error[:30]}...", "#FF6B6B",
                            True, 7000)

        if not config_changed_for_any_purpose:
            logger.debug(
                f"BCM: Config update for {backend_id_from_event}/{model_name_from_event} did not change status for any active purpose config.")

    def set_active_backend_for_purpose(self, purpose_key: str, backend_id: str, model_name: str):
        """Sets the desired backend and model for a purpose and initiates configuration."""
        if purpose_key not in self._purpose_configs:
            logger.error(f"BCM: Invalid purpose_key '{purpose_key}' provided.")
            return

        current_config = self._purpose_configs[purpose_key]
        if current_config["backend_id"] == backend_id and current_config["model_name"] == model_name:
            logger.info(
                f"BCM: {purpose_key} backend already targeting {backend_id}/{model_name}. Re-initiating configuration to ensure freshness/apply system prompt.")
        else:
            logger.info(f"BCM: Setting active {purpose_key} backend to ID='{backend_id}', Model='{model_name}'")

        # Persist the user's choice for this purpose
        current_config["backend_id"] = backend_id
        current_config["model_name"] = model_name
        # The system prompt for this purpose remains unchanged unless explicitly updated
        system_prompt = current_config.get("system_prompt")

        self._initiate_backend_configuration(purpose_key, backend_id, model_name, system_prompt)

    @Slot(str, str)  # new_prompt_text, backend_id_of_persona_target
    def _handle_chat_personality_update(self, new_prompt_text: str, backend_id_of_persona_target: str):
        """Updates the system prompt for the 'chat' purpose and reconfigures its active backend."""
        chat_config = self._purpose_configs["chat"]

        # The persona change is always for the *currently selected* chat backend.
        # backend_id_of_persona_target from the event should match chat_config["backend_id"].
        if backend_id_of_persona_target != chat_config["backend_id"]:
            logger.warning(f"BCM: Personality update for backend '{backend_id_of_persona_target}', "
                           f"but active chat backend is '{chat_config['backend_id']}'. "
                           f"Applying to active chat backend configuration.")

        new_system_prompt = new_prompt_text.strip() if new_prompt_text and new_prompt_text.strip() else None

        if chat_config["system_prompt"] != new_system_prompt:
            logger.info(f"BCM: Updating chat personality for active chat backend '{chat_config['backend_id']}'.")
            chat_config["system_prompt"] = new_system_prompt

            if chat_config["backend_id"] and chat_config["model_name"]:
                self._initiate_backend_configuration(
                    "chat", chat_config["backend_id"], chat_config["model_name"], chat_config["system_prompt"]
                )
                self.activeConfigurationForPurposeChanged.emit("chat")  # Notify that chat config effectively changed
            else:
                logger.error("BCM: Cannot apply personality - active chat backend/model not fully defined.")
        else:
            logger.debug("BCM: Chat personality submitted but it's the same as current. No change.")

    def set_temperature_for_purpose(self, purpose_key: str, temperature: float):
        if purpose_key not in self._purpose_configs:
            logger.error(f"BCM: Invalid purpose_key '{purpose_key}' for setting temperature.")
            return
        if not (0.0 <= temperature <= 2.0):  # Typical valid range
            logger.warning(f"BCM: Invalid temperature value {temperature} for {purpose_key}. Clamping to 0.0-2.0.")
            temperature = max(0.0, min(temperature, 2.0))

        if self._purpose_configs[purpose_key].get("temperature") != temperature:
            self._purpose_configs[purpose_key]["temperature"] = temperature
            logger.info(f"BCM: Temperature for purpose '{purpose_key}' set to {temperature:.2f}.")
            self.activeConfigurationForPurposeChanged.emit(purpose_key)
            self._event_bus.uiStatusUpdateGlobal.emit(
                f"{purpose_key.replace('_', ' ').title()} temperature: {temperature:.2f}", "#61afef", True, 2500)

    # --- Getters for other components (e.g., ChatManager, ChatRouter) ---
    def get_active_config_for_purpose(self, purpose_key: str) -> Optional[Dict[str, Any]]:
        """Returns a copy of the active configuration for a given purpose (e.g., "chat", "specialized_coder")."""
        config = self._purpose_configs.get(purpose_key)
        return config.copy() if config else None  # Return a copy to prevent external modification

    def is_purpose_configured_and_ready(self, purpose_key: str) -> bool:
        """Checks if the backend designated for a specific purpose is actually configured and ready."""
        config = self._purpose_configs.get(purpose_key)
        return config.get("is_actually_configured", False) if config else False