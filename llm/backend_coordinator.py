# app/llm/backend_coordinator.py
import asyncio
import logging
import uuid
from datetime import time
from typing import List, Optional, Dict, Any, Tuple

from PySide6.QtCore import QObject, QTimer  # QTimer for delayed model fetching

try:
    # Assuming BackendInterface is in the same package
    from .backend_interface import BackendInterface
    # Adapters are also in the same package
    from .gemini_adapter import GeminiAdapter
    from .ollama_adapter import OllamaAdapter
    from .gpt_adapter import GPTAdapter
    # Models are in app.models
    from models.chat_message import ChatMessage, MODEL_ROLE, USER_ROLE, ERROR_ROLE
    # EventBus is in core
    from core.event_bus import EventBus
    # Constants for default backend IDs
    from utils import constants
except ImportError as e_bc:
    logging.getLogger(__name__).critical(f"BackendCoordinator: Critical import error: {e_bc}", exc_info=True)
    # Fallback types for type hinting
    BackendInterface = type("BackendInterface", (object,), {})  # type: ignore
    GeminiAdapter = type("GeminiAdapter", (object,), {})  # type: ignore
    OllamaAdapter = type("OllamaAdapter", (object,), {})  # type: ignore
    GPTAdapter = type("GPTAdapter", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MODEL_ROLE, USER_ROLE = "model", "user"
    EventBus = type("EventBus", (object,), {"get_instance": lambda: type("DummyBus", (object,), {
        "llmRequestSent": type("Signal", (object,), {"emit": lambda *args, **kwargs: None})(),
        "llmStreamChunkReceived": type("Signal", (object,), {"emit": lambda *args, **kwargs: None})(),
        "llmResponseCompleted": type("Signal", (object,), {"emit": lambda *args, **kwargs: None})(),
        "llmResponseError": type("Signal", (object,), {"emit": lambda *args, **kwargs: None})(),
        "backendConfigurationChanged": type("Signal", (object,), {"emit": lambda *args, **kwargs: None})(),
        "backendBusyStateChanged": type("Signal", (object,), {"emit": lambda *args, **kwargs: None})(),
        "llmStreamStarted": type("Signal", (object,), {"emit": lambda *args, **kwargs: None})()
    })()})  # type: ignore
    constants = type("constants", (object,), {  # type: ignore
        "DEFAULT_CHAT_BACKEND_ID": "gemini_chat_default",
        "GENERATOR_BACKEND_ID": "ollama_generator_default"
    })
    raise

logger = logging.getLogger(__name__)


class BackendCoordinator(QObject):
    """
    Manages multiple LLM backend adapters, facilitates configuration,
    and routes LLM requests to the appropriate adapter.
    """

    # If backend_adapters dict is not passed, it will create default ones.
    def __init__(self, backend_adapters: Optional[Dict[str, BackendInterface]] = None,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._event_bus = EventBus.get_instance()

        if backend_adapters:
            self._backend_adapters = backend_adapters
        else:
            # If no adapters are passed (e.g. ApplicationOrchestrator creates this first),
            # instantiate them here. This is the key change for SRP.
            logger.info("BackendCoordinator: No adapters passed, instantiating defaults.")
            self._backend_adapters = {
                constants.DEFAULT_CHAT_BACKEND_ID: GeminiAdapter(),  # type: ignore
                "ollama_chat_default": OllamaAdapter(),
                "gpt_chat_default": GPTAdapter(),
                constants.GENERATOR_BACKEND_ID: OllamaAdapter(),  # Separate instance for generation
            }
            # Ensure all keys exist for the state dictionaries based on these adapters
            for adapter_id in self._backend_adapters.keys():
                if not hasattr(self, '_current_model_names') or adapter_id not in self._current_model_names:
                    self._initialize_adapter_state(adapter_id)

        # State tracking for each backend adapter
        self._current_model_names: Dict[str, Optional[str]] = {bid: None for bid in self._backend_adapters}
        self._current_system_prompts: Dict[str, Optional[str]] = {bid: None for bid in self._backend_adapters}
        self._is_configured_map: Dict[str, bool] = {bid: False for bid in self._backend_adapters}
        self._available_models_map: Dict[str, List[str]] = {bid: [] for bid in self._backend_adapters}
        self._last_errors_map: Dict[str, Optional[str]] = {bid: None for bid in self._backend_adapters}

        self._active_backend_tasks: Dict[str, asyncio.Task] = {}  # request_id -> asyncio.Task
        self._overall_is_busy: bool = False

        # Cache for model fetching to avoid rapid repeated API calls
        self._models_fetch_cache_timestamp: Dict[str, float] = {}  # backend_id -> last_fetch_timestamp
        self._models_fetch_cooldown_seconds = 60.0  # Cooldown period in seconds (e.g., 1 minute)

        # QTimer for scheduling asynchronous model list fetching to avoid blocking UI
        self._model_fetch_scheduler_timer: Optional[QTimer] = None

        logger.info(
            f"BackendCoordinator initialized with {len(self._backend_adapters)} adapter(s): {list(self._backend_adapters.keys())}")

    def _initialize_adapter_state(self, backend_id: str):
        """Helper to initialize state dictionaries if an adapter is added dynamically (less common)."""
        if not hasattr(self, '_current_model_names'): self._current_model_names = {}
        if not hasattr(self, '_current_system_prompts'): self._current_system_prompts = {}
        if not hasattr(self, '_is_configured_map'): self._is_configured_map = {}
        if not hasattr(self, '_available_models_map'): self._available_models_map = {}
        if not hasattr(self, '_last_errors_map'): self._last_errors_map = {}

        self._current_model_names[backend_id] = None
        self._current_system_prompts[backend_id] = None
        self._is_configured_map[backend_id] = False
        self._available_models_map[backend_id] = []
        self._last_errors_map[backend_id] = None

    def _update_overall_busy_state(self):
        """Updates and emits the overall busy state based on active tasks."""
        new_busy_state = any(task and not task.done() for task in self._active_backend_tasks.values())
        if self._overall_is_busy != new_busy_state:
            self._overall_is_busy = new_busy_state
            self._event_bus.backendBusyStateChanged.emit(self._overall_is_busy)
            logger.debug(f"BC: Overall busy state changed to: {self._overall_is_busy}")

    def configure_backend(self,
                          backend_id: str,
                          api_key: Optional[str],
                          model_name: str,
                          system_prompt: Optional[str] = None) -> bool:
        """
        Configures the specified backend adapter.
        Returns True if the configuration call was accepted by the adapter,
        False if adapter not found or immediate pre-check failed.
        The actual success of configuration is emitted via backendConfigurationChanged.
        """
        adapter = self._backend_adapters.get(backend_id)
        if not adapter:
            # Ensure state maps are initialized if a new backend_id is attempted
            if backend_id not in self._current_model_names: self._initialize_adapter_state(backend_id)

            self._is_configured_map[backend_id] = False
            self._last_errors_map[backend_id] = f"Adapter not found for backend_id '{backend_id}'."
            self._current_model_names[backend_id] = model_name  # Store attempted model
            self._current_system_prompts[backend_id] = system_prompt
            self._available_models_map[backend_id] = []  # No models if no adapter
            # Emit failure immediately if adapter doesn't exist
            self._event_bus.backendConfigurationChanged.emit(backend_id, model_name, False, [])
            logger.error(self._last_errors_map[backend_id])
            return False  # Configuration call not accepted

        # Call the adapter's configure method
        is_config_call_successful = adapter.configure(api_key=api_key, model_name=model_name,
                                                      system_prompt=system_prompt)

        # Update internal state based on the immediate result of the configure call
        self._is_configured_map[backend_id] = is_config_call_successful
        self._last_errors_map[backend_id] = adapter.get_last_error() if not is_config_call_successful else None
        self._current_model_names[backend_id] = model_name
        self._current_system_prompts[backend_id] = system_prompt

        # Emit the result of this configuration attempt
        # Available models might not be known yet, or we can use cached ones.
        # For now, send cached models; get_available_models will trigger fresh fetch if needed.
        cached_models = self._available_models_map.get(backend_id, [])
        self._event_bus.backendConfigurationChanged.emit(backend_id, model_name, is_config_call_successful,
                                                         cached_models)

        if is_config_call_successful:
            logger.info(
                f"BC: Configuration initiated for backend '{backend_id}' with model '{model_name}'. Fetching models...")
            # Schedule an asynchronous model fetch to update the dropdowns without blocking
            self._schedule_model_list_fetch(backend_id)
        else:
            logger.error(
                f"BC: Adapter-level configuration failed for backend '{backend_id}'. Last error: {self._last_errors_map[backend_id]}")

        return is_config_call_successful  # Return if the adapter accepted the call

    def _schedule_model_list_fetch(self, backend_id: str):
        """Schedules an asynchronous fetch of available models for a backend."""
        if not self._model_fetch_scheduler_timer:
            self._model_fetch_scheduler_timer = QTimer(self)
            self._model_fetch_scheduler_timer.setSingleShot(True)
            # Connect to a lambda that captures current backend_id for the timeout
            self._model_fetch_scheduler_timer.timeout.connect(
                lambda: asyncio.create_task(self._fetch_and_update_models_async(backend_id))
            )

        # If timer is already running for this backend_id, don't restart immediately unless debouncing.
        # For simplicity, just start/restart. A more complex system might queue or debounce.
        logger.debug(f"BC: Scheduling async model fetch for '{backend_id}' in 100ms.")
        self._model_fetch_scheduler_timer.start(100)  # Short delay

    async def _fetch_and_update_models_async(self, backend_id: str):
        """Asynchronously fetches models and updates the state and UI."""
        adapter = self._backend_adapters.get(backend_id)
        if not adapter or not self._is_configured_map.get(backend_id, False):
            logger.warning(f"BC: Cannot fetch models for '{backend_id}', adapter not found or not configured.")
            return

        logger.info(f"BC: Starting async model fetch for '{backend_id}'...")
        try:
            # The adapter's get_available_models() should be designed to be reasonably fast
            # or handle its own internal threading for long calls if necessary.
            # For now, assume it's acceptable to call directly in this asyncio task.
            available_models = await asyncio.to_thread(adapter.get_available_models)

            self._available_models_map[backend_id] = available_models
            self._models_fetch_cache_timestamp[backend_id] = time.time()  # Update timestamp
            logger.info(f"BC: Successfully fetched {len(available_models)} models for '{backend_id}'.")

            # Emit backendConfigurationChanged again, this time with updated models list
            # This signal will be picked up by UI components (like LeftPanel) to refresh dropdowns
            self._event_bus.backendConfigurationChanged.emit(
                backend_id,
                self._current_model_names.get(backend_id),  # Current selected model
                self._is_configured_map.get(backend_id, False),  # Current config status
                available_models  # The newly fetched list
            )
        except Exception as e_fetch:
            logger.error(f"BC: Failed to fetch models asynchronously for '{backend_id}': {e_fetch}", exc_info=True)
            # Optionally, emit with empty list or cached list on error
            # self._event_bus.backendConfigurationChanged.emit(backend_id, self._current_model_names.get(backend_id), self._is_configured_map.get(backend_id, False), self._available_models_map.get(backend_id, []))

    def get_available_models_for_backend(self, backend_id: str) -> List[str]:
        """
        Returns cached available models for a backend.
        If cache is stale or empty, and backend is configured, schedules an async refresh.
        """
        adapter = self._backend_adapters.get(backend_id)
        if not adapter:
            logger.warning(f"BC: No adapter found for '{backend_id}' when getting available models.")
            return []

        cached_models = self._available_models_map.get(backend_id, [])
        last_fetch_time = self._models_fetch_cache_timestamp.get(backend_id, 0)
        is_configured = self._is_configured_map.get(backend_id, False)

        # Refresh if configured AND (no models cached OR cache is stale)
        if is_configured and (
                not cached_models or (time.time() - last_fetch_time > self._models_fetch_cooldown_seconds)):
            logger.info(f"BC: Model cache for '{backend_id}' is empty or stale. Scheduling refresh.")
            self._schedule_model_list_fetch(backend_id)

        return cached_models

    def initiate_llm_chat_request(self,
                                  target_backend_id: str,
                                  history_to_send: List[ChatMessage],  # type: ignore
                                  options: Optional[Dict[str, Any]] = None
                                  ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates and prepares for an LLM chat request.
        Returns (success_flag, error_message_or_None, request_id_or_None).
        The actual streaming task is started by start_llm_streaming_task.
        """
        self._last_errors_map[target_backend_id] = None  # Clear previous error for this backend
        adapter = self._backend_adapters.get(target_backend_id)

        if not adapter:
            err_msg = f"Adapter not found for backend_id '{target_backend_id}'."
            self._last_errors_map[target_backend_id] = err_msg
            logger.error(f"BC: {err_msg}")
            return False, err_msg, None

        if not self._is_configured_map.get(target_backend_id, False):
            adapter_specific_error = adapter.get_last_error()  # Get error from adapter's last config attempt
            err_msg = f"Backend '{target_backend_id}' is not configured."
            if adapter_specific_error: err_msg += f" Adapter msg: {adapter_specific_error}"
            self._last_errors_map[target_backend_id] = err_msg
            logger.error(f"BC: {err_msg}")
            return False, err_msg, None

        request_id = f"llm_req_{uuid.uuid4().hex[:12]}"
        logger.info(f"BC: LLM chat request initiated for backend '{target_backend_id}'. Assigned ReqID: {request_id}")
        return True, None, request_id

    def start_llm_streaming_task(self,
                                 request_id: str,
                                 target_backend_id: str,
                                 history_to_send: List[ChatMessage],  # type: ignore
                                 is_modification_response_expected: bool,
                                 # Not directly used here, but good for metadata
                                 options: Optional[Dict[str, Any]] = None,
                                 request_metadata: Optional[Dict[str, Any]] = None):  # For purpose, project_id etc.
        """
        Starts the asynchronous LLM streaming task for a pre-validated request.
        """
        adapter = self._backend_adapters.get(target_backend_id)
        # We assume initiate_llm_chat_request was called and validated the adapter and config.
        if not adapter or not self._is_configured_map.get(target_backend_id, False):
            err = f"Cannot start stream for ReqID '{request_id}': Adapter '{target_backend_id}' not found or not configured."
            logger.error(f"BC: {err}")
            self._event_bus.llmResponseError.emit(request_id, err)
            self._update_overall_busy_state()  # Ensure busy state is correct
            return

        if request_id in self._active_backend_tasks and not self._active_backend_tasks[request_id].done():
            logger.warning(f"BC: Task for ReqID '{request_id}' is already active. Ignoring duplicate start request.")
            return

        logger.info(f"BC: Starting LLM streaming task for ReqID '{request_id}' on backend '{target_backend_id}'.")
        self._event_bus.llmRequestSent.emit(target_backend_id, request_id)  # Notify UI/Loggers

        try:
            # Create an asyncio task to run the stream processing
            task = asyncio.create_task(
                self._internal_get_response_stream(
                    backend_id=target_backend_id,
                    request_id=request_id,
                    adapter=adapter,
                    history=history_to_send,
                    options=options,
                    request_metadata=request_metadata or {}  # Ensure it's a dict
                )
            )
            self._active_backend_tasks[request_id] = task
            self._update_overall_busy_state()
        except Exception as e_create_task:
            logger.critical(
                f"BC: CRITICAL ERROR during asyncio.create_task for ReqID '{request_id}' with backend '{target_backend_id}': {type(e_create_task).__name__} - {e_create_task}",
                exc_info=True)
            err_msg_detail = f"Failed to launch LLM task for '{target_backend_id}'. Error: {type(e_create_task).__name__}."
            self._last_errors_map[target_backend_id] = err_msg_detail
            self._event_bus.llmResponseError.emit(request_id, err_msg_detail)
            self._update_overall_busy_state()

    async def _internal_get_response_stream(self,
                                            backend_id: str,
                                            request_id: str,
                                            adapter: BackendInterface,
                                            history: List[ChatMessage],  # type: ignore
                                            options: Optional[Dict[str, Any]],
                                            request_metadata: Dict[str, Any]):  # Now expects a dict
        """Internal async method to handle the streaming and event emission."""
        response_buffer = ""

        # Prepare usage_stats_dict with initial metadata
        usage_stats_dict: Dict[str, Any] = request_metadata.copy()  # Start with passed metadata
        usage_stats_dict["backend_id"] = backend_id  # Ensure these are set
        usage_stats_dict["request_id"] = request_id

        # Ensure project_id is present for event emission, using a default if necessary
        if "project_id" not in usage_stats_dict:
            # This part might be better handled by the caller (e.g. ChatManager) ensuring project_id is always in metadata
            logger.warning(
                f"BC: 'project_id' missing in request_metadata for ReqID '{request_id}'. Using default 'unknown_project'.")
            usage_stats_dict["project_id"] = "unknown_project"
        if "session_id" not in usage_stats_dict:
            logger.warning(
                f"BC: 'session_id' missing in request_metadata for ReqID '{request_id}'. Using default 'unknown_session'.")
            usage_stats_dict["session_id"] = "unknown_session"

        try:
            if not hasattr(adapter, 'get_response_stream'):  # Should not happen with BackendInterface
                raise AttributeError(f"Adapter '{backend_id}' is missing the 'get_response_stream' method.")

            self._event_bus.llmStreamStarted.emit(request_id)
            logger.info(f"BC: Stream started for ReqID '{request_id}' on backend '{backend_id}'.")

            chunk_count = 0
            async for chunk in adapter.get_response_stream(history, options):
                chunk_count += 1
                # logger.debug(f"BC: Emitting chunk #{chunk_count} for ReqID '{request_id}': '{chunk[:30].replace('\n', ' ')}...'")
                # Emit chunk for the specific session
                self._event_bus.messageChunkReceivedForSession.emit(
                    usage_stats_dict["project_id"], usage_stats_dict["session_id"], request_id, chunk
                )
                # Also emit generic chunk received if some components listen to that (like LLM logger)
                self._event_bus.llmStreamChunkReceived.emit(request_id, chunk)
                response_buffer += chunk

                # Cooperative yielding for responsiveness, especially with many small chunks
                if chunk_count % 5 == 0 or len(chunk) > 100:  # Adjust conditions as needed
                    await asyncio.sleep(0)  # Yield to the event loop

            logger.info(f"BC: Stream completed for ReqID '{request_id}'. Total chunks: {chunk_count}.")

            final_response_text = response_buffer.strip()
            token_usage = adapter.get_last_token_usage()
            if token_usage:
                usage_stats_dict["prompt_tokens"] = token_usage[0]
                usage_stats_dict["completion_tokens"] = token_usage[1]

            # Get model name from adapter if available (some adapters might store it differently)
            usage_stats_dict["model_name"] = getattr(adapter, "_model_name", "unknown_model")

            # Create the final ChatMessage object
            # The ID of this ChatMessage should match the placeholder ID used in the UI
            # This placeholder_id should be passed in request_metadata by the LlmRequestHandler
            final_message_id = request_metadata.get("placeholder_message_id", request_id)

            if final_response_text:
                completed_message = ChatMessage(id=final_message_id, role=MODEL_ROLE,
                                                parts=[final_response_text])  # type: ignore
            else:  # Handle empty response from LLM
                empty_msg_text = "[AI returned an empty response]"
                logger.warning(f"BC: Empty response received for ReqID '{request_id}'.")
                completed_message = ChatMessage(id=final_message_id, role=MODEL_ROLE,
                                                parts=[empty_msg_text])  # type: ignore

            logger.info(
                f"BC: Emitting messageFinalizedForSession for ReqID '{request_id}' (Placeholder ID: {final_message_id}).")
            self._event_bus.messageFinalizedForSession.emit(
                usage_stats_dict["project_id"], usage_stats_dict["session_id"],
                final_message_id,  # Use placeholder_id for UI update
                completed_message, usage_stats_dict, False  # is_error = False
            )
            # Also emit general completion if needed
            # self._event_bus.llmResponseCompleted.emit(request_id, completed_message, usage_stats_dict)


        except asyncio.CancelledError:
            logger.info(f"BC: Stream task for ReqID '{request_id}' was cancelled.")
            # Emit error for the specific session
            self._event_bus.messageFinalizedForSession.emit(
                usage_stats_dict["project_id"], usage_stats_dict["session_id"],
                request_metadata.get("placeholder_message_id", request_id),  # Use placeholder ID
                ChatMessage(id=request_metadata.get("placeholder_message_id", request_id), role=ERROR_ROLE,
                            parts=["[AI response cancelled by user]"]),  # type: ignore
                usage_stats_dict, True  # is_error = True
            )
            # self._event_bus.llmResponseError.emit(request_id, "[AI response cancelled by user]")
        except Exception as e_stream:
            error_msg_detail = adapter.get_last_error() or f"LLM Stream Task Error for ReqID '{request_id}' on backend '{backend_id}': {type(e_stream).__name__} - {e_stream}"
            self._last_errors_map[backend_id] = error_msg_detail
            logger.error(f"BC: {error_msg_detail}", exc_info=True)
            # Emit error for the specific session
            self._event_bus.messageFinalizedForSession.emit(
                usage_stats_dict["project_id"], usage_stats_dict["session_id"],
                request_metadata.get("placeholder_message_id", request_id),  # Use placeholder ID
                ChatMessage(id=request_metadata.get("placeholder_message_id", request_id), role=ERROR_ROLE,
                            parts=[f"[AI Error: {error_msg_detail}]"]),  # type: ignore
                usage_stats_dict, True  # is_error = True
            )
            # self._event_bus.llmResponseError.emit(request_id, error_msg_detail)
        finally:
            self._active_backend_tasks.pop(request_id, None)  # Remove task from active list
            self._update_overall_busy_state()  # Update busy state

    def cancel_current_task(self, request_id: Optional[str] = None):
        """Cancels the LLM task associated with the given request_id, or all tasks if request_id is None."""
        if request_id:
            task = self._active_backend_tasks.get(request_id)
            if task and not task.done():
                logger.info(f"BC: Attempting to cancel task for ReqID '{request_id}'.")
                task.cancel()
                # Busy state will be updated in the _internal_get_response_stream's finally block
            elif task and task.done():
                logger.info(f"BC: Task for ReqID '{request_id}' already done, no cancellation needed.")
            else:
                logger.warning(f"BC: No active task found for ReqID '{request_id}' to cancel.")
        else:  # Cancel all active tasks
            logger.info("BC: Attempting to cancel ALL active LLM tasks.")
            cancelled_count = 0
            for req_id_key, task_to_cancel in list(self._active_backend_tasks.items()):  # Iterate copy
                if task_to_cancel and not task_to_cancel.done():
                    task_to_cancel.cancel()
                    cancelled_count += 1
            if cancelled_count > 0:
                logger.info(f"BC: {cancelled_count} active tasks signalled for cancellation.")
            else:
                logger.info("BC: No active tasks to cancel.")
        # _update_overall_busy_state will be called by the tasks' finally blocks.

    def is_backend_configured(self, backend_id: str) -> bool:
        return self._is_configured_map.get(backend_id, False)

    def get_last_error_for_backend(self, backend_id: str) -> Optional[str]:
        # Prioritize error from the adapter instance itself if available
        adapter = self._backend_adapters.get(backend_id)
        if adapter:
            adapter_error = adapter.get_last_error()
            if adapter_error: return adapter_error
        # Fallback to error stored in coordinator's map
        return self._last_errors_map.get(backend_id)

    def is_any_backend_busy(self) -> bool:
        return self._overall_is_busy

    def get_current_configured_model(self, backend_id: str) -> Optional[str]:
        return self._current_model_names.get(backend_id)

    def get_current_system_prompt(self, backend_id: str) -> Optional[str]:
        return self._current_system_prompts.get(backend_id)

    def get_all_backend_ids(self) -> List[str]:
        """Returns a list of all backend IDs managed by this coordinator."""
        return list(self._backend_adapters.keys())