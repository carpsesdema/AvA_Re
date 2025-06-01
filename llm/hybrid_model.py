# app/llm/hybrid_model.py
import asyncio
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple

from PySide6.QtCore import Slot

try:
    # Assuming BackendInterface and BackendCoordinator are in the same package or accessible
    from .backend_interface import BackendInterface
    from .backend_coordinator import BackendCoordinator
    # Models are in models
    from models.chat_message import ChatMessage
    # UserInputHandler might be used for initial query classification
    from core.user_input_handler import UserInputHandler, ProcessedInput, UserInputIntent
    # EventBus might be used if this model needs to emit specific events
    # from core.event_bus import EventBus
except ImportError as e_hm:
    logging.getLogger(__name__).critical(f"HybridModel: Critical import error: {e_hm}", exc_info=True)
    # Fallback types for type hinting
    BackendInterface = type("BackendInterface", (object,), {})  # type: ignore
    BackendCoordinator = type("BackendCoordinator", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    UserInputHandler = type("UserInputHandler", (object,), {})  # type: ignore
    ProcessedInput = type("ProcessedInput", (object,), {})  # type: ignore
    UserInputIntent = type("UserInputIntent", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class HybridModel:
    """
    Implements a hybrid LLM strategy.
    This class will decide which underlying LLM backend (or combination of backends)
    to use based on the query, context, or other defined heuristics.
    It acts as a meta-model or a router on top of the BackendCoordinator.
    """

    def __init__(self,
                 backend_coordinator: BackendCoordinator,
                 user_input_handler: Optional[UserInputHandler] = None):
        """
        Initializes the HybridModel.

        Args:
            backend_coordinator: An instance of BackendCoordinator to access various LLM backends.
            user_input_handler: Optional UserInputHandler for query analysis.
        """
        if not isinstance(backend_coordinator, BackendCoordinator):  # type: ignore
            raise TypeError("HybridModel requires a valid BackendCoordinator instance.")

        self._backend_coordinator = backend_coordinator
        self._user_input_handler = user_input_handler if user_input_handler else UserInputHandler()  # type: ignore

        # Configuration for routing logic (examples)
        self._default_primary_backend_id: Optional[str] = None  # e.g., "gemini_chat_default"
        self._default_coder_backend_id: Optional[str] = None  # e.g., "ollama_generator_default"

        # You might load these from a config file or set them dynamically
        self.set_default_backends()

        logger.info("HybridModel initialized.")

    def set_default_backends(self,
                             primary_backend_id: Optional[str] = None,
                             coder_backend_id: Optional[str] = None):
        """
        Sets the default backend IDs for different task types.
        These can be overridden by more specific routing logic.
        """
        # These would typically come from constants or a configuration manager
        from utils import constants  # Lazy import for defaults
        self._default_primary_backend_id = primary_backend_id or getattr(constants, "DEFAULT_CHAT_BACKEND_ID", None)
        self._default_coder_backend_id = coder_backend_id or getattr(constants, "GENERATOR_BACKEND_ID", None)
        logger.info(
            f"HybridModel default backends set: Primary='{self._default_primary_backend_id}', Coder='{self._default_coder_backend_id}'")

    def _select_backend_for_request(self,
                                    processed_input: ProcessedInput,
                                    history: Optional[List[ChatMessage]] = None) -> Tuple[
        Optional[str], Optional[Dict[str, Any]]]:  # type: ignore
        """
        Determines which backend and model to use for a given request.
        This is the core routing logic of the hybrid model.

        Args:
            processed_input: The processed user input from UserInputHandler.
            history: The current chat history, which might influence backend selection.

        Returns:
            A tuple (backend_id, options_for_backend) or (None, None) if no suitable backend.
        """
        # Placeholder for your routing logic.
        # This could be based on:
        # - processed_input.intent
        # - Keywords in processed_input.processed_query
        # - Presence of code in history or query
        # - User preferences
        # - Model capabilities (e.g., vision, coding strength)

        intent = processed_input.intent
        query = processed_input.processed_query

        # Example basic routing:
        if intent == UserInputIntent.FILE_CREATION_REQUEST or \
                intent == UserInputIntent.PLAN_THEN_CODE_REQUEST or \
                intent == UserInputIntent.MICRO_TASK_REQUEST or \
                "code" in query.lower() or "python" in query.lower() or "javascript" in query.lower():
            if self._default_coder_backend_id and self._backend_coordinator.is_backend_configured(
                    self._default_coder_backend_id):
                logger.info(
                    f"HybridModel: Routing to coder backend '{self._default_coder_backend_id}' for intent '{intent.name}'.")
                # Options could be specific to coding tasks, e.g., lower temperature
                return self._default_coder_backend_id, {"temperature": 0.2}
            elif self._default_primary_backend_id and self._backend_coordinator.is_backend_configured(
                    self._default_primary_backend_id):
                logger.warning(
                    f"HybridModel: Coder backend not configured or specified. Falling back to primary backend '{self._default_primary_backend_id}' for coding-related task.")
                return self._default_primary_backend_id, {
                    "temperature": 0.5}  # Slightly higher temp for general model on code

        # Default to primary backend for other intents or general chat
        if self._default_primary_backend_id and self._backend_coordinator.is_backend_configured(
                self._default_primary_backend_id):
            logger.info(
                f"HybridModel: Routing to primary backend '{self._default_primary_backend_id}' for intent '{intent.name}'.")
            return self._default_primary_backend_id, {"temperature": 0.7}  # Default chat temperature

        logger.error("HybridModel: No suitable or configured backend found for the request.")
        return None, None

    async def get_hybrid_response_stream(self,
                                         original_query: str,  # The raw user input
                                         history: List[ChatMessage],  # type: ignore
                                         image_data: Optional[List[Dict[str, Any]]] = None,  # For multimodal
                                         # Context for selecting backend and forming request
                                         project_id: Optional[str] = None,
                                         session_id: Optional[str] = None
                                         ) -> AsyncGenerator[
        Dict[str, Any], None]:  # Yields dicts: {'type': 'chunk'/'error'/'info', 'content': ..., 'backend_id': ...}
        """
        Processes a query using the hybrid strategy and streams the response.

        This method will:
        1. Analyze the input query (potentially using UserInputHandler).
        2. Select the appropriate backend(s) based on the hybrid logic.
        3. Prepare the history and options for the selected backend.
        4. Initiate the request via BackendCoordinator.
        5. Stream back chunks, potentially adding metadata about which backend was used.
        """
        if not self._user_input_handler:
            logger.error("HybridModel: UserInputHandler not available. Cannot process query effectively.")
            yield {"type": "error", "content": "Internal error: Query processor missing.",
                   "backend_id": "hybrid_system"}
            return

        processed_input = self._user_input_handler.process_input(original_query, image_data or [])

        selected_backend_id, backend_options = self._select_backend_for_request(processed_input, history)

        if not selected_backend_id:
            logger.error(f"HybridModel: No backend selected for query: '{original_query[:50]}...'")
            yield {"type": "error", "content": "Could not determine an appropriate AI model for your request.",
                   "backend_id": "hybrid_system"}
            return

        logger.info(f"HybridModel: Using backend '{selected_backend_id}' for query '{original_query[:50]}...'")
        yield {"type": "info", "content": f"Routing to {selected_backend_id}...", "backend_id": selected_backend_id}

        # Prepare history for the chosen backend.
        # The `_format_history_for_api` method is specific to each adapter.
        # The HybridModel should pass the generic ChatMessage history to BackendCoordinator,
        # and the BackendCoordinator will ensure the chosen adapter formats it.

        # Prepare options
        request_options = backend_options or {}
        # If global temperature is managed elsewhere (e.g., BackendConfigManager through ChatManager)
        # and should override, fetch it here. For now, _select_backend_for_request provides options.

        try:
            # Initiate request using BackendCoordinator
            # The BackendCoordinator needs a request_id and metadata for its own eventing.
            # This HybridModel acts like a client to the BackendCoordinator here.

            # Step 1: Initiate (gets a request_id)
            success, error_msg, request_id = self._backend_coordinator.initiate_llm_chat_request(
                target_backend_id=selected_backend_id,
                history_to_send=history,
                options=request_options
            )

            if not success or not request_id:
                logger.error(f"HybridModel: Failed to initiate LLM request via BackendCoordinator: {error_msg}")
                yield {"type": "error", "content": f"Error with AI backend: {error_msg}",
                       "backend_id": selected_backend_id}
                return

            # Step 2: Start streaming task (BackendCoordinator handles its own async task and event emissions)
            # The HybridModel needs to subscribe to the EventBus signals (llmStreamChunkReceived, llmResponseCompleted, llmResponseError)
            # that are emitted by BackendCoordinator, filtered by this request_id, to then yield its own chunks.
            # This creates a slight challenge: how does this async generator wait for those events?
            #
            # Alternative: BackendCoordinator's stream method could directly return the async generator
            # from the adapter, and HybridModel just yields from it. This is simpler.
            # Let's assume BackendCoordinator's _internal_get_response_stream can be adapted or a new method added.

            # For now, let's assume a simplified path where HybridModel directly uses the adapter's stream
            # This means HybridModel might need more direct access or BackendCoordinator needs a more direct streaming passthrough.
            # This is a tricky part of the architecture.
            #
            # The most straightforward way without complex event waiting here is:
            # 1. HybridModel selects backend.
            # 2. HybridModel asks BackendCoordinator to get a stream from that backend *for it*.
            #    This means BackendCoordinator needs a method like `stream_from_backend(backend_id, history, options)`
            #    which returns the AsyncGenerator from the adapter.

            # Let's refine: BackendCoordinator's _internal_get_response_stream already emits
            # messageChunkReceivedForSession. If HybridModel is to be generic, it shouldn't depend on session_ids.
            # It should have its own way to get chunks for a request_id it makes.

            # Simpler approach for now: If HybridModel is used by something that *does* have a session context
            # (like ChatManager routing through ChatRouter to here), that context can be used to listen.
            # This means HybridModel might not be the one *yielding* chunks directly in this fashion
            # but rather *deciding* which backend ChatManager (via LlmRequestHandler) should use.

            # Let's redefine HybridModel's role:
            # It's primarily a *decision-making* component.
            # The actual call and streaming are handled by LlmRequestHandler or other coordinators.
            # So, `get_hybrid_response_stream` might not be the right method here.
            # Instead, it should provide `get_routing_decision`.

            # For this iteration, let's assume this method is called by a component that will handle the events.
            # This is not ideal for SRP of HybridModel itself.

            # The BackendCoordinator's `_internal_get_response_stream` emits `llmStreamChunkReceived(request_id, chunk)`
            # and `llmResponseCompleted/Error(request_id, ...)`.
            # This async generator needs to listen to those events for *this specific request_id*.
            # This typically involves creating an asyncio.Queue, having event handlers put items in it,
            # and this generator awaiting items from the queue.

            response_queue = asyncio.Queue()  # type: ignore

            @Slot(str, str)
            def on_chunk(res_request_id, chunk_text):
                if res_request_id == request_id:
                    asyncio.create_task(
                        response_queue.put({"type": "chunk", "content": chunk_text, "backend_id": selected_backend_id}))

            @Slot(str, object, dict)
            def on_complete(res_request_id, msg_obj, usage_stats):
                if res_request_id == request_id:
                    # Also pass placeholder_id if it was used by the caller
                    placeholder_id = request_metadata.get("placeholder_message_id", request_id)  # type: ignore
                    asyncio.create_task(response_queue.put(
                        {"type": "complete", "placeholder_id": placeholder_id, "message_obj": msg_obj,
                         "usage": usage_stats, "backend_id": selected_backend_id}))
                    asyncio.create_task(response_queue.put(None))  # End of stream sentinel

            @Slot(str, str)
            def on_error(res_request_id, err_text):
                if res_request_id == request_id:
                    placeholder_id = request_metadata.get("placeholder_message_id", request_id)  # type: ignore
                    asyncio.create_task(response_queue.put(
                        {"type": "error", "placeholder_id": placeholder_id, "content": err_text,
                         "backend_id": selected_backend_id}))
                    asyncio.create_task(response_queue.put(None))  # End of stream sentinel

            event_bus = self._backend_coordinator._event_bus  # Access event bus used by BC
            event_bus.llmStreamChunkReceived.connect(on_chunk)
            event_bus.llmResponseCompleted.connect(on_complete)  # BC emits this with original request_id
            event_bus.llmResponseError.connect(on_error)  # BC emits this with original request_id

            # Metadata for BackendCoordinator's task
            request_metadata = {
                "purpose": f"hybrid_routed_to_{selected_backend_id}",
                "original_hybrid_query": original_query,
                "project_id": project_id,  # Pass through for context
                "session_id": session_id,  # Pass through for context
                # If the caller of HybridModel used a placeholder, its ID should be here
                # so on_complete/on_error can signal to update the correct UI message.
                # "placeholder_message_id": caller_placeholder_id
            }

            self._backend_coordinator.start_llm_streaming_task(
                request_id=request_id,
                target_backend_id=selected_backend_id,
                history_to_send=history,
                is_modification_response_expected=False,  # Depends on routed task
                options=request_options,
                request_metadata=request_metadata
            )

            while True:
                item = await response_queue.get()
                if item is None:  # End of stream sentinel
                    break
                yield item
                response_queue.task_done()

        except Exception as e:
            logger.error(f"HybridModel: Error during hybrid response stream for backend {selected_backend_id}: {e}",
                         exc_info=True)
            yield {"type": "error", "content": f"Hybrid system error: {e}",
                   "backend_id": selected_backend_id or "hybrid_system"}
        finally:
            # Disconnect the temporary slots
            if 'event_bus' in locals():  # Check if event_bus was defined
                try:
                    event_bus.llmStreamChunkReceived.disconnect(on_chunk)
                    event_bus.llmResponseCompleted.disconnect(on_complete)
                    event_bus.llmResponseError.disconnect(on_error)
                except TypeError:  # Raised if a signal is not connected to a slot
                    pass
                except Exception as e_disconnect:
                    logger.warning(f"HybridModel: Error disconnecting event handlers: {e_disconnect}")

    def get_routing_decision(self,
                             original_query: str,
                             image_data: Optional[List[Dict[str, Any]]] = None,
                             history: Optional[List[ChatMessage]] = None) -> Tuple[
        Optional[str], Optional[Dict[str, Any]], Optional[ProcessedInput]]:  # type: ignore
        """
        Analyzes the input and returns the selected backend_id and options for that backend.
        This is a more direct method for components that will handle the LLM call themselves.

        Returns:
            Tuple of (backend_id, backend_options, processed_input_obj)
        """
        if not self._user_input_handler:
            logger.error("HybridModel: UserInputHandler not available for routing decision.")
            return None, None, None

        processed_input = self._user_input_handler.process_input(original_query, image_data or [])
        backend_id, backend_options = self._select_backend_for_request(processed_input, history)
        return backend_id, backend_options, processed_input

    # Placeholder for other hybrid strategies
    # async def get_combined_response(...) -> ...
    # async def get_critiqued_response(...) -> ...