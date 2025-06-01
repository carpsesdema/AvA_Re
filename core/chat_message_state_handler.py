# app/core/chat_message_state_handler.py
import logging
from typing import Optional, Dict, List, Tuple, Any  # Added List and Tuple

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    # Assuming ChatListModel is in ui package as per previous structure
    from ui.chat_list_model import ChatListModel
    from models.message_enums import MessageLoadingState # Corrected path
    from models.chat_message import ChatMessage # Corrected path
except ImportError as e_cmsh:
    logging.getLogger(__name__).critical(f"ChatMessageStateHandler: Critical import error: {e_cmsh}", exc_info=True)
    # Fallback types for type hinting if imports fail
    EventBus = type("EventBus", (object,), {})
    ChatListModel = type("ChatListModel", (object,), {})
    MessageLoadingState = type("MessageLoadingState", (object,), {}) # type: ignore
    ChatMessage = type("ChatMessage", (object,), {}) # type: ignore
    raise

logger = logging.getLogger(__name__)


class ChatMessageStateHandler(QObject):
    """
    Manages the loading state of chat messages in associated UI models.
    Listens to EventBus signals from LLM operations and updates the
    visual state of messages (e.g., loading, completed, error).
    This handler is designed to work with multiple ChatListModel instances,
    each potentially representing a different chat session or project context.
    """

    def __init__(self,
                 event_bus: EventBus,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(event_bus, EventBus): # type: ignore
            logger.critical("ChatMessageStateHandler requires a valid EventBus instance.")
            raise TypeError("ChatMessageStateHandler requires a valid EventBus instance.")

        self._event_bus = event_bus
        # Stores ChatListModel instances, keyed by (project_id, session_id) tuple.
        # This allows handling multiple concurrent chat sessions displayed in the UI.
        self._active_models: Dict[Tuple[str, str], ChatListModel] = {}

        self._connect_event_bus_subscriptions()
        logger.info("ChatMessageStateHandler initialized and connected to EventBus signals.")

    def _connect_event_bus_subscriptions(self):
        """Connects to relevant EventBus signals for message state updates."""
        # Handles the start of an LLM stream for a specific request.
        # This is used to set the message's visual state to 'LOADING'.
        self._event_bus.llmStreamStarted.connect(self._handle_llm_stream_started)

        # Handles the finalization of an LLM response (completion or error) for a specific session.
        # This updates the message's content and sets its state to 'COMPLETED' or 'ERROR'.
        self._event_bus.messageFinalizedForSession.connect(self._handle_message_finalized_for_session)

        # Listens for when a session's history is loaded into the UI.
        # This is primarily for logging/debugging; model registration is explicit.
        self._event_bus.activeSessionHistoryLoaded.connect(self._handle_active_session_history_loaded_info)

        # Listens for when a session's display is cleared.
        # This is for logging/debugging; model unregistration is typically handled by the UI component.
        self._event_bus.activeSessionHistoryCleared.connect(self._handle_active_session_cleared_info)


    def register_model_for_project_session(self, project_id: str, session_id: str, model: ChatListModel):
        """
        Registers a ChatListModel instance to be managed for a specific project and session.
        This allows the state handler to update the correct UI model.
        """
        if not all(isinstance(arg, str) and arg.strip() for arg in [project_id, session_id]):
            logger.warning("CMSH: Attempted to register model with invalid project_id or session_id.")
            return
        if not isinstance(model, ChatListModel): # type: ignore
            logger.warning(
                f"CMSH: Attempted to register invalid model type for project '{project_id}', session '{session_id}'. "
                f"Expected ChatListModel, got {type(model)}."
            )
            return

        key = (project_id, session_id)
        if key in self._active_models and self._active_models[key] == model:
            logger.debug(f"CMSH: Model for P:{project_id}/S:{session_id} is already registered.")
            return

        logger.info(f"CMSH: Registering ChatListModel for Project/Session: P:{project_id}/S:{session_id}.")
        self._active_models[key] = model

    def unregister_model_for_project_session(self, project_id: str, session_id: str):
        """
        Unregisters a ChatListModel instance, stopping state updates for it.
        """
        key = (project_id, session_id)
        if key in self._active_models:
            logger.info(f"CMSH: Unregistering ChatListModel for Project/Session: P:{project_id}/S:{session_id}.")
            del self._active_models[key]
        else:
            logger.debug(f"CMSH: No model found to unregister for Project/Session: P:{project_id}/S:{session_id}.")

    def _get_model_from_metadata(self, metadata: Dict[str, Any]) -> Optional[ChatListModel]:
        """
        Retrieves the ChatListModel associated with project/session IDs found in metadata.
        Metadata is expected to come from LLM event signals.
        """
        project_id = metadata.get("project_id")
        session_id = metadata.get("session_id")

        if project_id and session_id:
            key = (project_id, session_id)
            model = self._active_models.get(key)
            if model:
                return model
            else:
                logger.warning(
                    f"CMSH: No ChatListModel registered for Project/Session: P:{project_id}/S:{session_id} "
                    f"(from metadata context). Request ID: {metadata.get('request_id')}. "
                    f"Available model contexts: {list(self._active_models.keys())}"
                )
        else:
            logger.warning(
                f"CMSH: Missing project_id or session_id in event metadata for request {metadata.get('request_id')}. "
                "Cannot determine target model."
            )
        return None


    @Slot(str, str, list)
    def _handle_active_session_history_loaded_info(self, project_id: str, session_id: str, history: List[ChatMessage]): # type: ignore
        """Informational handler for when a session's history is loaded."""
        # This slot is mainly for logging or internal state updates if CMSH needed to track active sessions.
        # The actual registration of the model is done explicitly by the UI component (e.g., MainWindow).
        logger.debug(
            f"CMSH: Noted active session history loaded for P:{project_id} S:{session_id}. "
            "CMSH awaits explicit model registration if this context is displayed."
        )

    @Slot(str, str)
    def _handle_active_session_cleared_info(self, project_id: str, session_id: str):
        """Informational handler for when an active session's display is cleared."""
        # This is also primarily for logging. Unregistration is typically handled by the UI component
        # that owned the model for this P/S context.
        logger.debug(
            f"CMSH: Noted active session display cleared for P:{project_id} S:{session_id}. "
            "Associated model should be unregistered by its owner if no longer in use."
        )

    @Slot(str) # request_id
    def _handle_llm_stream_started(self, request_id: str):
        """
        Handles the start of an LLM stream. Sets the corresponding message to LOADING state.
        Iterates through all registered models to find the message by request_id,
        as the llmStreamStarted signal does not (yet) carry project/session context.
        """
        logger.debug(f"CMSH Event: llmStreamStarted for ReqID '{request_id}'.")
        model_updated = False
        for (pid, sid), model_instance in self._active_models.items():
            if model_instance.find_message_row_by_id(request_id) is not None:
                model_instance.update_message_loading_state_by_id(request_id, MessageLoadingState.LOADING)
                logger.debug(
                    f"CMSH: Set message for ReqID '{request_id}' to LOADING in model for P:{pid}/S:{sid}."
                )
                model_updated = True
                break # Assume request_id is unique across all active models for a given stream start

        if not model_updated:
            logger.warning(
                f"CMSH: No registered model found containing message for ReqID '{request_id}' to mark as LOADING. "
                "This might happen if the message wasn't added to a model before the stream started, "
                "or if the model for that session isn't registered."
            )

    @Slot(str, str, str, ChatMessage, dict, bool) # project_id, session_id, request_id, completed_message_obj, usage_stats_dict, is_error
    def _handle_message_finalized_for_session(self,
                                              project_id: str,
                                              session_id: str,
                                              request_id: str,
                                              completed_message: ChatMessage, # type: ignore
                                              usage_stats_with_metadata: dict, # This dict comes from BackendCoordinator, includes original metadata
                                              is_error: bool):
        """
        Handles message finalization (completion or error) for a specific project and session.
        Updates the message's loading state in the corresponding registered model.
        The ChatListModel itself is responsible for updating its content based on this event.
        """
        logger.debug(
            f"CMSH Event: messageFinalizedForSession for P:{project_id} S:{session_id} ReqID '{request_id}'. IsError: {is_error}"
        )
        target_model = self._active_models.get((project_id, session_id))

        if target_model:
            new_state = MessageLoadingState.ERROR if is_error else MessageLoadingState.COMPLETED
            # The ChatListModel's own slot connected to messageFinalizedForSession handles content updates.
            # This handler primarily ensures the visual loading state is correctly set on the model.
            success_state_update = target_model.update_message_loading_state_by_id(request_id, new_state)
            if success_state_update:
                logger.debug(
                    f"CMSH: Updated loading state for ReqID '{request_id}' to {new_state.name} in model P:{project_id}/S:{session_id}."
                )
            else:
                logger.warning(
                    f"CMSH: Failed to find message with ID '{request_id}' in its model (P:{project_id}/S:{session_id}) "
                    f"to mark as {new_state.name}. The model might have already been updated or cleared."
                )
        else:
            logger.warning(
                f"CMSH: No model registered for P:{project_id}/S:{session_id} (ReqID '{request_id}') "
                f"to handle message finalization. Event metadata: {usage_stats_with_metadata.get('purpose')}"
            )