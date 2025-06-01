# app/core/llm_request_handler.py
import logging
import uuid
import time  # For performance tracking
from typing import List, Optional, Dict, Any

from PySide6.QtCore import QObject, Slot  # Removed Signal as it might not emit its own specific signals for now

try:
    from models.chat_message import ChatMessage, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from models.message_enums import MessageLoadingState
    from llm.backend_coordinator import BackendCoordinator
    from utils.code_output_processor import CodeOutputProcessor  # Corrected path
    from core.event_bus import EventBus  # Corrected path
    from core.chat_session_manager import ChatSessionManager
except ImportError as e_llmrh:
    logging.getLogger(__name__).critical(f"LlmRequestHandler: Critical import error: {e_llmrh}", exc_info=True)
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MessageLoadingState = type("MessageLoadingState", (object,), {})  # type: ignore
    BackendCoordinator = type("BackendCoordinator", (object,), {})  # type: ignore
    CodeOutputProcessor = type("CodeOutputProcessor", (object,), {})  # type: ignore
    EventBus = type("EventBus", (object,), {})  # type: ignore
    ChatSessionManager = type("ChatSessionManager", (object,), {})  # type: ignore
    MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE = "model", "system", "error"
    raise

logger = logging.getLogger(__name__)


class LlmRequestHandler(QObject):
    """
    Handles specific, relatively self-contained LLM requests. It initiates these
    requests via the BackendCoordinator, tracks them, processes their immediate
    responses (e.g., code extraction), and updates the ChatSessionManager or
    emits events for other services.
    """

    def __init__(self,
                 backend_coordinator: BackendCoordinator,
                 event_bus: EventBus,
                 chat_session_manager: ChatSessionManager,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(backend_coordinator, BackendCoordinator):  # type: ignore
            raise TypeError("LlmRequestHandler requires a valid BackendCoordinator.")
        if not isinstance(event_bus, EventBus):  # type: ignore
            raise TypeError("LlmRequestHandler requires a valid EventBus.")
        if not isinstance(chat_session_manager, ChatSessionManager):  # type: ignore
            raise TypeError("LlmRequestHandler requires a valid ChatSessionManager.")

        self._backend_coordinator = backend_coordinator
        self._event_bus = event_bus
        self._session_manager = chat_session_manager
        self._code_processor = CodeOutputProcessor()  # type: ignore

        self._active_requests: Dict[str, Dict[str, Any]] = {}
        # Key: actual_llm_request_id
        # Value: {
        #   "purpose": str, "start_time": float,
        #   "project_id": str, "session_id": str,
        #   "placeholder_message_id": str, # ID of the message in UI to update
        #   "options": Optional[Dict[str,Any]],
        #   "filename": Optional[str] # For file creation
        # }

        self._connect_event_handlers()
        logger.info("LlmRequestHandler initialized.")

    def _connect_event_handlers(self):
        """Connect to EventBus signals for LLM responses."""
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_response_completed)
        self._event_bus.llmResponseError.connect(self._handle_llm_response_error)
        # Note: llmStreamChunkReceived is handled by MainWindow/ChatDisplayArea for direct UI update.
        # This handler acts on the *final* response or errors for requests it manages.

    def _initiate_request(self,
                          purpose: str,
                          history_to_send: List[ChatMessage],  # type: ignore
                          backend_id: str,
                          options: Optional[Dict[str, Any]],
                          project_id: str,
                          session_id: str,
                          placeholder_text: str,
                          is_modification: bool = False,
                          additional_metadata: Optional[Dict[str, Any]] = None
                          ) -> Optional[str]:
        """Internal helper to initiate an LLM request and manage placeholders."""

        # 1. Create and add placeholder message to the active session
        placeholder_message_id = f"msg_placeholder_{uuid.uuid4().hex[:8]}"
        placeholder_message = ChatMessage(  # type: ignore
            id=placeholder_message_id,
            role=MODEL_ROLE,  # AI is thinking
            parts=[placeholder_text],
            loading_state=MessageLoadingState.LOADING  # type: ignore
        )
        # Add to session manager, which will emit signal for UI
        self._session_manager.add_message(placeholder_message)

        # 2. Initiate the LLM request with BackendCoordinator
        success, error_msg, actual_llm_request_id = self._backend_coordinator.initiate_llm_chat_request(
            target_backend_id=backend_id,
            history_to_send=history_to_send,
            options=options
        )

        if not success or not actual_llm_request_id:
            logger.error(f"LRH: Failed to initiate LLM request for {purpose}: {error_msg}")
            # Update placeholder to error state
            error_ui_message = ChatMessage(  # type: ignore
                id=placeholder_message_id, role=ERROR_ROLE,
                parts=[f"[Error initiating AI request: {error_msg or 'Unknown'}]"],
                loading_state=MessageLoadingState.ERROR  # type: ignore
            )
            self._session_manager.update_message_in_history(error_ui_message)
            self._event_bus.hideLoader.emit()  # Ensure loader is hidden on failure
            return None

        # 3. Store info about this request, linking actual_llm_request_id to placeholder_message_id
        request_details = {
            "purpose": purpose, "start_time": time.time(),
            "project_id": project_id, "session_id": session_id,
            "placeholder_message_id": placeholder_message_id,
            "options": options
        }
        if additional_metadata:
            request_details.update(additional_metadata)
        self._active_requests[actual_llm_request_id] = request_details

        self._event_bus.showLoader.emit(f"{placeholder_text}...")  # Inform UI

        # 4. Start the streaming task
        self._backend_coordinator.start_llm_streaming_task(
            request_id=actual_llm_request_id,
            target_backend_id=backend_id,
            history_to_send=history_to_send,
            is_modification_response_expected=is_modification,
            options=options,
            request_metadata=request_details  # Pass our tracked metadata
        )
        return actual_llm_request_id

    def submit_normal_chat_request(self,
                                   history_to_send: List[ChatMessage],  # type: ignore
                                   backend_id: str,
                                   options: Optional[Dict[str, Any]],
                                   project_id: str,
                                   session_id: str
                                   ) -> Optional[str]:
        logger.debug(f"LRH: Submitting normal chat request to backend '{backend_id}' for P:{project_id}/S:{session_id}")
        return self._initiate_request(
            purpose="normal_chat",
            history_to_send=history_to_send,
            backend_id=backend_id,
            options=options,
            project_id=project_id,
            session_id=session_id,
            placeholder_text="AI is thinking..."
        )

    def submit_simple_file_creation_request(self,
                                            prompt_text: str,  # This is the full prompt for the LLM
                                            filename: str,
                                            backend_id: str,  # Should be a coder model
                                            options: Optional[Dict[str, Any]],
                                            project_id: str,
                                            session_id: str
                                            ) -> Optional[str]:
        logger.debug(
            f"LRH: Submitting file creation for '{filename}' to backend '{backend_id}' for P:{project_id}/S:{session_id}")
        history_for_llm = [ChatMessage(role="user", parts=[prompt_text])]  # type: ignore
        return self._initiate_request(
            purpose="simple_file_creation",
            history_to_send=history_for_llm,
            backend_id=backend_id,
            options=options,
            project_id=project_id,
            session_id=session_id,
            placeholder_text=f"Generating code for {filename}...",
            is_modification=True,  # Expecting code output
            additional_metadata={"filename": filename}
        )

    def submit_project_iteration_request(self,
                                         iteration_prompt_text: str,  # The full prompt for the iteration task
                                         history_for_context: List[ChatMessage],
                                         # type: ignore # Can be empty if prompt is self-contained
                                         backend_id: str,  # Typically a powerful chat/reasoning model
                                         options: Optional[Dict[str, Any]],
                                         project_id: str,
                                         session_id: str
                                         ) -> Optional[str]:
        logger.debug(
            f"LRH: Submitting project iteration request to backend '{backend_id}' for P:{project_id}/S:{session_id}")
        # The iteration_prompt_text should be the main content. History might provide broader context.
        # For now, let's assume the iteration_prompt_text is the primary input message.
        history_to_send = history_for_context + [
            ChatMessage(role="user", parts=[iteration_prompt_text])]  # type: ignore

        return self._initiate_request(
            purpose="project_iteration",
            history_to_send=history_to_send,  # type: ignore
            backend_id=backend_id,
            options=options,
            project_id=project_id,
            session_id=session_id,
            placeholder_text="Analyzing project for iteration...",
            is_modification=True,  # Expecting analytical output, possibly with code
            additional_metadata={}  # Add any specific metadata for iteration
        )

    @Slot(str, object,
          dict)  # request_id from BackendCoordinator, message_obj, usage_stats_dict (includes original_metadata)
    def _handle_llm_response_completed(self, llm_request_id: str, message_obj: Any, usage_stats_dict_from_bc: dict):
        if llm_request_id not in self._active_requests:
            logger.debug(
                f"LRH: Ignoring completed LLM event for ReqID '{llm_request_id}' as it's not managed by this handler or already processed.")
            return

        request_info = self._active_requests.pop(llm_request_id)
        purpose = request_info.get("purpose")
        start_time = request_info.get("start_time", time.time())
        project_id = request_info.get("project_id")
        session_id = request_info.get("session_id")
        placeholder_id = request_info.get("placeholder_message_id")

        logger.info(
            f"LRH: LLM response completed for managed ReqID '{llm_request_id}' (PlaceholderID: {placeholder_id}, Purpose: {purpose}). Time: {time.time() - start_time:.2f}s")
        self._event_bus.hideLoader.emit()

        if not isinstance(message_obj, ChatMessage):  # type: ignore
            logger.error(f"LRH: Received invalid message object type for ReqID {llm_request_id}")
            if placeholder_id:
                error_msg_obj = ChatMessage(id=placeholder_id, role=ERROR_ROLE,
                                            parts=["[System Error: Invalid AI response format received by handler]"],
                                            loading_state=MessageLoadingState.ERROR)  # type: ignore
                self._session_manager.update_message_in_history(error_msg_obj)
            return

        # Update the placeholder message with the final content
        if placeholder_id:
            final_message = ChatMessage(  # type: ignore
                id=placeholder_id,
                role=message_obj.role,  # type: ignore
                parts=message_obj.parts,  # type: ignore
                timestamp=message_obj.timestamp,  # type: ignore
                metadata=message_obj.metadata if message_obj.metadata else {},  # type: ignore
                loading_state=MessageLoadingState.COMPLETED  # type: ignore
            )
            final_message.metadata.update(
                {"original_llm_request_id": llm_request_id, "usage_stats": usage_stats_dict_from_bc})  # type: ignore
            self._session_manager.update_message_in_history(final_message)
        else:
            logger.error(f"LRH: No placeholder_id found for completed request {llm_request_id}. This is unexpected.")
            # Fallback: add as a new message, but this indicates a logic flaw.
            message_obj.id = llm_request_id  # Ensure it has an ID
            message_obj.loading_state = MessageLoadingState.COMPLETED  # type: ignore
            self._session_manager.add_message(message_obj)  # type: ignore

        # --- Purpose-specific post-processing ---
        if purpose == "simple_file_creation":
            filename = request_info.get("filename", "unknown_file.py")
            extracted_code, quality, notes = self._code_processor.process_llm_response(message_obj.text,
                                                                                       filename)  # type: ignore

            if extracted_code:
                logger.info(f"LRH: Code extracted for '{filename}'. Quality: {quality}. Notes: {notes}")
                self._event_bus.modificationFileReadyForDisplay.emit(filename, extracted_code)
                # Add a system message to chat indicating success
                sys_msg_parts = [f"[File '{filename}' generated successfully and is ready in the Code Viewer.]"]
                sys_msg_meta = {"is_file_creation_feedback": True, "filename": filename, "success": True}
                system_message = ChatMessage(role=SYSTEM_ROLE, parts=sys_msg_parts,
                                             metadata=sys_msg_meta)  # type: ignore
                self._session_manager.add_message(system_message)
            else:
                logger.error(f"LRH: Failed to extract code for '{filename}'. Notes: {notes}")
                err_parts = [
                    f"[Code extraction failed for '{filename}'. LLM response might be malformed or contain no code. Notes: {', '.join(notes)}]"]
                err_meta = {"is_file_creation_feedback": True, "filename": filename, "success": False}
                error_for_chat = ChatMessage(role=ERROR_ROLE, parts=err_parts, metadata=err_meta)  # type: ignore
                self._session_manager.add_message(error_for_chat)

        elif purpose == "project_iteration":
            logger.info(f"LRH: Project iteration analysis response received for {llm_request_id}.")
            # The main response is already updated in the history via placeholder.
            # If it contains code blocks, ChatManager's main _handle_llm_response_completed (old version)
            # had logic to extract and send to code viewer. That should be adapted.
            # For now, LlmRequestHandler doesn't auto-extract code from iteration responses.
            # The response itself is the "product" for the user to see in chat.
            # If code viewer integration is desired, it needs specific triggering.
            pass  # Further processing can be added if needed.

        # Other purpose-specific logic can be added here.

    @Slot(str, str)  # request_id from BackendCoordinator, error_message_str
    def _handle_llm_response_error(self, llm_request_id: str, error_message_str: str):
        """Handles LLM errors for requests initiated by this LlmRequestHandler."""
        if llm_request_id not in self._active_requests:
            logger.debug(
                f"LRH: Ignoring error LLM event for ReqID '{llm_request_id}' as it's not managed by this handler or already processed.")
            return

        request_info = self._active_requests.pop(llm_request_id)
        purpose = request_info.get("purpose")
        start_time = request_info.get("start_time", time.time())
        project_id = request_info.get("project_id")
        session_id = request_info.get("session_id")
        placeholder_id = request_info.get("placeholder_message_id")

        logger.error(
            f"LRH: LLM response error for managed ReqID '{llm_request_id}' (PlaceholderID: {placeholder_id}, Purpose: {purpose}) in P:{project_id}/S:{session_id}: {error_message_str}. Time: {time.time() - start_time:.2f}s")
        self._event_bus.hideLoader.emit()

        if placeholder_id:
            error_ui_message = ChatMessage(  # type: ignore
                id=placeholder_id,
                role=ERROR_ROLE,
                parts=[f"[AI Error ({purpose or 'task'}) Failed: {error_message_str}]"],
                loading_state=MessageLoadingState.ERROR  # type: ignore
            )
            self._session_manager.update_message_in_history(error_ui_message)
        else:  # Should not happen
            logger.error(f"LRH: No placeholder_id for error on request {llm_request_id}. Adding new error message.")
            error_ui_message = ChatMessage(role=ERROR_ROLE, parts=[
                f"[AI Error ({purpose or 'task'}): {error_message_str}]"])  # type: ignore
            self._session_manager.add_message(error_ui_message)

        # If it was a file creation that failed at LLM level
        if purpose == "simple_file_creation":
            filename = request_info.get("filename", "unknown_file.py")
            logger.warning(f"LRH: File creation for '{filename}' failed due to LLM error: {error_message_str}")
            # A system message about the failure is already in the placeholder update.
            # Could add another more specific one if needed.
            # sys_err_msg = ChatMessage(role=SYSTEM_ROLE, parts=[f"[System: Generation of file '{filename}' failed.]"])
            # self._session_manager.add_message(sys_err_msg)

    def cancel_request(self, placeholder_message_id: str):
        """
        Requests cancellation of an ongoing LLM task managed by this handler,
        identified by its placeholder_message_id in the UI.
        """
        actual_llm_request_id_to_cancel: Optional[str] = None
        for llm_req_id, info in self._active_requests.items():
            if info.get("placeholder_message_id") == placeholder_message_id:
                actual_llm_request_id_to_cancel = llm_req_id
                break

        if actual_llm_request_id_to_cancel:
            logger.info(
                f"LRH: Requesting cancellation for active LLM ReqID '{actual_llm_request_id_to_cancel}' (linked to PlaceholderID '{placeholder_message_id}').")
            self._backend_coordinator.cancel_current_task(actual_llm_request_id_to_cancel)
            # The _handle_llm_response_error (with a cancellation error typically)
            # will be called by BackendCoordinator, which will then pop it from _active_requests.
        else:
            logger.warning(
                f"LRH: Cannot cancel request for PlaceholderID '{placeholder_message_id}', no matching active LLM request found.")