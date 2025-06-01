# app/core/chat_manager.py
import logging
import asyncio
import os
import re  # For LLM log code streaming (temporary)
import uuid  # For LLM log code streaming (temporary)
import time  # For LLM log code streaming (temporary)
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from PySide6.QtCore import QObject, Slot

# --- Core Application Imports ---
try:
    from core.event_bus import EventBus
    from core.user_input_handler import UserInputHandler, ProcessedInput, UserInputIntent
    from core.chat_session_manager import ChatSessionManager
    from core.llm_request_handler import LlmRequestHandler
    from core.chat_ui_updater import ChatUiUpdater
    from core.backend_config_manager import BackendConfigManager
    from core.chat_router import ChatRouter
    from core.plan_and_code_coordinator import PlanAndCodeCoordinator
    from core.micro_task_coordinator import MicroTaskCoordinator
except ImportError as e_core_deps:
    logging.getLogger(__name__).critical(f"ChatManager: Critical error importing core dependencies: {e_core_deps}",
                                         exc_info=True)
    raise

# --- Model & Service Imports ---
try:
    from models.chat_message import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from models.message_enums import MessageLoadingState  # Not directly used by CM now, but good for context
    from llm.backend_coordinator import BackendCoordinator
    from services.llm_communication_logger import LlmCommunicationLogger
    from services.project_service import ProjectManager
    from services.upload_service import UploadService
    from llm.rag_system import RagSystem as RagHandler
    from utils import constants
    # Prompts are used by coordinators/handlers, not directly by this CM anymore
except ImportError as e_models_services:
    logging.getLogger(__name__).critical(f"ChatManager: Critical error importing models/services: {e_models_services}",
                                         exc_info=True)
    raise

if TYPE_CHECKING:
    from core.application_orchestrator import ApplicationOrchestrator
    from core.conversation_orchestrator import ConversationOrchestrator

logger = logging.getLogger(__name__)


class ChatManager(QObject):
    """
    Manages high-level chat interactions, user input routing (via ChatRouter),
    and coordinates with various processing components. It relies on specialized
    managers for session data, LLM requests, UI updates, and backend configurations.
    """

    def __init__(self, orchestrator: 'ApplicationOrchestrator', parent: Optional[QObject] = None):
        super().__init__(parent)
        logger.info("ChatManager (Refactored V3) initializing...")

        if not isinstance(orchestrator, QObject):
            raise TypeError("ChatManager requires a valid ApplicationOrchestrator instance.")

        self._orchestrator: 'ApplicationOrchestrator' = orchestrator
        self._event_bus: EventBus = self._orchestrator.get_event_bus()
        self._backend_coordinator: BackendCoordinator = self._orchestrator.get_backend_coordinator()
        self._llm_comm_logger: Optional[LlmCommunicationLogger] = self._orchestrator.get_llm_communication_logger()
        self._project_manager: ProjectManager = self._orchestrator.get_project_manager()
        self._upload_service: Optional[UploadService] = self._orchestrator.get_upload_service()
        self._rag_handler: Optional[RagHandler] = self._orchestrator.get_rag_handler()

        # Instantiate core helper modules
        self._user_input_handler = UserInputHandler()
        self._chat_session_manager = ChatSessionManager(event_bus=self._event_bus, parent=self)
        self._backend_config_manager = BackendConfigManager(
            event_bus=self._event_bus, backend_coordinator=self._backend_coordinator, parent=self
        )
        self._llm_request_handler = LlmRequestHandler(
            backend_coordinator=self._backend_coordinator,
            event_bus=self._event_bus,
            chat_session_manager=self._chat_session_manager,
            parent=self
        )
        self._chat_ui_updater = ChatUiUpdater(event_bus=self._event_bus, parent=self)  # Currently lean
        self._chat_router = ChatRouter(parent=self)

        # Instantiate complex task coordinators
        self._plan_and_code_coordinator = PlanAndCodeCoordinator(
            backend_coordinator=self._backend_coordinator, event_bus=self._event_bus,
            llm_comm_logger=self._llm_comm_logger, parent=self
        )
        self._micro_task_coordinator = MicroTaskCoordinator(
            backend_coordinator=self._backend_coordinator, event_bus=self._event_bus,
            llm_comm_logger=self._llm_comm_logger, parent=self
        )
        self._conversation_orchestrator: Optional['ConversationOrchestrator'] = None  # Injected by AppOrchestrator

        # Configure ChatRouter with its handlers
        self._chat_router.set_handlers(
            conversation_orchestrator=None,  # Will be updated via set_conversation_orchestrator
            plan_and_code_coordinator=self._plan_and_code_coordinator,
            micro_task_coordinator=self._micro_task_coordinator,
            llm_request_handler=self._llm_request_handler
        )

        # UI state tracking (minimal, mostly for LLM log visuals)
        self._llm_terminal_opened: bool = False
        self._active_code_streams_for_log: Dict[str, Dict[str, Any]] = {}

        self._connect_event_bus_subscriptions()
        logger.info("ChatManager (Refactored V3) initialized and sub-components created.")

    def set_conversation_orchestrator(self, conversation_orchestrator: 'ConversationOrchestrator'):
        """Called by ApplicationOrchestrator to inject the ConversationOrchestrator."""
        self._conversation_orchestrator = conversation_orchestrator
        if self._chat_router:  # Update router if already initialized
            self._chat_router.set_handlers(
                conversation_orchestrator=self._conversation_orchestrator,
                plan_and_code_coordinator=self._plan_and_code_coordinator,
                micro_task_coordinator=self._micro_task_coordinator,
                llm_request_handler=self._llm_request_handler
            )
        logger.info("ConversationOrchestrator reference set in ChatManager and passed to ChatRouter.")

    def initialize(self):
        """Post-initialization setup tasks for ChatManager."""
        logger.info("ChatManager post-init setup: Triggering RAG status check.")
        # BackendConfigManager handles its own default configurations during its __init__.
        self._check_rag_readiness_and_emit_status()  # Initial RAG status check

    def _connect_event_bus_subscriptions(self):
        logger.debug("CM: Connecting EventBus subscriptions...")
        self._event_bus.userMessageSubmitted.connect(self.handle_user_message)
        self._event_bus.newChatRequested.connect(self.request_new_chat_session)
        self._event_bus.requestRagScanDirectory.connect(self.request_global_rag_scan_directory)
        self._event_bus.requestProjectFilesUpload.connect(self.handle_project_files_upload_request)
        self._event_bus.llmStreamChunkReceived.connect(self._handle_llm_stream_chunk_for_code_log)
        logger.debug("CM: EventBus subscriptions connected.")

    # --- Session Management Facade ---
    @Slot(str, str, list)
    def set_active_session(self, project_id: str, session_id: str, history: List[ChatMessage]):
        logger.info(f"CM: Setting active session via ChatSessionManager to P:{project_id}/S:{session_id}.")
        # Cancel any tasks from previous session if necessary (coordinators should ideally handle this based on context change)
        self._cleanup_on_session_change()
        self._chat_session_manager.set_active_session(project_id, session_id, history)
        self._check_rag_readiness_and_emit_status()

    @Slot()
    def request_new_chat_session(self):
        logger.info("CM: UI requested new chat session.")
        current_pid = self._chat_session_manager.get_current_project_id()
        if not current_pid:
            self._event_bus.uiStatusUpdateGlobal.emit("Cannot start new chat: No active project.", "#e06c75", True,
                                                      3000)
            return
        self._cleanup_on_session_change()  # Clean up before signaling for new session
        self._event_bus.createNewSessionForProjectRequested.emit(current_pid)

    def _cleanup_on_session_change(self):
        """Cleans up state that might be tied to the old session before switching."""
        # LlmRequestHandler and coordinators should ideally cancel their own tasks
        # if they receive new requests for a different session or if ChatManager signals them.
        # For now, a simple cleanup of visual log streams.
        self._cleanup_code_streams_for_log()
        # Example: self._llm_request_handler.cancel_all_active_requests_for_session(old_session_id)
        # Example: self._plan_and_code_coordinator.cancel_sequence_if_active_for_session(old_session_id)

    # --- User Message Handling ---
    @Slot(str, list)
    def handle_user_message(self, text: str, image_data: List[Dict[str, Any]]):
        active_ctx = self._chat_session_manager.get_active_session_context()
        if not active_ctx:
            self._event_bus.uiStatusUpdateGlobal.emit("Error: No active session to send message.", "#FF6B6B", True,
                                                      3000);
            return

        project_id, session_id = active_ctx
        logger.info(f"CM: Handling user message for P:{project_id}/S:{session_id} - '{text[:50]}...'")

        processed_input: ProcessedInput = self._user_input_handler.process_input(text, image_data)
        user_msg_txt = processed_input.original_query.strip()
        if not user_msg_txt and not image_data: return

        if self._is_any_major_coordinator_busy() and processed_input.intent not in [UserInputIntent.NORMAL_CHAT,
                                                                                    UserInputIntent.CONVERSATIONAL_PLANNING]:
            self._event_bus.uiStatusUpdateGlobal.emit("Please wait for the current complex task to complete.",
                                                      "#e5c07b", True, 3000);
            return

        user_msg_parts = [user_msg_txt] if user_msg_txt else []
        if image_data: user_msg_parts.extend(image_data)
        user_message = ChatMessage(role=USER_ROLE, parts=user_msg_parts)
        self._chat_session_manager.add_message(user_message)
        self._log_llm_comm("USER", user_msg_txt)

        # Gather all necessary context for the ChatRouter
        chat_cfg = self._backend_config_manager.get_active_config_for_purpose("chat")
        spec_cfg = self._backend_config_manager.get_active_config_for_purpose("specialized_coder")
        project_dir = self._project_manager.get_project_files_dir(project_id) if self._project_manager else os.getcwd()
        project_obj = self._project_manager.get_project_by_id(project_id) if self._project_manager else None

        # History for LLM should be the most current from session manager
        history_for_llm = self._chat_session_manager.get_current_chat_history()

        asyncio.create_task(self._chat_router.route_request(
            processed_input=processed_input,
            chat_history_for_llm=history_for_llm,  # Pass the full history
            image_data=image_data,
            project_id=project_id, session_id=session_id,
            chat_backend_id=chat_cfg.get("backend_id") or constants.DEFAULT_CHAT_BACKEND_ID,  # type: ignore
            chat_model_name=chat_cfg.get("model_name") or constants.DEFAULT_GEMINI_CHAT_MODEL,  # type: ignore
            chat_temperature=chat_cfg.get("temperature", 0.7),  # type: ignore
            chat_system_prompt=chat_cfg.get("system_prompt"),
            specialized_backend_id=spec_cfg.get("backend_id") or constants.GENERATOR_BACKEND_ID,  # type: ignore
            specialized_model_name=spec_cfg.get("model_name") or constants.DEFAULT_OLLAMA_GENERATOR_MODEL,
            # type: ignore
            specialized_temperature=spec_cfg.get("temperature", 0.2),  # type: ignore
            specialized_system_prompt=spec_cfg.get("system_prompt"),
            current_project_directory=project_dir,
            project_name_for_context=project_obj.name if project_obj else None,  # type: ignore
            project_description_for_context=project_obj.description if project_obj else None  # type: ignore
        ))

    def _is_any_major_coordinator_busy(self) -> bool:
        if self._plan_and_code_coordinator and self._plan_and_code_coordinator.is_busy(): return True
        if self._micro_task_coordinator and self._micro_task_coordinator.is_busy(): return True
        # Add ConversationOrchestrator if it has an is_busy method
        # if self._conversation_orchestrator and self._conversation_orchestrator.is_busy(): return True
        return False

    # --- RAG Related UI Triggers ---
    @Slot(str)
    def request_global_rag_scan_directory(self, dir_path: str):
        if not self._upload_service or not self._upload_service.is_vector_db_ready(
                constants.GLOBAL_COLLECTION_ID):  # type: ignore
            self._event_bus.uiStatusUpdateGlobal.emit(f"RAG system not ready for Global Knowledge.", "#FF6B6B", True,
                                                      4000)
            return
        logger.info(f"CM: Global RAG scan requested for directory: {dir_path}")
        self._event_bus.showLoader.emit(f"Scanning '{os.path.basename(dir_path)}' for Global RAG...")
        rag_feedback_message: Optional[ChatMessage] = None
        try:
            if self._upload_service:
                rag_feedback_message = self._upload_service.process_directory_for_context(dir_path,
                                                                                          collection_id=constants.GLOBAL_COLLECTION_ID)  # type: ignore
        finally:
            self._event_bus.hideLoader.emit()

        active_ctx = self._chat_session_manager.get_active_session_context()
        if rag_feedback_message and active_ctx:
            self._chat_session_manager.add_message(rag_feedback_message)
            self._log_llm_comm("RAG_SCAN_GLOBAL", rag_feedback_message.text)
        elif rag_feedback_message:
            logger.info(f"Global RAG Scan completed (not added to active chat): {rag_feedback_message.text}")
        self._check_rag_readiness_and_emit_status()

    @Slot(list, str)
    def handle_project_files_upload_request(self, file_paths: List[str], project_id_target: str):
        active_ctx = self._chat_session_manager.get_active_session_context()
        pid_to_use = project_id_target or (active_ctx[0] if active_ctx else None)

        if not pid_to_use:
            self._event_bus.uiStatusUpdateGlobal.emit("Cannot add files to RAG: No project context.", "#e06c75", True,
                                                      3000);
            return
        if not self._upload_service or not self._upload_service.is_vector_db_ready(pid_to_use):
            self._event_bus.uiStatusUpdateGlobal.emit(f"RAG not ready for project '{pid_to_use[:8]}...'.", "#FF6B6B",
                                                      True, 4000);
            return

        logger.info(f"CM: Project RAG file upload for {len(file_paths)} files, project: {pid_to_use}")
        self._event_bus.showLoader.emit(f"Adding {len(file_paths)} files to RAG for '{pid_to_use[:8]}...'")
        rag_feedback_message: Optional[ChatMessage] = None
        try:
            if self._upload_service:
                rag_feedback_message = self._upload_service.process_files_for_context(file_paths,
                                                                                      collection_id=pid_to_use)
        finally:
            self._event_bus.hideLoader.emit()

        if rag_feedback_message and active_ctx and active_ctx[0] == pid_to_use:
            self._chat_session_manager.add_message(rag_feedback_message)
            self._log_llm_comm(f"RAG_UPLOAD_P:{pid_to_use[:6]}", rag_feedback_message.text)
        elif rag_feedback_message:
            logger.info(
                f"Project RAG file add for '{pid_to_use}' completed (not to active chat): {rag_feedback_message.text}")
        self._check_rag_readiness_and_emit_status()

    def _check_rag_readiness_and_emit_status(self):
        # This logic should remain to inform the UI about RAG status based on current context
        if not self._upload_service or not hasattr(self._upload_service,
                                                   '_vector_db_service') or not self._upload_service._vector_db_service:
            is_ready, text, color = False, "RAG Not Ready (Service Error)", "#e06c75"
        else:
            active_pid = self._chat_session_manager.get_current_project_id()
            is_ready, text, color = True, "RAG: Ready", "#98c379"  # Default optimistic
            if not self._upload_service._embedder_ready:
                is_ready, text, color = False, "RAG: Initializing embedder...", "#e5c07b"
            elif not active_pid:  # Global context
                g_size = self._upload_service._vector_db_service.get_collection_size(
                    constants.GLOBAL_COLLECTION_ID)  # type: ignore
                if g_size == -1:
                    is_ready, text, color = False, "Global RAG: DB Error", "#e06c75"
                elif g_size == 0:
                    text, color = "Global RAG: Ready (Empty)", "#e5c07b"
                else:
                    text = f"Global RAG: Ready ({g_size} chunks)"
            else:  # Project context
                proj = self._project_manager.get_project_by_id(active_pid) if self._project_manager else None
                p_name = proj.name[:15] if proj else active_pid[:8]
                p_size = self._upload_service._vector_db_service.get_collection_size(active_pid)  # type: ignore
                if p_size == -1:
                    is_ready, text, color = False, f"RAG for '{p_name}...': DB Error", "#e06c75"
                elif p_size == 0:
                    text, color = f"RAG for '{p_name}...': Ready (Empty)", "#e5c07b"
                else:
                    text = f"RAG for '{p_name}...': Ready ({p_size} chunks)"

        self._event_bus.ragStatusChanged.emit(is_ready, text, color)
        logger.debug(f"CM: RAG Status Emitted: Ready={is_ready}, Msg='{text}'")

    # --- LLM Log Code Streaming Visuals ---
    def _handle_llm_stream_chunk_for_code_log(self, request_id: str, chunk: str):
        if not self._llm_comm_logger: return
        stream_state = self._active_code_streams_for_log.get(request_id)
        if not stream_state:
            if chunk.strip().startswith("```"):
                lang_match = re.match(r"```(\w+)", chunk.strip())
                lang = lang_match.group(1) if lang_match else "python"
                block_id = self._llm_comm_logger.start_streaming_code_block(lang)
                self._active_code_streams_for_log[request_id] = {"block_id": block_id, "buffer": ""}
                clean_chunk = chunk.strip()[len(lang_match.group(0) if lang_match else "```"):]
                if clean_chunk.strip(): self._llm_comm_logger.stream_code_chunk(block_id, clean_chunk)
            return
        stream_state["buffer"] += chunk
        if "```" in stream_state["buffer"]:
            code_content, _, _ = stream_state["buffer"].partition("```")
            self._llm_comm_logger.stream_code_chunk(stream_state["block_id"], code_content)
            self._llm_comm_logger.finish_streaming_code_block(stream_state["block_id"])
            self._active_code_streams_for_log.pop(request_id, None)
        else:
            self._llm_comm_logger.stream_code_chunk(stream_state["block_id"], chunk)

    def _cleanup_code_streams_for_log(self):
        if self._active_code_streams_for_log and self._llm_comm_logger:
            for rs_id, state in list(self._active_code_streams_for_log.items()):
                if state.get('block_id'): self._llm_comm_logger.finish_streaming_code_block(state['block_id'])
            self._active_code_streams_for_log.clear()

    # --- Logging Facade ---
    def _log_llm_comm(self, sender: str, message: str):
        if self._llm_comm_logger:
            active_ctx = self._chat_session_manager.get_active_session_context()
            log_prefix = f"[P:{active_ctx[0][:6]}/S:{active_ctx[1][:6]}]" if active_ctx else "[NoSession]"
            self._llm_comm_logger.log_message(f"{log_prefix} {sender}", message)
            if not self._llm_terminal_opened:
                self._llm_terminal_opened = True
                self._event_bus.showLlmLogWindowRequested.emit()

    # --- Getters for Orchestrator/UI to access managed components ---
    def get_chat_session_manager(self) -> ChatSessionManager:
        return self._chat_session_manager

    def get_llm_request_handler(self) -> LlmRequestHandler:
        return self._llm_request_handler

    def get_chat_ui_updater(self) -> ChatUiUpdater:
        return self._chat_ui_updater

    def get_backend_config_manager(self) -> BackendConfigManager:
        return self._backend_config_manager

    def get_chat_router(self) -> ChatRouter:
        return self._chat_router

    def get_plan_and_code_coordinator(self) -> PlanAndCodeCoordinator:
        return self._plan_and_code_coordinator

    def get_micro_task_coordinator(self) -> MicroTaskCoordinator:
        return self._micro_task_coordinator

    def get_conversation_orchestrator(self) -> Optional['ConversationOrchestrator']:
        return self._conversation_orchestrator

    def get_user_input_handler(self) -> UserInputHandler:
        return self._user_input_handler

    # --- Convenience Getters for UI (delegating to BackendConfigManager) ---
    def get_current_active_chat_backend_id(self) -> str:
        return self._backend_config_manager.get_active_chat_backend_id()

    def get_model_for_backend(self, backend_id: str) -> Optional[str]:
        return self._backend_config_manager.get_active_config_for_purpose("chat").get(
            "model_name") if self._backend_config_manager.get_active_config_for_purpose("chat").get(
            "backend_id") == backend_id else self._backend_config_manager.get_active_config_for_purpose(
            "specialized_coder").get("model_name") if self._backend_config_manager.get_active_config_for_purpose(
            "specialized_coder").get(
            "backend_id") == backend_id else self._backend_coordinator.get_current_configured_model(
            backend_id)  # type: ignore

    def get_current_chat_temperature(self) -> float:
        return self._backend_config_manager.get_active_chat_temperature()

    def set_chat_temperature(self, temperature: float):
        self._backend_config_manager.set_temperature_for_purpose("chat", temperature)

    def get_current_chat_personality(self) -> Optional[str]:
        return self._backend_config_manager.get_active_chat_system_prompt()

    def get_current_active_specialized_backend_id(self) -> str:
        return self._backend_config_manager.get_active_specialized_backend_id()

    def get_current_active_specialized_model_name(self) -> Optional[str]:
        return self._backend_config_manager.get_active_specialized_model_name()

    # --- Status & Busy State ---
    def is_api_ready(self) -> bool:
        return self._backend_config_manager.is_purpose_configured_and_ready("chat")

    def is_specialized_api_ready(self) -> bool:
        return self._backend_config_manager.is_purpose_configured_and_ready("specialized_coder")

    def is_rag_ready(self) -> bool:
        active_pid = self._chat_session_manager.get_current_project_id()
        collection_to_check = active_pid if active_pid else constants.GLOBAL_COLLECTION_ID  # type: ignore
        return self._upload_service.is_vector_db_ready(
            collection_to_check) if self._upload_service else False  # type: ignore

    def is_overall_busy(self) -> bool:  # Checks major coordinators and BackendCoordinator
        return self._is_any_major_coordinator_busy() or \
            (self._backend_coordinator and self._backend_coordinator.is_any_backend_busy())

    # --- Cleanup ---
    def cleanup(self):
        logger.info("ChatManager cleanup initiated...")
        self._cleanup_code_streams_for_log()
        if self._plan_and_code_coordinator and hasattr(self._plan_and_code_coordinator, '_reset_sequence_state'):
            self._plan_and_code_coordinator._reset_sequence_state()
        if self._micro_task_coordinator and hasattr(self._micro_task_coordinator, '_reset_sequence_state'):
            self._micro_task_coordinator._reset_sequence_state()
        # Add cleanup for ConversationOrchestrator if it holds significant state/tasks
        # if self._conversation_orchestrator and hasattr(self._conversation_orchestrator, 'cleanup_all_conversations'):
        #    self._conversation_orchestrator.cleanup_all_conversations()
        logger.info("ChatManager cleanup complete.")