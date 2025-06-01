# app/core/application_orchestrator.py
import logging
import os
from typing import Optional, Dict, Any, TYPE_CHECKING, List

from PySide6.QtCore import QObject, Slot

# Forward references to handle potential circular dependencies
if TYPE_CHECKING:
    from app.services.project_service import ProjectManager
    from app.core.chat_manager import ChatManager
    from app.services.rag_sync_service import RagSyncService
    from app.services.live_code_analyzer import LiveCodeAnalyzer
    # Only BackendCoordinator is directly instantiated by ApplicationOrchestrator now
    from app.llm.backend_coordinator import BackendCoordinator

logger = logging.getLogger(__name__)

try:
    from core.event_bus import EventBus
    # BackendCoordinator will import its own adapters
    from app.llm.backend_coordinator import BackendCoordinator
    from app.services.upload_service import UploadService
    from app.services.terminal_service import TerminalService
    from app.services.update_service import UpdateService
    from app.services.llm_communication_logger import LlmCommunicationLogger
    from utils import constants

    # Import RAG handler with fallback
    try:
        from app.llm.rag_system import RagSystem as RagHandler
    except ImportError:
        logger.warning("RagHandler (RagSystem) not found. RAG functionality will be limited.")
        RagHandler = None  # type: ignore

    # Import RAG sync service
    try:
        from app.services.rag_sync_service import RagSyncService
    except ImportError:
        logger.warning("RagSyncService not available. Multi-project RAG sync disabled.")
        RagSyncService = None  # type: ignore

    # Import LiveCodeAnalyzer
    try:
        from app.services.live_code_analyzer import LiveCodeAnalyzer
    except ImportError:
        logger.warning("LiveCodeAnalyzer not available. Real-time code intelligence disabled.")
        LiveCodeAnalyzer = None  # type: ignore

except ImportError as e:
    logger.critical(f"Critical import error in ApplicationOrchestrator: {e}", exc_info=True)
    EventBus = type("EventBus", (object,), {})
    BackendCoordinator = type("BackendCoordinator", (object,), {})  # type: ignore
    UploadService = type("UploadService", (object,), {})
    TerminalService = type("TerminalService", (object,), {})
    UpdateService = type("UpdateService", (object,), {})
    LlmCommunicationLogger = type("LlmCommunicationLogger", (object,), {})  # type: ignore
    constants = type("constants", (object,), {})
    RagHandler = None  # type: ignore
    RagSyncService = None  # type: ignore
    LiveCodeAnalyzer = None  # type: ignore
    raise


class ApplicationOrchestrator(QObject):
    """
    Central orchestrator for the application. Initializes and coordinates core services,
    manages application state, and handles high-level events.
    """

    def __init__(self, project_manager: 'ProjectManager', parent: Optional[QObject] = None):
        super().__init__(parent)
        logger.info("ApplicationOrchestrator initializing...")

        if not isinstance(project_manager, QObject):
            logger.critical("ApplicationOrchestrator requires a valid ProjectManager instance.")
            raise TypeError("ApplicationOrchestrator requires a valid ProjectManager instance.")

        self.event_bus = EventBus.get_instance()
        self.project_manager: 'ProjectManager' = project_manager
        self.chat_manager: Optional['ChatManager'] = None
        self.conversation_orchestrator: Optional[Any] = None

        # BackendCoordinator is responsible for instantiating and managing LLM adapters.
        # It will create GeminiAdapter, OllamaAdapter, GPTAdapter internally.
        self.backend_coordinator: 'BackendCoordinator' = BackendCoordinator(parent=self)  # type: ignore

        self._initialize_core_services()
        self._connect_event_bus_handlers()
        logger.info("ApplicationOrchestrator initialization complete.")

    def _initialize_core_services(self):
        """Initializes all core application services."""
        logger.debug("Initializing core services...")
        try:
            self.upload_service: Optional[UploadService] = UploadService() if UploadService else None  # type: ignore
            if self.upload_service:
                logger.info("UploadService initialized.")
            else:
                logger.warning("UploadService could not be initialized.")

            self.rag_handler: Optional[RagHandler] = None  # type: ignore
            if RagHandler and self.upload_service:
                vector_db_service_instance = getattr(self.upload_service, '_vector_db_service', None)
                self.rag_handler = RagHandler(upload_service=self.upload_service,
                                              vector_db_service=vector_db_service_instance)  # type: ignore
                if self.rag_handler: logger.info("RagHandler (RagSystem) initialized.")
            else:
                logger.warning("RagHandler (RagSystem) could not be initialized.")

            self.live_code_analyzer: Optional[LiveCodeAnalyzer] = None  # type: ignore
            if LiveCodeAnalyzer:  # type: ignore
                self.live_code_analyzer = LiveCodeAnalyzer(event_bus=self.event_bus, parent=self)  # type: ignore
                if self.live_code_analyzer: logger.info("LiveCodeAnalyzer initialized.")
            else:
                logger.warning("LiveCodeAnalyzer could not be initialized.")

            self.rag_sync_service: Optional[RagSyncService] = None  # type: ignore
            if RagSyncService and self.upload_service and self.project_manager:  # type: ignore
                self.rag_sync_service = RagSyncService(  # type: ignore
                    upload_service=self.upload_service,  # type: ignore
                    project_manager=self.project_manager,  # type: ignore
                    parent=self  # type: ignore
                )
                if self.rag_sync_service: logger.info("RagSyncService initialized.")
            else:
                logger.warning("RagSyncService could not be initialized.")

            self.terminal_service: Optional[TerminalService] = TerminalService(
                parent=self) if TerminalService else None  # type: ignore
            if self.terminal_service:
                logger.info("TerminalService initialized.")
            else:
                logger.warning("TerminalService could not be initialized.")

            self.update_service: Optional[UpdateService] = UpdateService(
                parent=self) if UpdateService else None  # type: ignore
            if self.update_service:
                logger.info("UpdateService initialized.")
            else:
                logger.warning("UpdateService could not be initialized.")

            self.llm_communication_logger: Optional[LlmCommunicationLogger] = LlmCommunicationLogger(
                parent=self) if LlmCommunicationLogger else None  # type: ignore
            if self.llm_communication_logger:
                logger.info("LlmCommunicationLogger initialized.")
            else:
                logger.warning("LlmCommunicationLogger could not be initialized.")

            try:
                from app.core.conversation_orchestrator import ConversationOrchestrator
                self.conversation_orchestrator = ConversationOrchestrator(event_bus=self.event_bus, parent=self)
                logger.info("ConversationOrchestrator initialized.")
            except ImportError:
                logger.warning(
                    "ConversationOrchestrator not found in app.core. Conversational planning will be limited.")
                self.conversation_orchestrator = None
            except Exception as e_co:
                logger.error(f"Error initializing ConversationOrchestrator: {e_co}", exc_info=True)
                self.conversation_orchestrator = None
        except Exception as e_init_svc:
            logger.critical(f"CRITICAL ERROR during core service initialization: {e_init_svc}", exc_info=True)

    def set_chat_manager(self, chat_manager: 'ChatManager'):
        if not isinstance(chat_manager, QObject):
            logger.critical("Attempted to set invalid ChatManager instance.")
            raise TypeError("Invalid ChatManager instance provided.")
        self.chat_manager = chat_manager
        logger.info("ChatManager reference set in ApplicationOrchestrator.")

        if self.conversation_orchestrator and hasattr(self.conversation_orchestrator, 'set_dependencies'):
            if self.chat_manager and self.backend_coordinator:
                self.conversation_orchestrator.set_dependencies(self.chat_manager, self.backend_coordinator)
            else:
                logger.error(
                    "Cannot set dependencies for ConversationOrchestrator: ChatManager or BackendCoordinator missing.")

        if self.chat_manager and hasattr(self.chat_manager,
                                         'set_conversation_orchestrator') and self.conversation_orchestrator:
            self.chat_manager.set_conversation_orchestrator(self.conversation_orchestrator)  # type: ignore

    def _connect_event_bus_handlers(self):
        logger.debug("Connecting ApplicationOrchestrator EventBus handlers...")
        self.event_bus.createNewSessionForProjectRequested.connect(self._handle_create_new_session_requested)
        self.event_bus.createNewProjectRequested.connect(self._handle_create_new_project_requested)
        self.event_bus.messageFinalizedForSession.connect(self._handle_message_finalized_for_persistence)
        self.event_bus.modificationFileReadyForDisplay.connect(self._log_file_ready_for_display)
        self.event_bus.applyFileChangeRequested.connect(self._handle_apply_file_change_requested)

        if self.update_service:
            self.event_bus.checkForUpdatesRequested.connect(self.update_service.check_for_updates)
            self.event_bus.updateDownloadRequested.connect(self.update_service.download_update)
            self.event_bus.updateInstallRequested.connect(self._handle_update_install_request)
            self.update_service.update_available.connect(self.event_bus.updateAvailable.emit)
            self.update_service.no_update_available.connect(self.event_bus.noUpdateAvailable.emit)
            self.update_service.update_check_failed.connect(self.event_bus.updateCheckFailed.emit)
            self.update_service.update_downloaded.connect(self.event_bus.updateDownloaded.emit)
            self.update_service.update_download_failed.connect(self.event_bus.updateDownloadFailed.emit)
            self.update_service.update_progress.connect(self.event_bus.updateProgress.emit)
            self.update_service.update_status.connect(self.event_bus.updateStatusChanged.emit)
        else:
            logger.warning("UpdateService not initialized, update signals not connected.")

        if self.project_manager:
            self.project_manager.projectDeleted.connect(self._handle_project_deleted_cleanup)
        else:
            logger.warning("ProjectManager not initialized, project signals not connected.")

        self.event_bus.projectFilesSaved.connect(self._handle_project_file_saved_for_sync)
        self.event_bus.projectLoaded.connect(self._handle_project_loaded_ide_event)
        self.event_bus.focusSetOnFiles.connect(self._handle_ide_focus_set_on_files)
        self.event_bus.codeViewerProjectLoaded.connect(self._handle_code_viewer_project_load_event)

    @Slot(str)
    def _handle_update_install_request(self, file_path: str):
        if self.update_service:
            success = self.update_service.apply_update(file_path)
            if success: self.event_bus.applicationRestartRequested.emit()
        else:
            logger.error("Update install requested, but UpdateService is not available.")

    @Slot(str, str)
    def _log_file_ready_for_display(self, filename: str, content: str):
        logger.info(f"File ready for display in CodeViewer: '{filename}' (Content length: {len(content)})")

    @Slot(str, str, str, str)
    def _handle_apply_file_change_requested(self, project_id: str, relative_filepath: str, new_content: str,
                                            focus_prefix: str):
        try:
            if not self.project_manager:
                logger.error("Cannot apply file change: ProjectManager not available.")
                self.event_bus.uiErrorGlobal.emit("Project manager service not ready to save file.", False)
                return

            base_dir_to_use: str
            if focus_prefix and os.path.isabs(focus_prefix) and os.path.isdir(focus_prefix):
                project_files_root = self.project_manager.get_project_files_dir(project_id)
                if os.path.abspath(focus_prefix).startswith(os.path.abspath(project_files_root)):
                    base_dir_to_use = focus_prefix
                else:
                    logger.warning(
                        f"Focus prefix '{focus_prefix}' is outside project files root '{project_files_root}'. Defaulting to project files root.")
                    base_dir_to_use = project_files_root
            elif project_id:
                base_dir_to_use = self.project_manager.get_project_files_dir(project_id)
            else:
                logger.warning(
                    "No project_id or valid focus_prefix for applying file change. Using default fallback directory.")
                base_dir_to_use = self._get_default_generated_files_dir()

            os.makedirs(base_dir_to_use, exist_ok=True)
            unsafe_full_path = os.path.join(base_dir_to_use, relative_filepath)
            full_path = os.path.abspath(unsafe_full_path)

            if not full_path.startswith(os.path.abspath(base_dir_to_use)):
                logger.error(
                    f"Security Alert: Attempt to write file outside of designated directory. Path: '{full_path}', Base: '{base_dir_to_use}'")
                self.event_bus.uiErrorGlobal.emit(
                    f"Blocked attempt to write file outside designated area: {relative_filepath}", True)
                return

            file_dir = os.path.dirname(full_path)
            if file_dir: os.makedirs(file_dir, exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logger.info(f"Successfully applied file change to: {full_path}")

            if project_id: self.event_bus.projectFilesSaved.emit(project_id, full_path, new_content)
            self.event_bus.uiStatusUpdateGlobal.emit(f"File saved: {relative_filepath}", "#4ade80", True, 3000)
        except Exception as e:
            logger.error(f"Error applying file change to '{relative_filepath}': {e}", exc_info=True)
            self.event_bus.uiErrorGlobal.emit(f"Failed to save {relative_filepath}: {e}", False)

    def _get_default_generated_files_dir(self) -> str:
        fallback_dir = os.path.join(constants.USER_DATA_DIR, "generated_files_no_project")  # type: ignore
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir

    def initialize_application_state(self):
        logger.info("ApplicationOrchestrator: Initializing application state...")
        if not self.chat_manager or not self.project_manager:
            logger.error("Cannot initialize application state: ChatManager or ProjectManager not available.")
            return
        try:
            projects = self.project_manager.get_all_projects()
            active_project: Optional[Any] = None  # Using Any for Project due to import cycle potential
            if not projects:
                logger.info("No projects found. Creating a default project.")
                active_project = self.project_manager.create_project(name="Default Project",
                                                                     description="Default project")
            else:
                active_project = self.project_manager.get_current_project()
                if not active_project: active_project = projects[0]
            if not active_project:
                logger.error("Failed to obtain an active project.")
                return
            self.project_manager.switch_to_project(active_project.id)
            active_session: Optional[Any] = None
            if active_project.current_session_id:
                active_session = self.project_manager.get_session_by_id(active_project.current_session_id)
            if not active_session:
                sessions = self.project_manager.get_project_sessions(active_project.id)
                if sessions:
                    active_session = sessions[-1]
                else:
                    active_session = self.project_manager.create_session(active_project.id, "Main Chat")
                self.project_manager.switch_to_session(active_session.id)
            if not active_session:
                logger.error(f"Failed to obtain an active session for project '{active_project.name}'.")
                return
            self.chat_manager.set_active_session(active_project.id, active_session.id, active_session.message_history)
            logger.info(
                f"Application state initialized. Project: '{active_project.name}', Session: '{active_session.name}'.")
        except Exception as e:
            logger.error(f"Error initializing application state: {e}", exc_info=True)
            try:
                if self.project_manager and self.chat_manager:
                    fb_project = self.project_manager.create_project("Fallback Project")
                    fb_session = self.project_manager.create_session(fb_project.id, "Fallback Session")
                    self.chat_manager.set_active_session(fb_project.id, fb_session.id, [])
                    logger.info("Created fallback project/session due to initialization error.")
            except Exception as fallback_error:
                logger.critical(f"Failed to create even a fallback state: {fallback_error}", exc_info=True)

    @Slot(str, str)
    def _handle_create_new_project_requested(self, project_name: str, project_description: str):
        if not self.project_manager: logger.error("ProjectManager not available."); return
        try:
            new_project = self.project_manager.create_project(project_name, project_description)
            self.project_manager.switch_to_project(new_project.id)
            logger.info(f"Created and switched to new project: '{project_name}'")
        except Exception as e:
            logger.error(f"Error creating new project '{project_name}': {e}", exc_info=True)
            self.event_bus.uiErrorGlobal.emit(f"Failed to create project: {e}", False)

    @Slot(str)
    def _handle_create_new_session_requested(self, project_id: str):
        if not self.project_manager: logger.error("ProjectManager not available."); return
        try:
            project = self.project_manager.get_project_by_id(project_id)
            if not project: logger.error(f"Cannot create session: Project {project_id} not found."); return
            num_sessions = len(self.project_manager.get_project_sessions(project_id))
            new_session_name = f"Chat Session {num_sessions + 1}"
            new_session = self.project_manager.create_session(project_id, new_session_name)
            self.project_manager.switch_to_session(new_session.id)
            logger.info(f"Created and switched to new session '{new_session_name}' in project '{project.name}'.")
        except Exception as e:
            logger.error(f"Error creating new session for project {project_id}: {e}", exc_info=True)
            self.event_bus.uiErrorGlobal.emit(f"Failed to create session: {e}", False)

    @Slot(str, str, str, object, dict, bool)
    def _handle_message_finalized_for_persistence(self, project_id: str, session_id: str, request_id: str,
                                                  message_obj: Any, usage_stats_dict: dict, is_error: bool):
        if not self.chat_manager or not self.project_manager: return
        current_pm_project = self.project_manager.get_current_project()
        current_pm_session = self.project_manager.get_current_session()
        current_pm_project_id = current_pm_project.id if current_pm_project else None
        current_pm_session_id = current_pm_session.id if current_pm_session else None
        if project_id == current_pm_project_id and session_id == current_pm_session_id:
            current_history = self.chat_manager.get_current_chat_history()
            self.project_manager.update_current_session_history(current_history)
            logger.debug(f"Persisted updated session history for P:{project_id}/S:{session_id} (ReqID: {request_id})")
        else:
            logger.debug(f"Message finalized for non-active P/S ({project_id}/{session_id}) in PM. Not persisting.")

    @Slot(str)
    def _handle_project_deleted_cleanup(self, deleted_project_id: str):
        if not self.chat_manager or not self.project_manager: return
        current_cm_project_id = self.chat_manager.get_current_project_id()
        if deleted_project_id == current_cm_project_id:
            logger.info(f"Active project '{deleted_project_id}' was deleted. Re-initializing state.")
            self.initialize_application_state()
        else:
            logger.info(f"Project '{deleted_project_id}' deleted (was not active).")

    @Slot(str, str, str)
    def _handle_project_file_saved_for_sync(self, project_id: str, file_path: str, content: str):
        logger.info(f"Orchestrator: File save for project '{project_id}', file: '{os.path.basename(file_path)}'.")

    @Slot(str, str)
    def _handle_project_loaded_ide_event(self, project_id_from_ide: str, project_path_from_ide: str):
        logger.info(
            f"Orchestrator: Project loaded in IDE - ID:'{project_id_from_ide}', Path:'{project_path_from_ide}'.")
        if not self.project_manager: return
        existing_project_by_id = self.project_manager.get_project_by_id(project_id_from_ide)
        if not existing_project_by_id:
            logger.info(f"IDE-loaded project '{project_id_from_ide}' not found in PM. Considering auto-creation/sync.")
            if self.rag_sync_service and hasattr(self.rag_sync_service, 'request_manual_sync'):
                self.rag_sync_service.request_manual_sync(project_id_from_ide, project_path_from_ide)  # type: ignore
        else:
            current_pm_proj = self.project_manager.get_current_project()
            if current_pm_proj is None or current_pm_proj.id != project_id_from_ide:
                self.project_manager.switch_to_project(project_id_from_ide)

    @Slot(str, list)
    def _handle_ide_focus_set_on_files(self, project_id: str, file_paths: List[str]):
        logger.info(f"Orchestrator: IDE focus set on {len(file_paths)} file(s) in project '{project_id}'.")
        if self.chat_manager and hasattr(self.chat_manager, 'update_rag_focus_paths'):
            # self.chat_manager.update_rag_focus_paths(project_id, explicit_focus_paths=file_paths)
            pass
        self.event_bus.uiStatusUpdateGlobal.emit(f"AI focus updated to {len(file_paths)} file(s).", "#61dafb", True,
                                                 3000)

    @Slot(str, str, str)
    def _handle_code_viewer_project_load_event(self, project_name: str, project_path: str, project_id: str):
        logger.info(f"Orchestrator: CodeViewer loaded project '{project_name}' (ID: {project_id}).")
        self.event_bus.projectLoaded.emit(project_id, project_path)

    # Getter methods
    def get_event_bus(self) -> EventBus:
        return self.event_bus  # type: ignore

    def get_backend_coordinator(self) -> 'BackendCoordinator':
        return self.backend_coordinator

    def get_upload_service(self) -> Optional[UploadService]:
        return self.upload_service  # type: ignore

    def get_rag_handler(self) -> Optional[RagHandler]:
        return self.rag_handler  # type: ignore

    def get_rag_sync_service(self) -> Optional['RagSyncService']:
        return self.rag_sync_service

    def get_terminal_service(self) -> Optional[TerminalService]:
        return self.terminal_service  # type: ignore

    def get_update_service(self) -> Optional[UpdateService]:
        return self.update_service  # type: ignore

    def get_llm_communication_logger(self) -> Optional[LlmCommunicationLogger]:
        return self.llm_communication_logger  # type: ignore

    def get_project_manager(self) -> 'ProjectManager':
        return self.project_manager

    def get_live_code_analyzer(self) -> Optional['LiveCodeAnalyzer']:
        return self.live_code_analyzer

    def get_conversation_orchestrator(self) -> Optional[Any]:
        return self.conversation_orchestrator