# app/services/rag_sync_service.py
import logging
import os
import asyncio
from typing import Optional, Dict, Set, List # Added List
from pathlib import Path

from PySide6.QtCore import QObject, Slot, QTimer

try:
    from core.event_bus import EventBus
    # Assuming UploadService and ProjectManager are in the same 'services' package or accessible
    from .upload_service import UploadService
    from .project_service import ProjectManager
    # Constants might be needed for file types or ignored dirs if not passed directly
    from utils import constants
except ImportError as e:
    logging.getLogger(__name__).critical(f"RagSyncService import error: {e}", exc_info=True)
    # Fallback types for type hinting if imports fail
    EventBus = type("EventBus", (object,), {"get_instance": lambda: None})
    UploadService = type("UploadService", (object,), {})
    ProjectManager = type("ProjectManager", (object,), {})
    constants = type("constants", (object,), {})
    raise

logger = logging.getLogger(__name__)


class RagSyncService(QObject):
    """
    Service for synchronizing project files with RAG collections in real-time.
    It listens to file save events and project loading events to keep RAG up-to-date.
    """

    def __init__(self, upload_service: UploadService, project_manager: ProjectManager, parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(upload_service, UploadService): # type: ignore
            logger.critical("RagSyncService requires a valid UploadService instance.")
            raise TypeError("RagSyncService requires a valid UploadService instance.")
        if not isinstance(project_manager, ProjectManager): # type: ignore
            logger.critical("RagSyncService requires a valid ProjectManager instance.")
            raise TypeError("RagSyncService requires a valid ProjectManager instance.")

        self._upload_service = upload_service
        self._project_manager = project_manager
        self._event_bus = EventBus.get_instance()

        # Track sync state: project_id -> set of pending absolute file paths
        self._sync_queue: Dict[str, Set[str]] = {}
        self._sync_timer = QTimer(self) # Use self as parent for QTimer
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self._process_sync_queue)

        self._connect_signals()
        logger.info("RagSyncService initialized and connected to EventBus.")

    def _connect_signals(self):
        """Connect to EventBus signals relevant for RAG synchronization."""
        if self._event_bus:
            # Listen for file saves from any source (e.g., CodeViewer IDE)
            self._event_bus.projectFilesSaved.connect(self._handle_file_saved)
            # Listen for when projects are loaded (e.g., by ApplicationOrchestrator or IDE)
            self._event_bus.projectLoaded.connect(self._handle_project_loaded_for_rag_check)
            # Listen for explicit requests to initialize/re-scan a project's RAG
            self._event_bus.ragProjectInitializationRequested.connect(self._handle_project_rag_initialization_request)
        else:
            logger.error("RagSyncService: EventBus instance is None. Cannot connect signals.")

    @Slot(str, str, str) # project_id, file_path (absolute), content (content might not be needed here if we re-read)
    def _handle_file_saved(self, project_id: str, file_path: str, content: str):
        """
        Handle file save events. Queues the file for RAG synchronization.
        Content is ignored here as UploadService will re-read the file for processing.
        """
        if not project_id or not file_path:
            logger.warning("RagSyncService: Received file saved event with missing project_id or file_path.")
            return

        # Ensure file_path is absolute for consistency
        abs_file_path = os.path.abspath(file_path)
        logger.info(f"RAG sync requested due to file save: Project '{project_id}', File '{os.path.basename(abs_file_path)}'")

        if project_id not in self._sync_queue:
            self._sync_queue[project_id] = set()
        self._sync_queue[project_id].add(abs_file_path)

        # Debounce sync operations: wait a short period to batch multiple quick saves
        self._sync_timer.stop()
        self._sync_timer.start(2000) # 2-second debounce timer

        if self._event_bus:
            self._event_bus.uiStatusUpdateGlobal.emit(
                f"Queued RAG sync: {os.path.basename(abs_file_path)}",
                "#e5c07b", True, 3000 # Amber color for pending
            )

    @Slot(str, str) # project_id, project_path (absolute path to project root)
    def _handle_project_loaded_for_rag_check(self, project_id: str, project_path: str):
        """
        Handle project loading events. Checks if RAG exists, offers initialization if not.
        """
        logger.info(f"Project loaded: '{project_id}' at '{project_path}'. Checking RAG status.")

        if not self._upload_service:
            logger.error("RagSyncService: UploadService not available for RAG check.")
            return

        if self._upload_service.is_vector_db_ready(project_id):
            collection_size = self._upload_service._vector_db_service.get_collection_size(project_id) # type: ignore
            logger.info(f"RAG collection for project '{project_id}' exists (Size: {collection_size} chunks).")
            if self._event_bus:
                status_msg = f"Project RAG ready: {os.path.basename(project_path)} ({collection_size} items)" if collection_size > 0 else f"Project RAG ready (Empty): {os.path.basename(project_path)}"
                self._event_bus.uiStatusUpdateGlobal.emit(status_msg, "#4ade80", True, 4000)
        else:
            logger.info(f"No RAG collection found for project '{project_id}'. Triggering initialization request.")
            if self._event_bus:
                self._event_bus.ragProjectInitializationRequested.emit(project_id, project_path)

    @Slot(str, str) # project_id, project_path (absolute path to project root)
    def _handle_project_rag_initialization_request(self, project_id: str, project_path: str):
        """
        Handle explicit requests to initialize or re-scan RAG for an entire project.
        """
        logger.info(f"RAG initialization requested for project '{project_id}' at path '{project_path}'.")

        if self._event_bus:
            self._event_bus.showLoader.emit(f"Initializing RAG for '{os.path.basename(project_path)}'...")

        # Run the potentially long-running initialization in the background
        asyncio.create_task(self._initialize_project_rag_async(project_id, project_path))

    async def _initialize_project_rag_async(self, project_id: str, project_path: str):
        """Asynchronously initialize RAG for the entire project directory."""
        try:
            if not self._upload_service:
                raise Exception("UploadService is not available.")

            if not await self._upload_service.wait_for_embedder_ready():
                raise Exception("RAG embedder did not become ready in time.")

            logger.info(f"Starting full RAG scan for project '{project_id}', path: '{project_path}'")
            # process_directory_for_context will scan, chunk, embed, and add to DB
            result_message = self._upload_service.process_directory_for_context(project_path, project_id)

            if self._event_bus: self._event_bus.hideLoader.emit() # Hide loader regardless of outcome

            if result_message and result_message.role != ERROR_ROLE: # type: ignore
                success_msg = f"RAG for project '{os.path.basename(project_path)}' initialized/updated."
                logger.info(success_msg + f" Details: {result_message.text}") # type: ignore
                if self._event_bus:
                    self._event_bus.uiStatusUpdateGlobal.emit(success_msg, "#4ade80", True, 5000) # Green
                    self._event_bus.ragProjectSyncCompleted.emit(project_id, project_path, True) # Signal full project sync
            else:
                error_detail = result_message.text if result_message else "Unknown error." # type: ignore
                error_msg = f"Failed to initialize RAG for '{os.path.basename(project_path)}'."
                logger.error(error_msg + f" Details: {error_detail}")
                if self._event_bus:
                    self._event_bus.uiErrorGlobal.emit(f"{error_msg} {error_detail}", False)
                    self._event_bus.ragProjectSyncCompleted.emit(project_id, project_path, False)

        except Exception as e:
            logger.error(f"Error during asynchronous RAG initialization for project '{project_id}': {e}", exc_info=True)
            if self._event_bus:
                self._event_bus.hideLoader.emit()
                self._event_bus.uiErrorGlobal.emit(f"RAG initialization critical error: {e}", False)
                self._event_bus.ragProjectSyncCompleted.emit(project_id, project_path, False)

    def _process_sync_queue(self):
        """Processes the queued file synchronization operations."""
        if not self._sync_queue:
            return

        logger.info(f"Processing RAG sync queue for {len(self._sync_queue)} project(s).")
        # Create a copy for safe iteration as tasks might modify it indirectly or re-queue
        queue_copy = dict(self._sync_queue)
        self._sync_queue.clear() # Clear original queue

        for project_id, file_paths_set in queue_copy.items():
            if file_paths_set: # Ensure there are actually files to process
                asyncio.create_task(self._sync_files_for_project_async(project_id, list(file_paths_set)))
            else:
                logger.debug(f"Skipping project '{project_id}' in sync queue as its file set is empty.")


    async def _sync_files_for_project_async(self, project_id: str, file_paths: List[str]):
        """Asynchronously syncs a list of files for a given project to its RAG collection."""
        try:
            if not self._upload_service:
                logger.error(f"UploadService not available for RAG sync of project '{project_id}'.")
                return

            if not file_paths:
                logger.info(f"No files to sync for project '{project_id}'.")
                return

            logger.info(f"Syncing {len(file_paths)} file(s) to RAG collection for project '{project_id}'.")
            num_files_str = f"{len(file_paths)} file{'s' if len(file_paths) > 1 else ''}"
            if self._event_bus:
                self._event_bus.uiStatusUpdateGlobal.emit(
                    f"Updating RAG for '{self._project_manager.get_project_by_id(project_id).name if self._project_manager.get_project_by_id(project_id) else project_id[:8]}' ({num_files_str})...", # type: ignore
                    "#61dafb", False, 0 # Blue, not temporary
                )

            # Step 1: Remove old chunks for these specific files to avoid duplicates or stale data
            # This assumes UploadService has VectorDBService instance and it's ready.
            vector_db = getattr(self._upload_service, '_vector_db_service', None)
            if vector_db and self._upload_service.is_vector_db_ready(project_id):
                logger.debug(f"Attempting to remove old chunks for {len(file_paths)} files in project '{project_id}'.")
                for file_path in file_paths:
                    # Ensure file_path is absolute when interacting with VectorDBService if it expects that
                    abs_file_path = os.path.abspath(file_path)
                    if vector_db.remove_document_chunks_by_source(project_id, abs_file_path): # type: ignore
                        logger.debug(f"Successfully removed old chunks for '{abs_file_path}' from project '{project_id}'.")
                    else:
                        logger.warning(f"Failed or no chunks to remove for '{abs_file_path}' in project '{project_id}'.")
            else:
                logger.warning(f"VectorDB not ready or not available for project '{project_id}'. Skipping old chunk removal.")


            # Step 2: Process (re-chunk, re-embed, re-add) the updated files
            # process_files_for_context_async handles its own embedder readiness check.
            result_message = await self._upload_service.process_files_for_context_async(file_paths, project_id)

            if result_message and result_message.role != ERROR_ROLE: # type: ignore
                success_msg = f"RAG for '{self._project_manager.get_project_by_id(project_id).name if self._project_manager.get_project_by_id(project_id) else project_id[:8]}' updated ({num_files_str})." # type: ignore
                logger.info(success_msg + f" Details: {result_message.text}") # type: ignore
                if self._event_bus:
                    self._event_bus.uiStatusUpdateGlobal.emit(success_msg, "#4ade80", True, 4000) # Green
                    for file_path in file_paths:
                        self._event_bus.ragProjectSyncCompleted.emit(project_id, file_path, True)
            else:
                error_detail = result_message.text if result_message else "Unknown error." # type: ignore
                error_msg = f"Failed to sync {num_files_str} for '{self._project_manager.get_project_by_id(project_id).name if self._project_manager.get_project_by_id(project_id) else project_id[:8]}'." # type: ignore
                logger.error(error_msg + f" Details: {error_detail}")
                if self._event_bus:
                    self._event_bus.uiStatusUpdateGlobal.emit(f"{error_msg}", "#ef4444", True, 6000) # Red
                    for file_path in file_paths:
                        self._event_bus.ragProjectSyncCompleted.emit(project_id, file_path, False)

        except Exception as e:
            logger.error(f"Error during RAG sync for project '{project_id}': {e}", exc_info=True)
            if self._event_bus:
                self._event_bus.uiStatusUpdateGlobal.emit(
                    f"RAG sync critical error for project '{project_id[:8]}'", "#ef4444", True, 6000
                )
                for file_path in file_paths: # Emit failure for all attempted files in this batch
                     self._event_bus.ragProjectSyncCompleted.emit(project_id, file_path, False)


    def get_sync_status(self, project_id: str) -> Dict[str, Any]:
        """Get the current synchronization status for a project."""
        return {
            'project_id': project_id,
            'pending_files_count': len(self._sync_queue.get(project_id, set())),
            'is_sync_timer_active': self._sync_timer.isActive()
        }

    def request_manual_full_project_sync(self, project_id: str, project_path: str):
        """Manually requests a full re-scan and sync for a given project."""
        logger.info(f"Manual full RAG sync requested for project '{project_id}' at '{project_path}'.")
        if self._event_bus:
            self._event_bus.ragProjectInitializationRequested.emit(project_id, project_path)
        else:
            logger.error("Cannot request manual sync: EventBus not available.")