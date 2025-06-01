# app/ui/dialog_service.py
import logging
from typing import Optional

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QWidget, QDialog, QMessageBox  # QDialog for type hinting

try:
    from core.event_bus import EventBus
    # ChatManager is needed to get context for some dialogs (e.g., current personality)
    from core.chat_manager import ChatManager  # Corrected path

    # Import specific dialogs from their new potential location (e.g., app/ui/dialogs/)
    # Assuming a new subdirectory 'dialogs' within 'app/ui/'
    from ui.dialogs.llm_terminal_window import LlmTerminalWindow
    from ui.dialogs.personality_dialog import EditPersonalityDialog
    from ui.dialogs.code_viewer_dialog import CodeViewerWindow  # This is your existing one
    from ui.dialogs.project_rag_dialog import ProjectRagDialog
    from ui.dialogs.update_dialog import UpdateDialog

    # For UpdateInfo type hint if UpdateService is not directly used here
    from services.update_service import UpdateInfo
    from utils import constants
except ImportError as e_ds:
    logging.getLogger(__name__).critical(f"DialogService: Critical import error: {e_ds}", exc_info=True)
    # Fallback types for type hinting
    EventBus = type("EventBus", (object,), {})  # type: ignore
    ChatManager = type("ChatManager", (object,), {})  # type: ignore
    LlmTerminalWindow = type("LlmTerminalWindow", (QDialog,), {})  # type: ignore
    EditPersonalityDialog = type("EditPersonalityDialog", (QDialog,), {})  # type: ignore
    CodeViewerWindow = type("CodeViewerWindow", (QWidget,), {})  # type: ignore
    ProjectRagDialog = type("ProjectRagDialog", (QDialog,), {})  # type: ignore
    UpdateDialog = type("UpdateDialog", (QDialog,), {})  # type: ignore
    UpdateInfo = type("UpdateInfo", (object,), {})  # type: ignore
    constants = type("constants", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class DialogService(QObject):
    """
    Manages the creation, display, and interaction logic for various
    dialog windows within the application. It listens to EventBus signals
    to trigger dialogs and can also be called directly by other services.
    """

    def __init__(self, parent_window: QWidget, chat_manager: ChatManager, event_bus: EventBus):
        super().__init__(parent_window)  # Parent is typically MainWindow
        self.parent_window = parent_window  # Store reference to the main window for parenting dialogs

        if not isinstance(chat_manager, ChatManager):  # type: ignore
            logger.critical("DialogService initialized with invalid ChatManager instance.")
            raise TypeError("DialogService requires a valid ChatManager instance.")
        if not isinstance(event_bus, EventBus):  # type: ignore
            logger.critical("DialogService initialized with invalid EventBus instance.")
            raise TypeError("DialogService requires a valid EventBus instance.")

        self.chat_manager = chat_manager
        self._event_bus = event_bus

        # Keep references to non-modal dialogs that can be reshown
        self._llm_terminal_window: Optional[LlmTerminalWindow] = None
        self._code_viewer_window: Optional[CodeViewerWindow] = None
        # Modal dialogs are typically created on demand
        self._project_rag_dialog: Optional[ProjectRagDialog] = None  # Can be modal
        self._update_dialog: Optional[UpdateDialog] = None  # Can be modal or non-modal

        self._connect_event_bus_subscriptions()
        logger.info("DialogService initialized and connected to EventBus.")

    def _connect_event_bus_subscriptions(self):
        """Connect to EventBus signals that trigger dialog displays."""
        bus = self._event_bus
        bus.showLlmLogWindowRequested.connect(self.show_llm_terminal_window)
        bus.chatLlmPersonalityEditRequested.connect(self.trigger_edit_personality_dialog)
        bus.viewCodeViewerRequested.connect(lambda: self.show_code_viewer(ensure_creation=True))
        bus.showProjectRagDialogRequested.connect(self.trigger_show_project_rag_dialog)

        # Update-related dialog signals
        bus.updateAvailable.connect(self.show_update_dialog)
        bus.noUpdateAvailable.connect(self._handle_no_update_available_dialog)
        bus.updateCheckFailed.connect(self._handle_update_check_failed_dialog)

        # CodeViewer specific signals (if DialogService manages its lifecycle)
        bus.modificationFileReadyForDisplay.connect(self.display_file_in_code_viewer)
        # Note: CodeViewer's own internal signals (like apply_change_requested) should emit to EventBus directly,
        # not necessarily through DialogService, unless DS needs to mediate.

    @Slot()
    def show_llm_terminal_window(self, ensure_creation: bool = True) -> Optional[LlmTerminalWindow]:
        logger.debug(f"DialogService: Request to show LLM terminal (ensure_creation={ensure_creation}).")
        try:
            if self._llm_terminal_window is None and ensure_creation:
                llm_logger_instance = self.chat_manager.get_llm_communication_logger()
                if llm_logger_instance:
                    # Parent to None to make it a top-level window, or self.parent_window for modality control
                    self._llm_terminal_window = LlmTerminalWindow(llm_logger_instance, parent=None)
                    logger.info("DialogService: Created new LlmTerminalWindow instance.")
                    # The LlmTerminalWindow itself should connect to the logger's signals in its __init__.
                else:
                    logger.error(
                        "DialogService: LLM Communication Logger not available, cannot create LlmTerminalWindow.")
                    QMessageBox.warning(self.parent_window, "Service Unavailable", "LLM Log service is not ready.")
                    return None

            if self._llm_terminal_window:
                self._llm_terminal_window.show()
                self._llm_terminal_window.activateWindow()  # Bring to front
                self._llm_terminal_window.raise_()
                logger.debug("DialogService: LLM terminal window shown and activated.")
            return self._llm_terminal_window
        except Exception as e_term:
            logger.error(f"Error showing LlmTerminalWindow: {e_term}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open LLM Terminal:\n{e_term}")
            return None

    @Slot()
    def show_code_viewer(self, ensure_creation: bool = True) -> Optional[CodeViewerWindow]:
        logger.debug(f"DialogService: Request to show Code Viewer (ensure_creation={ensure_creation}).")
        try:
            if self._code_viewer_window is None and ensure_creation:
                self._code_viewer_window = CodeViewerWindow(parent=self.parent_window)  # Parent to main window
                logger.info("DialogService: Created new CodeViewerWindow instance.")
                # Connect signals from CodeViewer to EventBus if they are for broader app consumption
                if hasattr(self._code_viewer_window, 'apply_change_requested'):
                    self._code_viewer_window.apply_change_requested.connect(
                        self._event_bus.applyFileChangeRequested.emit)  # type: ignore
                if hasattr(self._code_viewer_window, 'projectFilesSaved'):
                    self._code_viewer_window.projectFilesSaved.connect(
                        self._event_bus.projectFilesSaved.emit)  # type: ignore
                if hasattr(self._code_viewer_window, 'focusSetOnFiles'):
                    self._code_viewer_window.focusSetOnFiles.connect(
                        self._event_bus.focusSetOnFiles.emit)  # type: ignore

            if self._code_viewer_window:
                self._code_viewer_window.show()
                self._code_viewer_window.activateWindow()
                self._code_viewer_window.raise_()
                logger.debug("DialogService: Code viewer window shown and activated.")
            return self._code_viewer_window
        except Exception as e_cv:
            logger.error(f"Error showing CodeViewerWindow: {e_cv}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open Code Viewer:\n{e_cv}")
            return None

    @Slot(str, str)  # filename, content
    def display_file_in_code_viewer(self, filename: str, content: str):
        """Displays a given file content in the Code Viewer window."""
        logger.info(f"DialogService: Request to display file '{filename}' in Code Viewer.")
        try:
            code_viewer = self.show_code_viewer(ensure_creation=True)  # Ensure it exists and is visible
            if not code_viewer:
                logger.error("DialogService: Code Viewer window not available for displaying file.")
                QMessageBox.warning(self.parent_window, "Viewer Error", "Code Viewer is not available.")
                return

            # Get project context for the "Apply" functionality
            project_id_context = self.chat_manager.get_chat_session_manager().get_current_project_id()
            project_files_dir = None
            if project_id_context and self.chat_manager.get_project_manager():
                project_files_dir = self.chat_manager.get_project_manager().get_project_files_dir(project_id_context)

            # CodeViewerWindow should have a method to add/update a file
            # The old version had `update_or_add_file`. The new one might have `add_generated_file`.
            if hasattr(code_viewer, 'add_generated_file'):  # Prefer new method if exists
                code_viewer.add_generated_file(filename=filename, content=content, project_id=project_id_context)
            elif hasattr(code_viewer, 'update_or_add_file'):  # Fallback to old method signature
                code_viewer.update_or_add_file(
                    filename=filename, content=content, is_ai_modification=True,  # Assume AI modification
                    original_content=None,  # No original content for new display
                    project_id_for_apply=project_id_context,
                    focus_prefix_for_apply=project_files_dir
                )
            else:
                logger.error("CodeViewerWindow does not have a recognized method to display file content.")
                QMessageBox.warning(self.parent_window, "Viewer Error", "Code Viewer cannot display the file.")
                return

            logger.info(f"DialogService: File '{filename}' sent to Code Viewer.")
        except Exception as e_disp:
            logger.error(f"Error displaying file '{filename}' in Code Viewer: {e_disp}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Display Error",
                                 f"Could not display file in Code Viewer:\n{e_disp}")

    @Slot()
    def trigger_edit_personality_dialog(self) -> None:
        logger.debug("DialogService: Request to show Edit Personality dialog.")
        try:
            # Get current personality from BackendConfigManager via ChatManager
            current_prompt = self.chat_manager.get_current_chat_personality()
            # The persona is tied to the active chat backend
            active_chat_backend_id = self.chat_manager.get_current_active_chat_backend_id()

            dialog = EditPersonalityDialog(current_prompt or "", parent=self.parent_window)  # Pass empty if None
            if dialog.exec() == QDialog.DialogCode.Accepted:
                new_prompt_text = dialog.get_prompt_text()
                logger.info(
                    f"DialogService: Personality dialog accepted. New prompt for '{active_chat_backend_id}': '{new_prompt_text[:50]}...'")
                self._event_bus.chatLlmPersonalitySubmitted.emit(new_prompt_text, active_chat_backend_id)
        except Exception as e_pers_dlg:
            logger.error(f"Error showing EditPersonalityDialog: {e_pers_dlg}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error",
                                 f"Could not open Personality Editor:\n{e_pers_dlg}")

    @Slot()
    def trigger_show_project_rag_dialog(self):
        logger.debug("DialogService: Request to show Project RAG File Add dialog.")
        project_manager = self.chat_manager.get_project_manager()
        if not project_manager:
            QMessageBox.critical(self.parent_window, "Error", "Project manager service is not available.")
            return

        current_project = project_manager.get_current_project()
        if not current_project:
            QMessageBox.information(self.parent_window, "No Active Project",
                                    "Please select or create a project first.")
            return
        try:
            # Create a new instance each time for modal dialogs, or manage visibility if non-modal
            self._project_rag_dialog = ProjectRagDialog(
                project_id=current_project.id, project_name=current_project.name, parent=self.parent_window
            )
            self._project_rag_dialog.exec()  # Modal execution
        except Exception as e_pr_dlg:
            logger.error(f"Error showing ProjectRagDialog: {e_pr_dlg}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open Project RAG Dialog:\n{e_pr_dlg}")

    @Slot(object)  # UpdateInfo object
    def show_update_dialog(self, update_info: UpdateInfo):
        logger.info(f"DialogService: Showing update dialog for version {update_info.version}")
        try:
            if self._update_dialog and self._update_dialog.isVisible(): self._update_dialog.close()
            self._update_dialog = UpdateDialog(update_info, parent=self.parent_window)
            # Connect signals from the dialog to the EventBus for the UpdateService to handle
            self._update_dialog.download_requested.connect(
                lambda info: self._event_bus.updateDownloadRequested.emit(info))
            self._update_dialog.install_requested.connect(
                lambda path: self._event_bus.updateInstallRequested.emit(path))
            # UpdateDialog itself might listen to progress/status from EventBus to update its UI
            self._update_dialog.show()
        except Exception as e:
            logger.error(f"Error showing update dialog: {e}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Update Dialog Error", f"Could not show update dialog:\n{e}")

    @Slot()
    def _handle_no_update_available_dialog(self):
        logger.info("DialogService: No update available information received.")
        QMessageBox.information(self.parent_window, "No Updates",
                                f"You are running the latest version of {getattr(constants, 'APP_NAME', 'AvA')}.\n"
                                f"Current version: {getattr(constants, 'APP_VERSION', 'N/A')}")

    @Slot(str)
    def _handle_update_check_failed_dialog(self, error_message: str):
        logger.error(f"DialogService: Update check failed: {error_message}")
        QMessageBox.warning(self.parent_window, "Update Check Failed",
                            f"Could not check for updates:\n{error_message}\n\nPlease check your internet connection.")

    def close_non_modal_dialogs(self):
        """Closes any non-modal dialogs managed by this service, e.g., on application shutdown."""
        logger.info("DialogService: Closing non-modal dialogs...")
        if self._llm_terminal_window and self._llm_terminal_window.isVisible():
            self._llm_terminal_window.close()
        if self._code_viewer_window and self._code_viewer_window.isVisible():
            self._code_viewer_window.close()
        if self._update_dialog and self._update_dialog.isVisible() and not self._update_dialog.isModal():
            self._update_dialog.close()
        # Modal dialogs like EditPersonalityDialog or ProjectRagDialog close themselves.
        logger.info("DialogService: Non-modal dialogs close requested.")