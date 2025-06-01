# app/ui/main_window.py
import logging
import os
import sys
from typing import Optional, List

from PySide6.QtCore import Qt, Slot, QTimer, QEvent, QSize
from PySide6.QtGui import QFont, QIcon, QCloseEvent, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QApplication, QMessageBox,
    QLabel
)

try:
    # Core application components
    from core.event_bus import EventBus
    from core.chat_manager import ChatManager  # Manager for chat logic
    from core.chat_session_manager import ChatSessionManager  # For session data signals
    from core.chat_message_state_handler import ChatMessageStateHandler  # For message UI state

    # UI components
    from .left_panel import LeftControlPanel
    from ui.dialogs.dialog_service import DialogService
    from .chat_display_area import ChatDisplayArea
    from .chat_input_bar import ChatInputBar
    from .loading_overlay import LoadingOverlay

    # Models and utils
    from models.chat_message import ChatMessage  # For type hinting
    from models.message_enums import MessageLoadingState  # For type hinting
    from utils import constants
    # Project service for window title context
    from services.project_service import ProjectManager, Project, ChatSession

except ImportError as e_main_window:
    # Fallback for critical import errors to allow basic error display
    logging.basicConfig(level=logging.DEBUG)  # Ensure logging is configured
    logging.critical(f"CRITICAL IMPORT ERROR in MainWindow: {e_main_window}", exc_info=True)
    # Attempt to show a Qt message box if QApplication can be instantiated
    # This part is tricky as QApplication might not be available if PySide6 itself failed.
    try:
        _dummy_app = QApplication(sys.argv) if QApplication.instance() is None else QApplication.instance()
        if _dummy_app:  # Check if app instance was successfully created/retrieved
            QMessageBox.critical(None, "Application Startup Error",
                                 f"Failed to import critical UI components:\n{e_main_window}\n\n"
                                 "The application cannot start. Please check the logs and your Python environment.")
    except Exception as msg_e:
        logging.critical(f"Failed to show critical import error message box: {msg_e}")
    sys.exit(1)  # Exit if critical components are missing

logger = logging.getLogger(__name__)


class MainWindow(QWidget):
    """
    The main window of the application, orchestrating the UI layout,
    connecting UI elements to core logic via the EventBus and ChatManager.
    """

    def __init__(self,
                 chat_manager: ChatManager,
                 app_base_path: str,  # For resolving asset paths if needed
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        logger.info("MainWindow initializing...")

        if not isinstance(chat_manager, ChatManager):
            logger.critical("MainWindow requires a valid ChatManager instance.")
            raise TypeError("MainWindow requires a valid ChatManager instance.")

        self.chat_manager = chat_manager
        self._project_manager: ProjectManager = self.chat_manager.get_project_manager()
        self._chat_session_manager: ChatSessionManager = self.chat_manager.get_chat_session_manager()
        self.app_base_path = app_base_path
        self._event_bus = EventBus.get_instance()

        # UI Components
        self.left_panel: Optional[LeftControlPanel] = None
        self.active_chat_display_area: Optional[ChatDisplayArea] = None
        self.active_chat_input_bar: Optional[ChatInputBar] = None
        self.status_label: Optional[QLabel] = None
        self._status_clear_timer: Optional[QTimer] = None
        self._loading_overlay: Optional[LoadingOverlay] = None

        # Services and Handlers
        self.dialog_service: Optional[DialogService] = None
        self._chat_message_state_handler: Optional[ChatMessageStateHandler] = None

        self._current_base_status_text: str = "Status: Initializing..."
        self._current_base_status_color: str = "#abb2bf"  # Default text color

        self._init_core_ui_services()  # Initialize DialogService, etc.
        self._init_ui_layout()  # Setup QSplitter, panels
        self._apply_global_styles()  # Load stylesheet
        self._connect_ui_interactions()  # Connect widget signals
        self._connect_event_bus_listeners()  # Listen to app-wide events
        self._setup_window_properties()  # Title, icon, size
        self._setup_global_shortcuts()  # Keyboard shortcuts

        # Initialize ChatMessageStateHandler *after* ChatDisplayArea and its model are created
        if self.active_chat_display_area and self.active_chat_display_area.get_model():
            self._chat_message_state_handler = ChatMessageStateHandler(self._event_bus, parent=self)
            # Initial registration will happen in _handle_active_session_history_loaded
            logger.info("ChatMessageStateHandler initialized for MainWindow.")
        else:
            logger.error("MainWindow: ChatDisplayArea or its model not available for ChatMessageStateHandler init.")

        logger.info("MainWindow initialized successfully.")
        QTimer.singleShot(100, self._post_init_setup)  # For tasks after UI is shown

    def _init_core_ui_services(self):
        """Initialize UI-related services like DialogService."""
        try:
            self.dialog_service = DialogService(self, self.chat_manager, self._event_bus)
        except Exception as e_ds:
            logger.critical(f"Failed to initialize DialogService in MainWindow: {e_ds}", exc_info=True)
            # This is critical, consider how to handle app startup failure
            QMessageBox.critical(self, "Fatal Error", f"Could not initialize Dialog Service:\n{e_ds}")
            QApplication.quit()  # type: ignore

    def _init_ui_layout(self):
        """Sets up the main layout and instantiates primary UI panels."""
        main_hbox_layout = QHBoxLayout(self)
        main_hbox_layout.setContentsMargins(0, 0, 0, 0)
        main_hbox_layout.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.setObjectName("MainSplitter")
        main_splitter.setHandleWidth(2)  # Make handle slightly more visible
        main_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #353b45; } QSplitter::handle:hover { background-color: #454c57; }")

        self.left_panel = LeftControlPanel(chat_manager=self.chat_manager, parent=main_splitter)
        self.left_panel.setObjectName("LeftPanelContainer")  # For styling if needed
        self.left_panel.setMinimumWidth(270)  # Adjusted min width
        self.left_panel.setMaximumWidth(450)  # Adjusted max width

        right_panel_widget = QWidget(main_splitter)
        right_panel_widget.setObjectName("RightPanelWidget")
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(8, 8, 8, 8)  # Margins for content within right panel
        right_panel_layout.setSpacing(6)

        self.active_chat_display_area = ChatDisplayArea(parent=right_panel_widget)
        self.active_chat_input_bar = ChatInputBar(parent=right_panel_widget)

        right_panel_layout.addWidget(self.active_chat_display_area, 1)  # Chat display takes most space
        right_panel_layout.addWidget(self.active_chat_input_bar)

        # Status Bar
        status_bar_widget = QWidget(self)  # Parent to MainWindow for consistent styling scope
        status_bar_widget.setObjectName("StatusBarWidget")
        status_bar_widget.setFixedHeight(28)  # Fixed height for status bar
        status_bar_layout = QHBoxLayout(status_bar_widget)
        status_bar_layout.setContentsMargins(10, 2, 10, 2)  # Padding for status text
        status_bar_layout.setSpacing(10)
        self.status_label = QLabel(self._current_base_status_text, status_bar_widget)
        self.status_label.setFont(
            QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), getattr(constants, 'CHAT_FONT_SIZE', 10) - 2))
        self.status_label.setObjectName("StatusLabel")  # For QSS
        status_bar_layout.addWidget(self.status_label, 1, Qt.AlignmentFlag.AlignVCenter)  # Align text vertically
        right_panel_layout.addWidget(status_bar_widget)  # Add status bar to right panel layout

        main_splitter.addWidget(self.left_panel)
        main_splitter.addWidget(right_panel_widget)
        main_splitter.setSizes([280, 720])  # Initial sizes
        main_splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch as much
        main_splitter.setStretchFactor(1, 1)  # Right panel stretches

        main_hbox_layout.addWidget(main_splitter)
        self.setLayout(main_hbox_layout)

        # Initialize Loading Overlay (must be after main layout is set)
        self._loading_overlay = LoadingOverlay(parent=self)  # Parent to MainWindow
        self._loading_overlay.hide()  # Start hidden

    def _apply_global_styles(self):
        logger.debug("MainWindow applying global stylesheet...")
        try:
            stylesheet_path = ""
            # Prefer stylesheet in app/ui/ if it exists
            ui_style_path = os.path.join(getattr(constants, 'UI_DIR_PATH', 'app/ui'),
                                         getattr(constants, 'STYLESHEET_FILENAME', 'style.qss'))
            if os.path.exists(ui_style_path):
                stylesheet_path = ui_style_path
            else:  # Fallback to checking other paths from constants
                for path_candidate in getattr(constants, 'STYLE_PATHS_TO_CHECK', []):
                    if os.path.exists(path_candidate):
                        stylesheet_path = path_candidate
                        break

            if stylesheet_path:
                with open(stylesheet_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
                logger.info(f"Loaded global stylesheet from: {stylesheet_path}")
            else:
                logger.warning("Global stylesheet not found. Application will use default Qt styling.")
        except Exception as e_style:
            logger.error(f"Error loading/applying global stylesheet: {e_style}", exc_info=True)

    def _connect_ui_interactions(self):
        """Connect signals from UI elements to EventBus or ChatManager methods."""
        if not self.active_chat_input_bar or not self.active_chat_display_area:
            logger.error("MainWindow: ChatInputBar or ChatDisplayArea not initialized. Cannot connect UI interactions.")
            return

        # ChatInputBar -> EventBus (for ChatManager to pick up)
        self.active_chat_input_bar.sendMessageRequested.connect(
            lambda text, images: self._event_bus.userMessageSubmitted.emit(text, images)
        )
        # ChatDisplayArea -> EventBus (for status updates)
        self.active_chat_display_area.textCopied.connect(
            lambda text, color: self._event_bus.uiTextCopied.emit(text, color)
        )
        # LeftPanel connections are handled within LeftPanel itself, emitting to EventBus.

    def _connect_event_bus_listeners(self):
        """Connect MainWindow slots to EventBus signals from core logic."""
        bus = self._event_bus
        # UI Status and Feedback
        bus.uiStatusUpdateGlobal.connect(self.update_status_bar_message)
        bus.uiErrorGlobal.connect(self._handle_global_error_event)
        bus.uiTextCopied.connect(
            lambda msg, color: self.update_status_bar_message(msg, color, True, 1500))  # Brief display

        # Loading Overlay
        bus.showLoader.connect(self._show_loading_overlay)
        bus.hideLoader.connect(self._hide_loading_overlay)
        bus.updateLoaderMessage.connect(self._update_loading_message)

        # Input Bar Busy State (controlled by various coordinators/handlers)
        bus.uiInputBarBusyStateChanged.connect(self._update_input_bar_busy_state)
        bus.backendBusyStateChanged.connect(self._update_input_bar_busy_state)  # Can also make input bar busy

        # Active Session Data Flow (from ChatSessionManager via ChatManager or direct connection)
        # ChatSessionManager signals are now the source of truth for active session display.
        if self._chat_session_manager:
            self._chat_session_manager.activeSessionChanged.connect(self._handle_active_session_changed_for_display)
            self._chat_session_manager.activeSessionCleared.connect(self._handle_active_session_cleared_for_display)
            self._chat_session_manager.newMessageAddedToActiveSession.connect(self._handle_new_message_for_display)
            self._chat_session_manager.messageInActiveSessionUpdated.connect(self._handle_message_updated_in_display)
            self._chat_session_manager.activeSessionHistoryRefreshed.connect(
                self._handle_active_session_history_refreshed_for_display)
        else:
            logger.error("MainWindow: ChatSessionManager not available. Cannot connect session data signals.")

        # LLM Stream Chunks for active session (from BackendCoordinator)
        # These update the content of a message being streamed.
        bus.messageChunkReceivedForSession.connect(self._handle_message_chunk_for_display)
        # Finalization of a message (from BackendCoordinator), used by ChatMessageStateHandler
        # MainWindow doesn't need to *directly* connect to messageFinalizedForSession for display updates
        # if ChatMessageStateHandler and ChatListModel handle it based on ChatSessionManager signals.
        # However, it was in the original, so let's review if it's needed for anything else here.
        # The ChatMessageStateHandler will update the model's loading state.
        # The LlmRequestHandler (or other coordinators) will update the message content in ChatSessionManager.
        # So, direct connection here might be redundant if CSM signals are comprehensive.
        # Let's rely on CSM signals for content, and CMSH for loading state icons.
        logger.debug("MainWindow EventBus listeners connected.")

    def _setup_window_properties(self):
        self.setWindowTitle(getattr(constants, 'APP_NAME', "AvA Chat"))
        try:
            icon_path = os.path.join(getattr(constants, 'ASSETS_PATH', 'assets'),
                                     getattr(constants, 'APP_ICON_FILENAME', 'Synchat.ico'))
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                logger.warning(f"Application icon not found: {icon_path}")
        except Exception as e_icon:
            logger.error(f"Error setting window icon: {e_icon}")

        self.setMinimumSize(QSize(900, 700))  # Sensible minimum size
        self.resize(QSize(1100, 800))  # Default startup size
        self.update_window_title_with_context()  # Initial title

    def _setup_global_shortcuts(self):
        # Example: Ctrl+N for New Chat Session (handled by LeftPanel now, but could be global)
        # QShortcut(QKeySequence("Ctrl+N"), self, self._event_bus.newChatRequested.emit)

        # Escape key to cancel ongoing LLM operations
        shortcut_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        shortcut_escape.activated.connect(self._handle_escape_key_action)
        logger.debug("Global shortcuts (Escape) set up.")

    def _post_init_setup(self):
        """Tasks to run shortly after the UI is shown and event loop is running."""
        if self.active_chat_input_bar: self.active_chat_input_bar.set_input_focus()
        self.update_status_bar_message("Ready.", "#98c379", True, 2000)  # Initial ready message
        # ApplicationOrchestrator will call chat_manager.initialize(), which handles backend configs.
        # ChatManager.initialize() will also call _check_rag_readiness_and_emit_status.
        # So, no need to call them explicitly from here if AppOrchestrator handles init sequence.

    # --- EventBus Slot Implementations for UI Updates ---
    @Slot(str, str, bool, int)
    def update_status_bar_message(self, message: str, color_hex: str, is_temporary: bool = False,
                                  duration_ms: int = 3000):
        if not self.status_label: return

        self._current_base_status_text = message  # Store as base if not temporary
        self._current_base_status_color = color_hex
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"QLabel#StatusLabel {{ color: {color_hex}; }}")

        if self._status_clear_timer: self._status_clear_timer.stop(); self._status_clear_timer.deleteLater(); self._status_clear_timer = None
        if is_temporary:
            self._status_clear_timer = QTimer(self)
            self._status_clear_timer.setSingleShot(True)
            self._status_clear_timer.timeout.connect(self._clear_temporary_status_message)
            self._status_clear_timer.start(duration_ms)

    def _clear_temporary_status_message(self):
        # Revert to a more persistent "Ready" or last non-temporary status.
        # This needs to be more intelligent, perhaps BackendConfigManager emits current "Ready" state.
        # For now, a simple "Ready."
        # This should ideally fetch current state from BackendConfigManager or similar.
        # Let's assume ChatManager can provide this.
        ready_msg, ready_color = "Ready.", "#98c379"
        if self.chat_manager:
            if self.chat_manager.is_api_ready():  # Check if primary chat LLM is configured
                active_chat_model = self.chat_manager.get_model_for_backend(
                    self.chat_manager.get_current_active_chat_backend_id())
                ready_msg = f"Ready ({active_chat_model.split('/')[-1][:20]}...)" if active_chat_model else "Ready."
            else:
                ready_msg = "Chat LLM Not Configured"
                ready_color = "#e5c07b"

        self.update_status_bar_message(ready_msg, ready_color, False)

    @Slot(str, bool)
    def _handle_global_error_event(self, error_message: str, is_critical: bool):
        self.update_status_bar_message(f"Error: {error_message[:100]}...", "#FF6B6B", True, 7000)
        if is_critical:
            QMessageBox.critical(self, "Critical Application Error", error_message)

    @Slot(bool)
    def _update_input_bar_busy_state(self, is_busy: bool):
        if self.active_chat_input_bar: self.active_chat_input_bar.set_globally_busy_state(is_busy)
        if self.left_panel: self.left_panel.set_panel_enabled_state(not is_busy)  # Also enable/disable left panel

    # --- Slots for ChatSessionManager signals ---
    @Slot(str, str, list)  # project_id, session_id, initial_history
    def _handle_active_session_changed_for_display(self, project_id: str, session_id: str,
                                                   history: List[ChatMessage]):  # type: ignore
        if self.active_chat_display_area and self.active_chat_display_area.get_model():
            logger.info(f"MW: Active session changed to P:{project_id}/S:{session_id}. Loading history to display.")
            self.active_chat_display_area.load_history_into_display(project_id, session_id, history)
            # Register the model with ChatMessageStateHandler for this new context
            if self._chat_message_state_handler:
                self._chat_message_state_handler.register_model_for_project_session(
                    project_id, session_id, self.active_chat_display_area.get_model()  # type: ignore
                )
            self.update_window_title_with_context()
        else:
            logger.error("MW: ChatDisplayArea or its model not available for active session change.")

    @Slot(str, str)  # old_project_id, old_session_id
    def _handle_active_session_cleared_for_display(self, project_id: str, session_id: str):
        if self.active_chat_display_area and self.active_chat_display_area.get_model():
            logger.info(f"MW: Active session P:{project_id}/S:{session_id} cleared. Clearing display.")
            self.active_chat_display_area.clear_display(project_id, session_id)
            # Unregister model from ChatMessageStateHandler
            if self._chat_message_state_handler:
                self._chat_message_state_handler.unregister_model_for_project_session(project_id, session_id)
            self.update_window_title_with_context()  # Title reflects no active session
        else:
            logger.error("MW: ChatDisplayArea or its model not available for active session clear.")

    @Slot(str, str, object)  # project_id, session_id, ChatMessage object
    def _handle_new_message_for_display(self, project_id: str, session_id: str, message: ChatMessage):  # type: ignore
        if self.active_chat_display_area:
            # CDA will internally check if P/S matches its current context
            self.active_chat_display_area.add_message_to_display(project_id, session_id, message)

    @Slot(str, str, str, object)  # project_id, session_id, message_id, updated_message_obj
    def _handle_message_updated_in_display(self, project_id: str, session_id: str, message_id: str,
                                           updated_message: ChatMessage):  # type: ignore
        # This is for full message object replacement/update.
        # ChatDisplayArea's model should find message by ID and update its data, then emit dataChanged.
        if self.active_chat_display_area and self.active_chat_display_area.get_model():
            model = self.active_chat_display_area.get_model()
            row = model.find_message_row_by_id(message_id)  # type: ignore
            if row is not None:
                model.updateMessage(row, updated_message)  # type: ignore # Assuming updateMessage exists or similar
                self.active_chat_display_area._scroll_to_bottom_if_needed()  # type: ignore
            else:
                logger.warning(
                    f"MW: Message {message_id} to update not found in display model for P:{project_id}/S:{session_id}")

    @Slot(str, str, list)  # project_id, session_id, full_new_history
    def _handle_active_session_history_refreshed_for_display(self, project_id: str, session_id: str,
                                                             history: List[ChatMessage]):  # type: ignore
        if self.active_chat_display_area:
            logger.info(f"MW: History refreshed for P:{project_id}/S:{session_id}. Reloading display.")
            self.active_chat_display_area.load_history_into_display(project_id, session_id, history)

    # --- Slots for BackendCoordinator Streaming Signals (for active session) ---
    @Slot(str, str, str, str)  # project_id, session_id, request_id, chunk_text
    def _handle_message_chunk_for_display(self, project_id: str, session_id: str, request_id: str, chunk_text: str):
        """Appends a chunk to a streaming message in the active display area."""
        if self.active_chat_display_area:
            # CDA internally checks if project_id/session_id match its current context
            self.active_chat_display_area.append_chunk_to_message_display(project_id, session_id, request_id,
                                                                          chunk_text)

    # Note: messageFinalizedForSession is now primarily handled by LlmRequestHandler and other coordinators,
    # which then update ChatSessionManager. ChatSessionManager's signals then update the UI.
    # ChatMessageStateHandler also listens to messageFinalizedForSession to update loading icons.

    # --- Loading Overlay Slots ---
    @Slot(str)
    def _show_loading_overlay(self, message: str):
        if self._loading_overlay: self._loading_overlay.show_loading(message)

    @Slot()
    def _hide_loading_overlay(self):
        if self._loading_overlay: self._loading_overlay.hide_loading()

    @Slot(str)
    def _update_loading_message(self, message: str):
        if self._loading_overlay and self._loading_overlay.isVisible():
            self._loading_overlay.update_message(message)

    # --- Other UI Logic ---
    def update_window_title_with_context(self):
        base_title = getattr(constants, 'APP_NAME', "AvA Chat")
        details = []
        current_project: Optional[
            Project] = self._project_manager.get_current_project() if self._project_manager else None  # type: ignore
        current_session: Optional[
            ChatSession] = self._project_manager.get_current_session() if self._project_manager else None  # type: ignore

        if current_project: details.append(f"Project: {current_project.name[:25]}")
        if current_session: details.append(f"Session: {current_session.name[:25]}")

        # Add active chat LLM model from BackendConfigManager via ChatManager
        if self.chat_manager:
            active_chat_model = self.chat_manager.get_model_for_backend(
                self.chat_manager.get_current_active_chat_backend_id())
            if active_chat_model:
                model_short = active_chat_model.split('/')[-1].split(':')[-1][:20]  # Shorten
                details.append(f"LLM: {model_short}")

        self.setWindowTitle(f"{base_title} - [{', '.join(details)}]" if details else base_title)

    @Slot()
    def _handle_escape_key_action(self):
        """Handles the Escape key press, e.g., to cancel ongoing LLM operations."""
        # This should now primarily signal the active coordinators/handlers to cancel.
        # BackendCoordinator has a cancel_current_task(request_id=None) for all.
        # More specific cancellation might be needed.
        if self._backend_coordinator and self._backend_coordinator.is_any_backend_busy():  # type: ignore
            logger.info("MW: Escape pressed. Requesting cancellation of all active backend tasks.")
            self._backend_coordinator.cancel_current_task(None)  # type: ignore
            self.update_status_bar_message("Attempting to cancel AI task(s)...", "#e5c07b", True, 2000)
        elif self._plan_and_code_coordinator and self._plan_and_code_coordinator.is_busy():  # type: ignore
            # TODO: Add cancel method to PlanAndCodeCoordinator
            logger.info("MW: Escape pressed during Plan & Code. (Cancellation TBD)")
        elif self._micro_task_coordinator and self._micro_task_coordinator.is_busy():  # type: ignore
            # TODO: Add cancel method to MicroTaskCoordinator
            logger.info("MW: Escape pressed during Micro-Task. (Cancellation TBD)")

    def resizeEvent(self, event: QEvent):  # type: ignore
        super().resizeEvent(event)
        if self._loading_overlay and self._loading_overlay.isVisible():
            self._loading_overlay.resize(self.size())

    def closeEvent(self, event: QCloseEvent):
        logger.info("MainWindow closeEvent triggered. Performing application cleanup...")
        if self._loading_overlay: self._loading_overlay.hide_loading()
        if self.dialog_service: self.dialog_service.close_non_modal_dialogs()
        if self.chat_manager: self.chat_manager.cleanup()

        # Unregister model from ChatMessageStateHandler if a session was active
        if self._chat_message_state_handler and self._chat_session_manager:
            active_ctx = self._chat_session_manager.get_active_session_context()
            if active_ctx:
                self._chat_message_state_handler.unregister_model_for_project_session(active_ctx[0], active_ctx[1])

        logger.info("MainWindow cleanup finished. Accepting close event.")
        event.accept()

    def showEvent(self, event: QEvent):  # type: ignore
        super().showEvent(event)
        if self.active_chat_input_bar:
            QTimer.singleShot(100, self.active_chat_input_bar.set_input_focus)
        # Initial status message after UI is shown
        QTimer.singleShot(150, self._clear_temporary_status_message)  # This will set a proper ready/not ready message
        self.update_window_title_with_context()