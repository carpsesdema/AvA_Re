# app/core/chat_ui_updater.py
import logging
from typing import Optional, List # Added List for type hinting

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    # If ChatUiUpdater needed to know about specific backends, it might import constants
    # from utils import constants
except ImportError as e_cuu:
    logging.getLogger(__name__).critical(f"ChatUiUpdater: Critical import error: {e_cuu}", exc_info=True)
    EventBus = type("EventBus", (object,), {}) # type: ignore
    raise

logger = logging.getLogger(__name__)

class ChatUiUpdater(QObject):
    """
    Handles UI updates based on application events.
    Listens to the EventBus and can translate more complex event sequences
    or states into simpler UI update signals if needed, or directly manage
    minor UI elements not covered by MainWindow's direct connections.

    Currently, its role is minimal as MainWindow directly connects to many
    global UI update signals from the EventBus (e.g., uiStatusUpdateGlobal,
    showLoader, hideLoader, uiInputBarBusyStateChanged). This class can be
    expanded if more sophisticated, centralized UI update logic is required
    that doesn't fit well within MainWindow or specific component controllers.
    """

    def __init__(self, event_bus: EventBus, parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(event_bus, EventBus): # type: ignore
            raise TypeError("ChatUiUpdater requires a valid EventBus instance.")

        self._event_bus = event_bus
        self._connect_event_handlers()
        logger.info("ChatUiUpdater initialized.")

    def _connect_event_handlers(self):
        """
        Connect to EventBus signals that might trigger UI updates
        not directly handled by MainWindow or that require some intermediate logic.
        """
        # Example: If backend configuration changes should trigger a specific, perhaps less intrusive,
        # UI notification than the one already handled by LeftPanel or a global status message.
        self._event_bus.backendConfigurationChanged.connect(self._on_backend_configuration_changed_status)
        self._event_bus.ragStatusChanged.connect(self._on_rag_status_changed_notification)

        # Example: Listening to text copied to show a brief confirmation
        self._event_bus.uiTextCopied.connect(self._show_text_copied_confirmation)

        logger.debug("ChatUiUpdater: Event handlers connected.")

    @Slot(str, str, bool, list)
    def _on_backend_configuration_changed_status(self, backend_id: str, model_name: str, is_configured: bool, available_models: List[str]):
        """
        Provides a user-friendly status update when a backend configuration changes.
        This complements the updates shown in LeftPanel.
        """
        # This method might be too verbose if LeftPanel already shows detailed status.
        # It's an example of how ChatUiUpdater *could* centralize such feedback.
        # We need to ensure it doesn't create redundant status messages if ChatManager/BackendConfigManager
        # also emit global status updates for the same events.

        # Let's assume BackendConfigManager is the primary source for detailed status on *active* backends.
        # This handler could provide more general feedback or notifications for *any* backend change if desired.
        # For now, keeping it simple and avoiding redundancy.
        # logger.debug(f"ChatUiUpdater: Noted backend config change: {backend_id} - {model_name} (Configured: {is_configured})")
        pass # Currently, BackendConfigManager and LeftPanel handle this well.

    @Slot(bool, str, str)
    def _on_rag_status_changed_notification(self, is_ready: bool, status_text: str, status_color: str):
        """
        Handles RAG status changes. Can provide additional global notifications
        beyond what LeftPanel might display.
        """
        logger.debug(f"ChatUiUpdater: Noted RAG status - {status_text}")
        # Example: If RAG becomes globally unready during an operation, show a prominent warning.
        # if not is_ready and "Error" in status_text:
        #     self._event_bus.uiStatusUpdateGlobal.emit(f"RAG System Alert: {status_text}", status_color, True, 7000)
        # For now, LeftPanel's display is the primary RAG status indicator.
        pass

    @Slot(str, str)
    def _show_text_copied_confirmation(self, message: str, color: str):
        """
        Shows a temporary global status update when text is copied.
        This is a good example of a small, specific UI update this class can handle.
        """
        self._event_bus.uiStatusUpdateGlobal.emit(message, color, True, 1500) # Show for 1.5 seconds
        logger.debug(f"ChatUiUpdater: Emitted text copied confirmation: '{message}'")


    # If ChatUiUpdater were to manage the loading overlay or input bar state directly
    # (instead of MainWindow connecting to EventBus signals for those), methods would look like this:

    # def show_global_loader(self, message: str):
    #     self._event_bus.showLoader.emit(message)

    # def hide_global_loader(self):
    #     self._event_bus.hideLoader.emit()

    # def update_global_loader_message(self, message: str):
    #     self._event_bus.updateLoaderMessage.emit(message)

    # def set_global_input_bar_busy(self, is_busy: bool):
    #     self._event_bus.uiInputBarBusyStateChanged.emit(is_busy)