# main.py
import asyncio
import logging
import os
import signal
import sys
import traceback  # For detailed error logging
from typing import Optional


# --- Setup Logging (as early as possible) ---
def setup_logging():
    """Configures application-wide logging."""
    # Determine log level from environment variable or default to INFO
    log_level_str = os.getenv('AVA_LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Get USER_DATA_DIR from constants if possible, otherwise use a fallback
    log_file_dir = os.path.expanduser("~/.ava_pys6_data_p1")  # Default, might be overridden by constants
    log_file_name = "ava_app.log"  # Default

    try:
        from utils import constants as app_constants  # Try to import our constants
        log_file_dir = app_constants.USER_DATA_DIR
        log_file_name = app_constants.LOG_FILE_NAME
        log_format = app_constants.LOG_FORMAT
        log_date_format = app_constants.LOG_DATE_FORMAT
    except ImportError:
        logger_setup = logging.getLogger(__name__)  # Use a temp logger for this message
        logger_setup.warning("main.py: Could not import utils.constants for logging config. Using defaults.")
        log_format = '%(asctime)s.%(msecs)03d - %(levelname)-8s - [%(name)s:%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
        log_date_format = '%Y-%m-%d %H:%M:%S'

    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, log_file_name)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=log_date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),  # Log to console
            logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  # Log to file
        ]
    )
    # Reduce noise from very verbose libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('google.generativeai').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)  # Chroma can be noisy at INFO
    logging.getLogger('PIL').setLevel(logging.WARNING)


setup_logging()  # Call it immediately
logger = logging.getLogger(__name__)  # Now get the properly configured logger

# --- Application Imports ---
try:
    import qasync  # For asyncio integration with Qt
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt  # , QTimer (QTimer not directly used in this main.py logic)
    from PySide6.QtGui import QIcon

    # Core application structure imports (using new paths)
    from core.application_orchestrator import ApplicationOrchestrator
    from core.chat_manager import ChatManager
    from services.project_service import ProjectManager  # Services are now in app.services
    from ui.main_window import MainWindow  # UI components are in app.ui
    from utils import constants  # For app name, version, icon paths etc.
    # Config needs to be imported early to set up env vars for libraries like sentence-transformers
    import core.config  # This will execute the early setup in config.py

except ImportError as e_main_imports:
    logger.critical(f"Failed to import critical application modules: {e_main_imports}", exc_info=True)
    # Attempt to show a message box even if some Qt parts failed, as QApplication might still be available.
    try:
        _app_for_error = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Fatal Import Error",
                             f"Could not import essential modules: {e_main_imports}\n\n"
                             "The application cannot start. Please check logs and ensure all dependencies are installed correctly.")
    except Exception as e_msgbox:
        logger.critical(f"Failed to show critical import error dialog: {e_msgbox}")
    sys.exit(1)
except Exception as e_unexpected_startup:
    logger.critical(f"Unexpected error during initial imports: {e_unexpected_startup}", exc_info=True)
    sys.exit(1)


class AvaApplicationRunner:
    """
    Manages the application lifecycle, including asynchronous initialization,
    event loop management, and graceful shutdown.
    """

    def __init__(self):
        self.qt_app: Optional[QApplication] = None
        self.async_event_loop: Optional[qasync.QEventLoop] = None
        self.main_window: Optional[MainWindow] = None
        self.project_manager: Optional[ProjectManager] = None
        self.app_orchestrator: Optional[ApplicationOrchestrator] = None
        self.chat_manager: Optional[ChatManager] = None
        self._shutdown_requested = False

    def _setup_signal_handlers(self):
        """Sets up OS signal handlers for graceful termination."""
        signals_to_catch = (signal.SIGTERM, signal.SIGINT)
        if sys.platform != "win32":  # SIGHUP not on windows
            signals_to_catch += (signal.SIGHUP,)

        for sig in signals_to_catch:
            try:
                signal.signal(sig, self._handle_system_signal)
            except (ValueError, OSError) as e:  # ValueError if trying to set signal in non-main thread
                logger.warning(f"Could not set signal handler for {sig}: {e}")

    def _handle_system_signal(self, signum, frame):
        logger.info(f"Received system signal {signal.Signals(signum).name}. Initiating shutdown...")
        if not self._shutdown_requested:
            self._shutdown_requested = True
            # Schedule async shutdown on the event loop if it's running
            if self.async_event_loop and self.async_event_loop.is_running():
                asyncio.ensure_future(self.shutdown_application_async(), loop=self.async_event_loop)
            else:  # If loop not running, try to quit directly (less graceful)
                if self.qt_app: self.qt_app.quit()

    async def initialize_application(self):
        """Asynchronously initializes all core components of the application."""
        logger.info(f"Starting {constants.APP_NAME} v{constants.APP_VERSION} initialization...")

        # 1. Create QApplication instance
        self.qt_app = QApplication.instance() or QApplication(sys.argv)
        self.qt_app.setApplicationName(constants.APP_NAME)
        self.qt_app.setApplicationVersion(constants.APP_VERSION)
        self.qt_app.setOrganizationName("AvAUnified")  # Or your organization

        # 2. Set Application Icon
        try:
            icon_path = os.path.join(constants.ASSETS_PATH, constants.APP_ICON_FILENAME)
            if os.path.exists(icon_path):
                self.qt_app.setWindowIcon(QIcon(icon_path))
            else:
                logger.warning(f"Application icon not found at: {icon_path}")
        except Exception as e_icon:
            logger.error(f"Error setting window icon: {e_icon}")

        # 3. Setup qasync event loop for asyncio and Qt integration
        self.async_event_loop = qasync.QEventLoop(self.qt_app)
        asyncio.set_event_loop(self.async_event_loop)

        # 4. Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # 5. Initialize Core Services (Order matters for dependencies)
        logger.info("Initializing core services...")
        self.project_manager = ProjectManager()  # Manages project data persistence
        self.app_orchestrator = ApplicationOrchestrator(project_manager=self.project_manager)
        self.chat_manager = ChatManager(orchestrator=self.app_orchestrator)
        self.app_orchestrator.set_chat_manager(self.chat_manager)  # Complete wiring

        # Give ConversationOrchestrator (created in AppOrchestrator) its dependencies
        convo_orchestrator = self.app_orchestrator.get_conversation_orchestrator()
        if convo_orchestrator and self.chat_manager and self.app_orchestrator.get_backend_coordinator():
            convo_orchestrator.set_dependencies(self.chat_manager, self.app_orchestrator.get_backend_coordinator())

        # Also pass ConversationOrchestrator to ChatManager if it needs it
        if self.chat_manager and convo_orchestrator:
            self.chat_manager.set_conversation_orchestrator(convo_orchestrator)

        # 6. Initialize Main Window (UI)
        logger.info("Initializing MainWindow...")
        # Determine app_base_path for asset loading etc.
        app_base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(app_base_path):  # If _MEIPASS points to the exe itself
            app_base_path = os.path.dirname(app_base_path)

        self.main_window = MainWindow(chat_manager=self.chat_manager, app_base_path=app_base_path)
        self.main_window.show()
        logger.info("MainWindow shown.")

        # 7. Perform post-UI initialization steps for services
        # ApplicationOrchestrator initializes project/session state, ChatManager initializes backends
        self.app_orchestrator.initialize_application_state()  # Loads/creates projects & sessions
        # ChatManager's initialize will now primarily be about its own setup,
        # as BackendConfigManager (owned by ChatManager) handles backend configs.
        if self.chat_manager: self.chat_manager.initialize()

        logger.info("Asynchronous application initialization complete.")

    async def run_event_loop(self):
        """Runs the Qt event loop using qasync."""
        if not self.qt_app or not self.async_event_loop:
            logger.critical("Application or event loop not initialized. Cannot run.")
            return 1

        logger.info("Starting application event loop...")
        exit_code = 0
        try:
            # qasync's QEventLoop context manager handles running and closing the loop.
            with self.async_event_loop:
                # The loop runs here until app.quit() is called or shutdown is triggered.
                # We can add a monitor for self._shutdown_requested if needed,
                # but qasync should handle Qt's quit signals.
                # self.async_event_loop.run_forever() # This is handled by 'with'

                # Keep the Python part alive while Qt loop runs
                while not self._shutdown_requested:  # and not self.qt_app.closingDown(): # closingDown might not be reliable here
                    await asyncio.sleep(0.1)  # Check periodically
                    if self.qt_app.applicationState() == Qt.ApplicationState.ApplicationQuit:  # More reliable check
                        logger.info("Qt ApplicationState is ApplicationQuit.")
                        self._shutdown_requested = True  # Ensure our flag is set
                        break

                # If loop exited due to external quit, ensure our shutdown is called
                if not self._shutdown_requested:
                    logger.info("Event loop exited, ensuring shutdown.")
                    await self.shutdown_application_async()


        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down...")
            await self.shutdown_application_async()
        except Exception as e_loop:
            logger.critical(f"Unhandled exception in event loop: {e_loop}", exc_info=True)
            exit_code = 1
            await self.shutdown_application_async(is_error=True)  # Attempt graceful shutdown even on error

        logger.info(f"Application event loop finished. Exit code: {exit_code}")
        return exit_code

    async def shutdown_application_async(self, is_error: bool = False):
        """Performs graceful shutdown of application components."""
        if not self._shutdown_requested and not is_error:  # If not already requested, mark it
            self._shutdown_requested = True

        logger.info("Initiating asynchronous application shutdown...")

        # 1. Inform ChatManager and other services to clean up
        if self.chat_manager and hasattr(self.chat_manager, 'cleanup'):
            try:
                self.chat_manager.cleanup()
            except Exception as e_cm_clean:
                logger.error(f"Error during ChatManager cleanup: {e_cm_clean}")

        # 2. Close main window if still open (should trigger its closeEvent)
        if self.main_window and self.main_window.isVisible():
            logger.debug("Closing MainWindow...")
            self.main_window.close()  # This should call MainWindow's closeEvent
            await asyncio.sleep(0.1)  # Give a moment for UI events

        # 3. Stop BackendCoordinator tasks (important for network calls)
        if self.app_orchestrator and self.app_orchestrator.get_backend_coordinator():
            logger.debug("Cancelling active backend tasks in BackendCoordinator...")
            try:
                self.app_orchestrator.get_backend_coordinator().cancel_current_task(None)  # Cancel all
                await asyncio.sleep(0.2)  # Allow cancellation to propagate
            except Exception as e_bc_cancel:
                logger.error(f"Error cancelling BackendCoordinator tasks: {e_bc_cancel}")

        # 4. Close qasync event loop (qasync handles this when 'with' exits)
        # if self.async_event_loop and self.async_event_loop.is_running():
        #     logger.debug("Stopping asyncio event loop.")
        #     self.async_event_loop.stop()

        # 5. Quit QApplication
        if self.qt_app:
            logger.debug("Requesting QApplication quit.")
            self.qt_app.quit()

        logger.info("Asynchronous application shutdown process completed.")


async def main_async_wrapper():
    """Wrapper to run the application and handle its lifecycle."""
    runner = AvaApplicationRunner()
    exit_code = 1  # Default to error
    try:
        await runner.initialize_application()
        exit_code = await runner.run_event_loop()
    except Exception as e_fatal:
        logger.critical(f"Fatal error during application lifecycle: {e_fatal}", exc_info=True)
        # Attempt to show a final error message if GUI is still possible
        try:
            app = QApplication.instance() or QApplication([])
            if app:
                QMessageBox.critical(None, "Application Crashed",
                                     f"A fatal error occurred: {e_fatal}\n\n"
                                     "Please check logs for details. The application will now exit.")
        except Exception:
            pass  # Ignore if GUI can't be shown
    finally:
        # Ensure final shutdown call if not already handled by signals or loop exit
        if not runner._shutdown_requested:
            await runner.shutdown_application_async(is_error=(exit_code != 0))
    return exit_code


if __name__ == "__main__":
    # Ensure a QCoreApplication exists if running certain Qt things before QApplication fully starts
    # However, QApplication should be the main one for GUI apps.
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) # Good for modern displays
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # For Windows, ProactorEventLoop can sometimes be more compatible with Qt/asyncio
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    final_exit_code = 1  # Default to error
    try:
        # asyncio.run() is a good way to run the main async function
        final_exit_code = asyncio.run(main_async_wrapper())
    except KeyboardInterrupt:
        logger.info("Application terminated by KeyboardInterrupt in main.")
        final_exit_code = 0
    except Exception as e_top_level:
        logger.critical(f"Top-level unhandled exception: {e_top_level}", exc_info=True)
        # This might indicate an issue before or after the main async wrapper's try/finally
    finally:
        logger.info(f"Application exiting with final code: {final_exit_code}")
        sys.exit(final_exit_code)