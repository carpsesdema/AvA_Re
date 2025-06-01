# app/ui/loading_overlay.py
import logging
import os
from typing import Optional

from PySide6.QtCore import Qt  # Added QSize
from PySide6.QtGui import QMovie, QFont, QPaintEvent, QPainter, QColor  # Added QPaintEvent, QPainter, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel  # Removed QHBoxLayout as it's not used directly

try:
    from utils import constants
except ImportError:
    # Fallback constants if utils.constants is not available
    class constants_fallback:
        ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assets")  # Adjust path if necessary
        CHAT_FONT_FAMILY = "Segoe UI"
        CHAT_FONT_SIZE = 10
        LOADING_GIF_FILENAME = "loading_orb.gif"  # Default name from previous context


    constants = constants_fallback  # type: ignore
    logging.getLogger(__name__).warning("LoadingOverlay: Could not import utils.constants, using fallback values.")

logger = logging.getLogger(__name__)


class LoadingOverlay(QWidget):
    """
    A semi-transparent overlay widget with an animated GIF and a message
    to indicate busy states to the user. It covers its parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LoadingOverlay")  # For styling via QSS

        # Overlay should float above other widgets and not interfere with layout if hidden
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.ToolTip)  # ToolTip makes it float
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # Allows for custom painter background
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)  # Block mouse events when visible

        self._movie: Optional[QMovie] = None
        self._gif_label: Optional[QLabel] = None
        self._message_label: Optional[QLabel] = None

        self._background_color = QColor(13, 17, 23, 190)  # Dark, semi-transparent (rgba)

        self._init_ui_elements()
        self._setup_layout()
        self._load_animation_asset()

        self.hide()  # Start hidden
        logger.info("LoadingOverlay initialized.")

    def _init_ui_elements(self):
        """Initializes the UI labels for GIF and message."""
        self._gif_label = QLabel(self)
        self._gif_label.setObjectName("LoadingOverlayGifLabel")
        self._gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gif_label.setFixedSize(70, 70)  # Adjusted size for GIF

        self._message_label = QLabel("Loading...", self)
        self._message_label.setObjectName("LoadingOverlayMessageLabel")
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font_size = getattr(constants, 'CHAT_FONT_SIZE', 10)
        message_font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), font_size + 3, QFont.Weight.Bold)
        self._message_label.setFont(message_font)
        self._message_label.setStyleSheet("color: #E0E0E0; padding: 10px;")  # Light text color

    def _setup_layout(self):
        """Sets up the layout to center the GIF and message."""
        layout = QVBoxLayout(self)
        layout.addStretch(1)  # Push content to center vertically
        if self._gif_label: layout.addWidget(self._gif_label, 0, Qt.AlignmentFlag.AlignCenter)
        if self._message_label: layout.addWidget(self._message_label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        self.setLayout(layout)

    def _load_animation_asset(self):
        """Loads the animated GIF from the assets folder."""
        # Use the filename from constants if available
        gif_filename = getattr(constants, 'LOADING_GIF_FILENAME', "loading_orb.gif")  # Default if not in constants
        assets_dir = getattr(constants, 'ASSETS_PATH', "./assets")  # Default if not in constants
        gif_path = os.path.join(assets_dir, gif_filename)

        if not os.path.exists(gif_path):
            logger.error(f"Loading GIF not found at: {gif_path}")
            if self._gif_label: self._gif_label.setText("⏳")  # Fallback emoji
            return

        try:
            self._movie = QMovie(gif_path)
            if not self._movie.isValid():
                logger.error(f"Invalid GIF file: {gif_path}. Movie error: {self._movie.lastErrorString()}")
                if self._gif_label: self._gif_label.setText("⚙️")
                self._movie = None  # Ensure movie is None if invalid
                return

            if self._gif_label:
                self._gif_label.setMovie(self._movie)
                self._movie.setScaledSize(
                    self._gif_label.sizeHint() if self._gif_label.sizeHint().isValid() else self._gif_label.size())
            logger.info(f"Loading animation loaded successfully from: {gif_path}")

        except Exception as e:
            logger.error(f"Error loading animation GIF '{gif_path}': {e}", exc_info=True)
            if self._gif_label: self._gif_label.setText("⚡")
            self._movie = None

    def paintEvent(self, event: QPaintEvent):
        """Custom paint event to draw the semi-transparent background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self._background_color)
        # No need to call super().paintEvent(event) if we are fully custom painting the background

    def show_loading(self, message: str = "Processing..."):
        """Shows the loading overlay with a specified message."""
        if self._message_label: self._message_label.setText(message)

        parent_widget = self.parentWidget()
        if parent_widget:
            self.resize(parent_widget.size())  # Cover the parent
            self.move(0, 0)  # Align with parent's top-left

        if self._movie and self._movie.isValid():
            self._movie.start()
            logger.debug("Loading animation started.")

        self.show()
        self.raise_()  # Ensure it's on top
        logger.info(f"Loading overlay shown with message: '{message}'")

    def hide_loading(self):
        """Hides the loading overlay."""
        if self._movie and self._movie.isValid():
            self._movie.stop()
            logger.debug("Loading animation stopped.")

        self.hide()
        logger.info("Loading overlay hidden.")

    def update_message(self, message: str):
        """Updates the message on the visible loading overlay."""
        if self._message_label and self.isVisible():
            self._message_label.setText(message)
            logger.debug(f"Loading overlay message updated: '{message}'")

    # Ensure the overlay resizes with its parent if it's visible
    def resizeEvent(self, event: QPaintEvent):  # QResizeEvent is more appropriate
        super().resizeEvent(event)  # type: ignore
        if self.parentWidget() and self.isVisible():
            self.resize(self.parentWidget().size())

    def showEvent(self, event: QPaintEvent):  # QShowEvent
        super().showEvent(event)  # type: ignore
        if self.parentWidget():  # Ensure it covers parent on show
            self.resize(self.parentWidget().size())
            self.move(0, 0)