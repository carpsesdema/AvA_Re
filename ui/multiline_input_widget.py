# app/ui/multiline_input_widget.py
import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot, QTimer, QSize  # Added QSize
from PySide6.QtGui import QFont, QFontMetrics, QTextOption, QKeyEvent, QPalette, \
    QColor  # Added QPalette for placeholder text
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit

try:
    from utils import constants
except ImportError as e_miw:
    logging.getLogger(__name__).critical(f"Critical import error in MultilineInputWidget: {e_miw}", exc_info=True)
    # Fallback constants
    constants = type("constants", (object,), {"CHAT_FONT_FAMILY": "Arial", "CHAT_FONT_SIZE": 10})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class MultilineInputWidget(QTextEdit):  # Inherit directly from QTextEdit
    """
    A custom QTextEdit widget designed for multiline chat input.
    It handles dynamic height adjustment based on content,
    and emits a signal when the user intends to send a message (e.g., by pressing Enter).
    Shift+Enter inserts a newline.
    """
    sendMessageRequested = Signal()  # Emitted when Enter is pressed (without Shift)
    # textChanged signal is inherited from QTextEdit

    MIN_LINES = 1
    MAX_LINES = 6  # Max number of text lines before scrolling appears
    LINE_PADDING_VERTICAL = 8  # Approximate vertical padding around text within the widget

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("MultilineInputTextEdit")  # For styling

        self._min_height_calculated: int = 30  # Calculated minimum height
        self._max_height_calculated: int = 150  # Calculated maximum height
        self._placeholder_text = "Type a message (Shift+Enter for newline)..."
        self._is_placeholder_visible = False  # To track placeholder state

        self._init_settings()
        self._calculate_and_set_height_limits()
        self._update_widget_height()  # Initial height
        self.set_placeholder()  # Set initial placeholder

    def _init_settings(self):
        """Initializes widget-specific settings."""
        self.setAcceptRichText(False)  # Plain text input
        self.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # Show scrollbar when content exceeds max height

        try:
            font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), getattr(constants, 'CHAT_FONT_SIZE', 10))
            self.setFont(font)
        except Exception as e_font:
            logger.error(f"Error setting font for MultilineInputWidget: {e_font}")

        # Connect textChanged signal to update height and placeholder
        self.textChanged.connect(self._on_text_changed_actions)

    def _calculate_and_set_height_limits(self):
        """Calculates min/max height based on font metrics and line counts."""
        fm = self.fontMetrics()  # QFontMetrics from the current font
        line_h = fm.height()

        # Consider document margins and frame width for accurate height calculation
        doc_margin_vertical = self.document().documentMargin() * 2  # Top and bottom margin
        frame_w_vertical = self.frameWidth() * 2  # Top and bottom frame

        # Base content height for min/max lines
        min_content_h = line_h * self.MIN_LINES
        max_content_h = line_h * self.MAX_LINES

        # Total height includes content, internal margins, and frame
        self._min_height_calculated = int(
            min_content_h + doc_margin_vertical + frame_w_vertical + self.LINE_PADDING_VERTICAL)
        self._max_height_calculated = int(
            max_content_h + doc_margin_vertical + frame_w_vertical + self.LINE_PADDING_VERTICAL)

        self.setMinimumHeight(self._min_height_calculated)
        self.setMaximumHeight(self._max_height_calculated)  # This makes the widget scroll past max lines
        logger.debug(
            f"MultilineInput calculated heights: Min={self._min_height_calculated}, Max={self._max_height_calculated}")

    @Slot()
    def _on_text_changed_actions(self):
        """Actions to perform when text content changes."""
        self._update_widget_height()
        self.update_placeholder_visibility()
        # The inherited textChanged signal is already emitted by QTextEdit itself.

    def _update_widget_height(self):
        """Dynamically adjusts the widget's height based on its content."""
        if not self.document(): return

        # Ensure document width is set for correct height calculation of wrapped text
        # Use viewport width if available, otherwise widget width.
        viewport_w = self.viewport().width() if self.viewport() else self.width()
        doc_width = viewport_w if viewport_w > 0 else self.width() - (self.frameWidth() * 2) - int(
            self.document().documentMargin() * 2)
        if doc_width <= 0: doc_width = 100  # Fallback minimum

        self.document().setTextWidth(doc_width)
        content_height = self.document().size().height()

        doc_margin_vertical = self.document().documentMargin() * 2
        frame_w_vertical = self.frameWidth() * 2

        target_widget_height = int(content_height + doc_margin_vertical + frame_w_vertical + self.LINE_PADDING_VERTICAL)

        # Clamp between calculated min and max heights
        clamped_height = max(self._min_height_calculated, min(target_widget_height, self._max_height_calculated))

        if self.height() != clamped_height:
            self.setFixedHeight(clamped_height)
            # logger.debug(f"MultilineInput height updated to: {clamped_height} (Content: {content_height})")

    def keyPressEvent(self, event: QKeyEvent):
        """Handles key presses, specifically Enter and Shift+Enter."""
        key = event.key()
        modifiers = event.modifiers()

        is_enter_key = key in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        is_shift_modifier = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        if is_enter_key and not is_shift_modifier:
            self.sendMessageRequested.emit()  # Signal to send the message
            event.accept()  # Consume the event, don't insert newline
        elif is_enter_key and is_shift_modifier:
            super().keyPressEvent(event)  # Default behavior: insert newline
        else:
            super().keyPressEvent(event)  # Default behavior for other keys

    def get_text(self) -> str:
        """Returns the plain text content, stripping leading/trailing whitespace."""
        return self.toPlainText().strip()

    def clear_text(self):
        """Clears the text from the input widget."""
        super().clear()  # Use QTextEdit's clear method
        # textChanged signal will be emitted by clear(), triggering height update and placeholder.

    def set_focus(self):  # Renamed for consistency
        """Sets keyboard focus to this widget."""
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_enabled(self, enabled: bool):  # Renamed for consistency
        """Sets the enabled state of the input widget."""
        self.setEnabled(enabled)
        self.update_placeholder_visibility()  # Placeholder might change with enabled state

    # --- Placeholder Text Handling ---
    def set_placeholder(self):
        """Sets the placeholder text if the input is empty and not focused."""
        if not self.toPlainText() and not self.hasFocus():
            self.setText(self._placeholder_text)
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.Text, QColor(Qt.GlobalColor.gray))  # Placeholder color
            self.setPalette(palette)
            self._is_placeholder_visible = True
        elif self._is_placeholder_visible and self.toPlainText() == self._placeholder_text and not self.hasFocus():
            # Already showing placeholder, do nothing
            pass
        else:  # Has text or focus, ensure normal color
            self.clear_placeholder()

    def clear_placeholder(self):
        """Clears the placeholder text and resets text color."""
        if self._is_placeholder_visible and self.toPlainText() == self._placeholder_text:
            self.clear()  # Clears the placeholder text

        # Reset to default text color
        palette = self.palette()
        # Assuming you have a way to get the default text color, e.g., from style or constants
        # For now, using a common default. This should ideally come from the theme.
        default_text_color = QColor(getattr(constants, "INPUT_TEXT_COLOR_HEX", "#E0E0E0"))
        palette.setColor(QPalette.ColorRole.Text, default_text_color)
        self.setPalette(palette)
        self._is_placeholder_visible = False

    def update_placeholder_visibility(self):
        """Shows or hides placeholder based on content and focus."""
        if not self.toPlainText() and not self.hasFocus() and self.isEnabled():
            self.set_placeholder()
        elif self._is_placeholder_visible:  # If placeholder is visible but shouldn't be
            self.clear_placeholder()

    def focusInEvent(self, event: QKeyEvent):  # Changed to QFocusEvent, but QKeyEvent might be from old code
        """Overrides focusInEvent to handle placeholder text."""
        self.clear_placeholder()
        super().focusInEvent(event)  # Call base class implementation

    def focusOutEvent(self, event: QKeyEvent):  # Changed to QFocusEvent
        """Overrides focusOutEvent to handle placeholder text."""
        self.update_placeholder_visibility()
        super().focusOutEvent(event)  # Call base class implementation

    # Ensure widget resizes correctly when shown or its font changes
    def showEvent(self, event: QKeyEvent):  # Changed to QShowEvent
        super().showEvent(event)
        QTimer.singleShot(0, self._update_widget_height)  # Recalculate height after it's shown

    def resizeEvent(self, event: QKeyEvent):  # Changed to QResizeEvent
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_widget_height)  # Recalculate height on resize

    # Placeholder for image data handling, if this widget were to support it
    # def get_attached_image_data(self) -> List[Dict[str, Any]]:
    #     return []