# app/ui/dialogs/llm_terminal_window.py
import logging
from typing import Optional, Dict, Any

from PySide6.QtCore import Slot, QTimer, QSize, Qt
from PySide6.QtGui import QFont, QTextCursor, QAction, QIcon, QCloseEvent
from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QToolBar, QWidget

try:
    # Assuming LlmCommunicationLogger is in services
    from services.llm_communication_logger import LlmCommunicationLogger
    from utils import constants  # For font constants
    # For icons if qtawesome is used
    import qtawesome as qta

    QTAWESOME_AVAILABLE = True
except ImportError:
    LlmCommunicationLogger = type("LlmCommunicationLogger", (object,), {})  # type: ignore
    constants = type("constants", (object,), {})  # type: ignore
    qta = None  # type: ignore
    QTAWESOME_AVAILABLE = False
    logging.getLogger(__name__).warning("LlmTerminalWindow: qtawesome or other dependencies not found.")

logger = logging.getLogger(__name__)


class LlmTerminalWindow(QDialog):
    """
    A dialog window to display LLM communication logs, including
    dynamically streamed code blocks with syntax highlighting.
    """
    MAX_BLOCK_COUNT = 500  # Max log entries before trimming

    def __init__(self, llm_logger: LlmCommunicationLogger, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("LLM Communication Log")
        self.setObjectName("LlmTerminalWindow")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)

        if not isinstance(llm_logger, LlmCommunicationLogger):  # type: ignore
            logger.error("LlmTerminalWindow initialized without a valid LlmCommunicationLogger.")
            # Potentially show an error in the window itself or disable functionality
            self._llm_logger = None
        else:
            self._llm_logger = llm_logger

        self._text_browser: Optional[QTextBrowser] = None
        self._current_code_block_elements: Dict[str, Any] = {}  # block_id -> QTextCursor or other UI element reference

        self._init_ui()
        self._connect_signals()

        if self._llm_logger:
            self._load_existing_logs()  # Load logs that might have accumulated before window was shown
        else:
            if self._text_browser:
                self._text_browser.append("<p style='color: red;'>Error: LLM Logger not available.</p>")

        logger.info("LlmTerminalWindow initialized.")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Toolbar will have its own margins
        layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar("LLM Log Toolbar")
        toolbar.setIconSize(
            constants.ACTION_BUTTON_SIZE if hasattr(constants, "ACTION_BUTTON_SIZE") else QSize(16, 16))  # type: ignore

        clear_action = QAction(self._get_icon("fa5s.trash-alt", color="#E06C75"), "Clear Log", self)
        clear_action.triggered.connect(self._clear_log_display)
        toolbar.addAction(clear_action)

        self.scroll_to_bottom_action = QAction(self._get_icon("fa5s.arrow-down", color="#61AFEF"), "Auto-scroll", self)
        self.scroll_to_bottom_action.setCheckable(True)
        self.scroll_to_bottom_action.setChecked(True)  # Auto-scroll by default
        self.scroll_to_bottom_action.triggered.connect(self._toggle_autoscroll)
        toolbar.addAction(self.scroll_to_bottom_action)

        toolbar.addSeparator()
        self.wrap_lines_action = QAction(self._get_icon("fa5s.align-left", color="#98C379"), "Wrap Lines", self)
        self.wrap_lines_action.setCheckable(True)
        self.wrap_lines_action.setChecked(True)
        self.wrap_lines_action.triggered.connect(self._toggle_line_wrap)
        toolbar.addAction(self.wrap_lines_action)

        layout.addWidget(toolbar)

        # Text Browser for logs
        self._text_browser = QTextBrowser(self)
        self._text_browser.setObjectName("LlmLogBrowser")
        font_family = getattr(constants, 'CHAT_FONT_FAMILY', "Consolas")
        font_size = getattr(constants, 'CHAT_FONT_SIZE', 10)
        self._text_browser.setFont(QFont(font_family, font_size - 1))  # Slightly smaller for logs
        self._text_browser.setReadOnly(True)
        self._text_browser.setOpenExternalLinks(True)  # For any links in logs
        self._text_browser.setLineWrapMode(
            QTextBrowser.LineWrapMode.WidgetWidth if self.wrap_lines_action.isChecked() else QTextBrowser.LineWrapMode.NoWrap)

        # Basic styling - more can be done via QSS file
        self._text_browser.setStyleSheet("""
            QTextBrowser#LlmLogBrowser {
                background-color: #161B22; /* Dark background */
                color: #C9D1D9; /* Light text */
                border: 1px solid #30363D;
                padding: 5px;
            }
        """)
        layout.addWidget(self._text_browser, 1)
        self.setLayout(layout)

    def _get_icon(self, icon_name: str, color: str = "white") -> QIcon:
        if QTAWESOME_AVAILABLE and qta:
            try:
                return qta.icon(icon_name, color=color)
            except:
                pass
        return QIcon()

    def _connect_signals(self):
        if self._llm_logger:
            self._llm_logger.new_log_entry.connect(self.add_log_entry)
            self._llm_logger.code_block_stream_started.connect(self.handle_code_block_stream_started)
            self._llm_logger.code_block_chunk_received.connect(self.handle_code_block_chunk_received)
            self._llm_logger.code_block_stream_finished.connect(self.handle_code_block_stream_finished)

    def _load_existing_logs(self):
        if self._llm_logger and self._text_browser:
            all_logs_html = self._llm_logger.get_all_logs_html()
            if all_logs_html:
                self._text_browser.setHtml(all_logs_html)  # Set initial content
                self._maybe_scroll_to_bottom()
                # FIXED: Handle missing get_all_logs method gracefully
                try:
                    if hasattr(self._llm_logger, 'get_all_logs'):
                        log_count = len(self._llm_logger.get_all_logs())
                        logger.info(f"Loaded {log_count} existing log entries.")
                    else:
                        logger.info("Loaded existing log entries (count unavailable).")
                except (AttributeError, TypeError):
                    logger.info("Loaded existing log entries (count unavailable).")

    @Slot(str)
    def add_log_entry(self, formatted_html_log_entry: str):
        """Appends a pre-formatted HTML log entry to the text browser."""
        if self._text_browser:
            # Trim old entries if log gets too long
            doc = self._text_browser.document()
            if doc.blockCount() > self.MAX_BLOCK_COUNT + 50:  # Add some buffer
                cursor = QTextCursor(doc)
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                for _ in range(doc.blockCount() - self.MAX_BLOCK_COUNT):
                    cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
                    cursor.removeSelectedText()
                    cursor.deleteChar()  # Remove the newline character of the deleted block
                logger.debug(f"Trimmed LLM log to {self.MAX_BLOCK_COUNT} entries.")

            self._text_browser.append(formatted_html_log_entry)  # append handles HTML
            self._maybe_scroll_to_bottom()

    @Slot(str, str)  # block_id, language_hint
    def handle_code_block_stream_started(self, block_id: str, language_hint: str):
        if not self._text_browser: return

        # Create a placeholder for the code block. We'll use a unique HTML element ID.
        # The placeholder will be updated with chunks and then replaced with highlighted code.
        placeholder_html = (
            f'<div id="{block_id}_container" style="margin: 5px 0; padding: 3px; border-left: 3px solid #777; background-color: #222A33; border-radius: 3px;">'
            f'<div style="font-size:0.8em; color:#888; margin-bottom:3px;">Streaming code ({language_hint})... Block ID: {block_id[:8]}</div>'
            f'<pre id="{block_id}" style="white-space: pre-wrap; word-wrap: break-word; color: #A0A0A0; margin:0; padding:0; font-family: Consolas, monospace; font-size: 11px;"></pre>'
            f'</div>'
        )
        self._text_browser.append(placeholder_html)

        # Find the <pre> element to append chunks to it. This is tricky with QTextBrowser.
        # A more robust way might be to store the document position (cursor)
        # or use block numbers if IDs aren't directly findable in QTextDocument.
        # For now, we'll replace the whole block content later.
        # Storing the cursor position where the <pre> tag content starts.
        cursor = self._text_browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        # Move cursor inside the <pre id="block_id"> tag. This is heuristic.
        # It's hard to get an exact cursor inside a dynamically generated HTML element.
        # A simpler approach is to find the block containing the ID and replace its content.
        self._current_code_block_elements[block_id] = {"cursor_pos_before_pre": cursor.position() - 100}  # Approximate
        # "language": language_hint # Store for final highlighting if needed
        self._maybe_scroll_to_bottom()

    @Slot(str, str)  # block_id, plain_text_chunk
    def handle_code_block_chunk_received(self, block_id: str, plain_text_chunk: str):
        if not self._text_browser or block_id not in self._current_code_block_elements:
            return

        # This is where it gets tricky with QTextBrowser and dynamic HTML updates.
        # Appending directly to an innerHTML of a specific element is not straightforward.
        # Option 1: Rebuild the HTML for the block on each chunk (can be slow).
        # Option 2: Use QTextCursor to insert text at a known position (better).
        # Option 3: Use a placeholder character and replace it, or manage separate blocks.

        # For simplicity and performance, we'll let LlmCommunicationLogger provide the full highlighted block at the end.
        # The LlmTerminalWindow will just show raw chunks being appended to a placeholder.
        # When the "finished" signal comes, it replaces the placeholder with the final HTML.

        # Find the placeholder <pre> tag and append to it.
        # This requires searching the document, which can be slow.
        # A simpler approach for live updates: just append to the end of the document for now,
        # and then the "finished" signal can replace the whole temporary block.

        # Let's assume the logger sends a final highlighted block.
        # Here, we can append the plain text chunk to the *current* end of the text browser
        # if we want to show raw streaming.
        # However, the `LlmCommunicationLogger` is designed to emit a `code_block_stream_finished`
        # with the *final highlighted HTML*. This window should primarily handle that.
        # The `code_block_chunk_received` signal from the logger is more for UIs that want to
        # build up the plain text incrementally themselves.

        # For this terminal, let's just show that chunks are arriving.
        # The final replacement will happen in handle_code_block_stream_finished.
        # We could try to update the content of the <pre id="block_id"> tag if QTextBrowser allowed easy DOM manipulation.
        # Since it doesn't, we'll rely on the finish signal.
        # The LlmCommunicationLogger itself emits a "SYSTEM_CODE_STREAM: Starting code block..."
        # and "SYSTEM_CODE_STREAM: Code block ... completed".
        # This window could also show "Chunk received..." messages.

        # The current design of LlmCommunicationLogger is that it sends the final block.
        # If we want this window to update per chunk, LlmCommunicationLogger needs to send highlighted chunks
        # or this window needs to do highlighting per chunk (inefficient).

        # Let's assume for now that the LlmCommunicationLogger's `code_block_chunk_received` is for
        # other types of displays, and this one waits for `code_block_stream_finished`.
        # We can add a small indicator that chunks are flowing.
        # self.add_log_entry(f"<span style='color:#555;'>Chunk for {block_id[:8]}...</span>")
        pass  # We will update with the full block on finish.

    @Slot(str, str, str)  # block_id, final_plain_text_content (unused here), final_highlighted_html_content
    def handle_code_block_stream_finished(self, block_id: str, final_plain_text_content: str,
                                          final_highlighted_html_content: str):
        if not self._text_browser or block_id not in self._current_code_block_elements:
            logger.warning(f"LlmTerminalWindow: Received finish for unknown or already processed block_id: {block_id}")
            return

        # Replace the placeholder div with the final highlighted content.
        # This is tricky with QTextBrowser. We can't easily replace an element by ID.
        # Workaround: We can remove the "Streaming..." message and append the final block.
        # A more robust way would be to find the block number of the placeholder and replace that block.

        # For now, let's append the final highlighted content as a new entry.
        # The LlmCommunicationLogger already logs "Code block completed".
        # This signal provides the actual content to display.

        # Construct the final display HTML for this block
        block_info = self._current_code_block_elements.get(block_id, {})
        language_hint = block_info.get("language", "code")

        final_block_html = (
            f'<div style="margin: 5px 0; padding: 0; border-left: 3px solid #888; background-color: #1c1c1c; border-radius: 3px;">'  # Container
            f'<div style="font-size:0.8em; color:#999; margin-bottom:3px; padding: 3px 6px; background-color: #2a2a2a; border-top-left-radius:3px; border-top-right-radius:3px;">Code Block ({language_hint}) - ID: {block_id[:8]}</div>'
            f'{final_highlighted_html_content}'  # This should be the <pre>...</pre> block from logger
            f'</div>'
        )
        self.add_log_entry(final_block_html)  # Add it as a new log entry

        self._current_code_block_elements.pop(block_id, None)  # Clean up
        self._maybe_scroll_to_bottom()

    def _maybe_scroll_to_bottom(self):
        if self._text_browser and self.scroll_to_bottom_action and self.scroll_to_bottom_action.isChecked():
            # FIXED: Use lambda to properly call setValue with PySide6
            QTimer.singleShot(0, lambda: self._text_browser.verticalScrollBar().setValue(
                self._text_browser.verticalScrollBar().maximum()))

    @Slot()
    def _clear_log_display(self):
        if self._text_browser:
            self._text_browser.clear()
        if self._llm_logger:  # Also clear the logger's internal buffer if desired
            # self._llm_logger.clear_logs() # This would prevent reloading old logs
            pass
        self.add_log_entry("<p style='color: #888;'>Log cleared.</p>")

    @Slot(bool)
    def _toggle_autoscroll(self, checked: bool):
        logger.debug(f"Autoscroll toggled: {checked}")
        if checked and self._text_browser:
            self._maybe_scroll_to_bottom()

    @Slot(bool)
    def _toggle_line_wrap(self, checked: bool):
        if self._text_browser:
            self._text_browser.setLineWrapMode(
                QTextBrowser.LineWrapMode.WidgetWidth if checked else QTextBrowser.LineWrapMode.NoWrap)
            logger.debug(f"Line wrap toggled: {checked}")

    def closeEvent(self, event: QCloseEvent):
        logger.info("LlmTerminalWindow closing.")
        # Disconnect signals to prevent issues if logger outlives this window
        if self._llm_logger:
            try:
                self._llm_logger.new_log_entry.disconnect(self.add_log_entry)
            except (TypeError, RuntimeError):
                pass
            try:
                self._llm_logger.code_block_stream_started.disconnect(self.handle_code_block_stream_started)
            except (TypeError, RuntimeError):
                pass
            try:
                self._llm_logger.code_block_chunk_received.disconnect(self.handle_code_block_chunk_received)
            except (TypeError, RuntimeError):
                pass
            try:
                self._llm_logger.code_block_stream_finished.disconnect(self.handle_code_block_stream_finished)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)