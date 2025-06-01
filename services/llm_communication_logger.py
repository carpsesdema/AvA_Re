# app/services/llm_communication_logger.py
import logging
import uuid
import html
from datetime import datetime
from typing import Dict, List, Optional, Any
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

try:
    import pygments
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import HtmlFormatter # Using HtmlFormatter
    from pygments.util import ClassNotFound
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    # Fallback definitions for type hinting if pygments is not available
    HtmlFormatter = type("HtmlFormatter", (object,), {}) # type: ignore
    logger.warning("Pygments not available. Code highlighting in LLM Log will be disabled.")


class LlmCommunicationLogger(QObject):
    # Signal for regular log entries to be displayed (e.g., in a QTextBrowser)
    new_log_entry = Signal(str)  # formatted_html_log_entry

    # Signals for dynamic code block streaming
    code_block_stream_started = Signal(str, str)  # block_id, language_hint
    code_block_chunk_received = Signal(str, str)  # block_id, plain_text_chunk
    code_block_stream_finished = Signal(str, str, str)  # block_id, final_plain_text_content, final_highlighted_html_content

    # Senders that should have their messages styled with HTML
    STYLED_CONTENT_SENDERS = {
        "GEMINI_CHAT_DEFAULT RESPONSE", "GPT_CHAT_DEFAULT RESPONSE", "OLLAMA_CHAT_DEFAULT RESPONSE",
        "PACC:SEQ_START", "PACC:MULTI_FILE_COMPLETE", # PlanAndCodeCoordinator messages
        "MTC:SEQ_START", "MTC:MICRO_TASK_COMPLETE", # MicroTaskCoordinator messages
        "RAG_SCAN_GLOBAL", "RAG_UPLOAD",
        "AUTONOMOUS_CODING_REQUEST", "AUTONOMOUS_CODING_ERROR", "AUTONOMOUS_CODING_EXCEPTION",
        "CODE_EXTRACTION", "CODE_EXTRACTION_ERROR",
        "SYSTEM_CODE_STREAM", "CONVERSATION_ORCHESTRATOR"
    }

    # Define styles for different senders for better visual distinction
    HTML_SENDER_STYLES = {
        "GEMINI_CHAT_DEFAULT RESPONSE": {"color": "#4fc3f7", "symbol": "🧠"}, # Light blue
        "GPT_CHAT_DEFAULT RESPONSE": {"color": "#81c784", "symbol": "🤖"},    # Green
        "OLLAMA_CHAT_DEFAULT RESPONSE": {"color": "#ffb74d", "symbol": "🦙"}, # Orange
        "PACC:SEQ_START": {"color": "#e57373", "symbol": "⚙️P"},              # Light red
        "PACC:MULTI_FILE_COMPLETE": {"color": "#aed581", "symbol": "✅P"},    # Light green
        "MTC:SEQ_START": {"color": "#ba68c8", "symbol": "⚙️M"},               # Purple
        "MTC:MICRO_TASK_COMPLETE": {"color": "#ce93d8", "symbol": "✅M"},     # Light Purple
        "RAG_SCAN_GLOBAL": {"color": "#7986cb", "symbol": "🔍G"},             # Indigo
        "RAG_UPLOAD": {"color": "#4db6ac", "symbol": "🔍P"},                  # Teal
        "AUTONOMOUS_CODING_REQUEST": {"color": "#64b5f6", "symbol": "💡"},   # Blue
        "AUTONOMOUS_CODING_ERROR": {"color": "#ef5350", "symbol": "❌A"},      # Red
        "AUTONOMOUS_CODING_EXCEPTION": {"color": "#ff5722", "symbol": "💥A"},  # Deep Orange
        "CODE_EXTRACTION": {"color": "#26a69a", "symbol": "📦"},             # Dark Teal
        "CODE_EXTRACTION_ERROR": {"color": "#ff7043", "symbol": "⚠️E"},       # Orange
        "SYSTEM_CODE_STREAM": {"color": "#9575cd", "symbol": "📜S"},          # Deep Purple
        "CONVERSATION_ORCHESTRATOR": {"color": "#ffee58", "symbol": "🗣️"},  # Yellow
        "USER": {"color": "#E0E0E0", "symbol": "👤"}, # Style for user messages if logged here
        "SYSTEM": {"color": "#B0B0B0", "symbol": "ℹ️"} # General system messages
    }
    # Default style for senders not in HTML_SENDER_STYLES
    DEFAULT_STYLE = {"color": "#B0B0B0", "symbol": "💬"}


    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._log_entries: List[str] = [] # Stores formatted HTML log entries

        # State for managing active code block streaming
        self._active_code_streams: Dict[str, Dict[str, Any]] = {}
        # Structure: {block_id: {'language': str, 'buffer': List[str], 'initial_html_emitted': bool}}

        logger.info("Enhanced LlmCommunicationLogger initialized with code streaming support")

    def log_message(self, sender: str, message: str) -> None:
        """Logs a regular message (non-code streaming) to the LLM communication log."""
        if not sender or not message:
            logger.debug(f"Skipping empty log message from sender: '{sender}'")
            return

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3] # Milliseconds
        escaped_message = html.escape(message)

        style_info = self.HTML_SENDER_STYLES.get(sender.upper(), self.DEFAULT_STYLE)
        color = style_info["color"]
        symbol = style_info["symbol"]

        # Basic HTML structure for each log entry
        # Using <pre> for message content to preserve whitespace and newlines better
        formatted_entry = (
            f'<div style="margin: 3px 0; padding: 5px 10px; border-left: 3px solid {color}; background: rgba(255,255,255,0.03); border-radius: 3px;">'
            f'<span style="color: #777; font-size: 10px; font-family: monospace;">[{timestamp}]</span> '
            f'<strong style="color: {color};">{symbol} {html.escape(sender)}:</strong>'
            f'<pre style="color: #D0D0D0; margin: 4px 0 0 15px; padding:0; white-space: pre-wrap; word-wrap: break-word; font-size: 12px; font-family: Consolas, monospace;">{escaped_message}</pre>'
            f'</div>'
        )

        self._log_entries.append(formatted_entry)
        self.new_log_entry.emit(formatted_entry)

    def start_streaming_code_block(self, language_hint: str = "python") -> str:
        """
        Initiates a new code streaming session.
        Emits code_block_stream_started for the UI to prepare.
        """
        block_id = f"code_stream_{uuid.uuid4().hex[:8]}"
        self._active_code_streams[block_id] = {
            'language': language_hint.lower() if language_hint else "plaintext",
            'buffer': [],
            'initial_html_emitted': False # Track if the placeholder has been emitted
        }
        self.log_message("SYSTEM_CODE_STREAM", f"Starting code block streaming (lang: {language_hint}, id: {block_id})...")
        self.code_block_stream_started.emit(block_id, language_hint.lower() if language_hint else "plaintext")
        logger.info(f"Started code streaming block {block_id} (lang: {language_hint})")
        return block_id

    def stream_code_chunk(self, block_id: str, chunk: str) -> None:
        """
        Adds a chunk of raw code to an active stream.
        Emits code_block_chunk_received for live UI updates.
        """
        if block_id not in self._active_code_streams:
            logger.warning(f"Attempted to stream chunk to unknown block_id: {block_id}")
            return

        self._active_code_streams[block_id]['buffer'].append(chunk)
        self.code_block_chunk_received.emit(block_id, chunk) # Send raw chunk for immediate display
        logger.debug(f"Streamed chunk to block {block_id}: {len(chunk)} chars")

    def finish_streaming_code_block(self, block_id: str) -> None:
        """
        Finalizes an active code stream.
        Generates syntax-highlighted HTML for the complete code block.
        Emits code_block_stream_finished for the UI to replace the streamed content.
        """
        if block_id not in self._active_code_streams:
            logger.warning(f"Attempted to finish unknown block_id: {block_id}")
            return

        stream_info = self._active_code_streams.pop(block_id) # Remove from active streams
        language = stream_info['language']
        complete_code = ''.join(stream_info['buffer'])

        highlighted_html = self._highlight_code_block_html(complete_code, language)

        self.log_message("SYSTEM_CODE_STREAM", f"Code block {block_id} completed ({len(complete_code)} chars).")
        self.code_block_stream_finished.emit(block_id, complete_code, highlighted_html)
        logger.info(f"Finished code streaming block {block_id}")

    def _highlight_code_block_html(self, code: str, language_hint: str) -> str:
        """Generates syntax-highlighted HTML for a code block using Pygments."""
        if not PYGMENTS_AVAILABLE or not code.strip():
            # Fallback: simple preformatted, HTML-escaped text
            return f'<pre style="white-space: pre-wrap; word-wrap: break-word; color: #D0D0D0; background-color: #1E1E1E; padding: 8px; border-radius: 4px;">{html.escape(code)}</pre>'

        try:
            lexer = None
            if language_hint:
                try:
                    lexer = get_lexer_by_name(language_hint, stripall=True)
                except ClassNotFound:
                    logger.debug(f"Lexer for '{language_hint}' not found, trying to guess.")
            if not lexer: # If hint was empty or lexer not found by hint
                try:
                    lexer = guess_lexer(code, stripall=True)
                    logger.debug(f"Guessed lexer: {lexer.name}")
                except ClassNotFound:
                    logger.warning(f"Could not guess lexer for code block. Using plaintext.")
                    lexer = get_lexer_by_name("text", stripall=True) # Fallback to plaintext

            # Using 'material' style which is dark-friendly.
            # noclasses=True ensures styles are inline, useful for QTextBrowser.
            # wrapcode=True will wrap long lines within the <pre> block.
            # classprefix can be used if you prefer CSS classes over inline styles.
            formatter = HtmlFormatter(
                style='material',
                noclasses=True, # Inline styles are generally better for QTextBrowser
                wrapcode=True,  # Wraps long lines
                # classprefix="pgmnt-", # Example if using CSS classes
                # cssclass="codehilite", # Outer div class if not full=True
                # No prestyles needed if noclasses=True
                # prestyles="padding: 8px; border-radius: 4px; overflow-x: auto; background-color: #1E1E1E; color: #D0D0D0;"
                # Removed full=True to get just the highlighted code content, not a full HTML document.
            )
            highlighted_code = highlight(code, lexer, formatter)
            # The formatter output (without full=True) is typically ready to be embedded.
            # We might want to wrap it in our own <pre> for consistent styling if the formatter doesn't provide one.
            # With noclasses=True, Pygments usually produces <span> tags with inline styles.
            # Let's wrap this in a <pre> tag with our desired overall style.
            return f'<pre style="white-space: pre-wrap; word-wrap: break-word; background-color: #1E1E1E; padding: 8px; border-radius: 4px; overflow-x:auto;">{highlighted_code}</pre>'

        except Exception as e:
            logger.error(f"Code highlighting failed for language '{language_hint}': {e}", exc_info=True)
            # Fallback for safety
            return f'<pre style="white-space: pre-wrap; word-wrap: break-word; color: #D0D0D0; background-color: #1E1E1E; padding: 8px; border-radius: 4px;">{html.escape(code)}</pre>'


    def get_all_logs_html(self) -> str:
        """Returns all logged entries concatenated as a single HTML string."""
        return "".join(self._log_entries)

    def clear_logs(self) -> None:
        """Clears all logged entries."""
        self._log_entries.clear()
        # Optionally, emit a signal that logs were cleared if UI needs to react
        self.log_message("SYSTEM", "Communication log cleared.")
        logger.info("LlmCommunicationLogger logs cleared.")

    def get_active_streams_count(self) -> int:
        """Returns the number of currently active code streams."""
        return len(self._active_code_streams)

    def cancel_all_streams(self) -> None:
        """Cancels all active code streams. Useful for cleanup or reset."""
        num_cancelled = len(self._active_code_streams)
        if num_cancelled > 0:
            self.log_message("SYSTEM_CODE_STREAM", f"Cancelling {num_cancelled} active code stream(s).")
            for block_id in list(self._active_code_streams.keys()): # list() for safe iteration
                # Emit finish signal with potentially partial content or an error message
                # For simplicity, we'll just mark them as finished without emitting content here,
                # assuming the UI will handle the abrupt end.
                # A more sophisticated version might emit a "cancelled" signal.
                if block_id in self._active_code_streams: # Check again as it might be removed by another thread
                    stream_info = self._active_code_streams.pop(block_id, None)
                    if stream_info:
                        logger.info(f"Cancelled active code stream: {block_id}")
                        # Optionally emit a specific cancellation signal for this block_id
                        # self.code_block_stream_cancelled.emit(block_id)
            logger.info(f"Cancelled {num_cancelled} active code streams.")
        self._active_code_streams.clear()