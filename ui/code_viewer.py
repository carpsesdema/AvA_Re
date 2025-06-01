# app/ui/code_viewer.py
import logging
import os
from typing import Optional, Dict, Any

from PySide6.QtCore import Signal, Slot  # Added QRegularExpression
from PySide6.QtGui import (
    QFont, QIcon, QSyntaxHighlighter, QTextDocument, QTextCharFormat,
    QColor, QKeySequence, QShortcut, QTextOption, QCloseEvent
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QComboBox, QFileDialog, QMessageBox, QDialog, QApplication,
    QSizePolicy
)

try:
    from utils import constants
    from core.event_bus import EventBus  # For emitting apply changes
    # Pygments for syntax highlighting (optional, QSyntaxHighlighter is primary here)
    import pygments
    from pygments.lexers import get_lexer_by_name, guess_lexer
    # from pygments.formatters.html import HtmlFormatter # Not used directly if QSyntaxHighlighter is main
    from pygments.util import ClassNotFound

    PYGMENTS_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning(
        "CodeViewer: Pygments library not found. Syntax highlighting may be impacted for some language guessing. "
        "Install with: pip install Pygments"
    )
    PYGMENTS_AVAILABLE = False
    QSyntaxHighlighter = type("QSyntaxHighlighter", (object,), {})  # type: ignore
    constants = type("constants", (object,), {})  # type: ignore
    EventBus = type("EventBus", (object,), {})  # type: ignore
    # Define QRegularExpression fallback if PySide6 itself failed partially (unlikely but for safety)
    try:
        from PySide6.QtCore import QRegularExpression
    except ImportError:
        QRegularExpression = type("QRegularExpression", (object,), {})  # type: ignore

logger = logging.getLogger(__name__)


# --- PythonSyntaxHighlighter (Example using QSyntaxHighlighter) ---
class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent: QTextDocument):
        super().__init__(parent)
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#C586C0"))  # VS Code Purple-ish
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            "\\bdef\\b", "\\bclass\\b", "\\bimport\\b", "\\bfrom\\b", "\\breturn\\b",
            "\\bif\\b", "\\belif\\b", "\\belse\\b", "\\bfor\\b", "\\bwhile\\b",
            "\\btry\\b", "\\bexcept\\b", "\\bfinally\\b", "\\bwith\\b", "\\bas\\b",
            "\\bpass\\b", "\\bbreak\\b", "\\bcontinue\\b", "\\byield\\b",
            "\\band\\b", "\\bor\\b", "\\bnot\\b", "\\bin\\b", "\\bis\\b",
            "\\bTrue\\b", "\\bFalse\\b", "\\bNone\\b", "\\blambda\\b",
            "\\basync\\b", "\\bawait\\b", "\\bglobal\\b", "\\bnonlocal\\b", "\\bassert\\b",
            "\\bdel\\b", "\\braise\\b"
        ]
        for word in keywords:
            pattern = QRegularExpression(word)  # CORRECTED
            rule = (pattern, keyword_format)
            self.highlighting_rules.append(rule)

        self_format = QTextCharFormat()
        self_format.setForeground(QColor("#4EC9B0"))  # VS Code Teal for 'self'
        self.highlighting_rules.append((QRegularExpression("\\bself\\b"), self_format))  # CORRECTED

        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))  # VS Code Orange/Brown for strings
        # Standard strings
        self.highlighting_rules.append((QRegularExpression(r"\"[^\"\\]*(\\.[^\"\\]*)*\""), string_format))
        self.highlighting_rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))
        # Multiline strings (triple quotes)
        # Note: QSyntaxHighlighter processes block by block, so robust multiline string highlighting
        # often requires handling block states (self.previousBlockState(), self.setCurrentBlockState()).
        # For simplicity, these regexes try to match within a single block.
        # A more robust solution for multiline strings might need a state machine.
        self.highlighting_rules.append((QRegularExpression(r"'''[^']*(''[^']|'[^']|[^'])*'''"), string_format))
        self.highlighting_rules.append((QRegularExpression(r'"""[^"]*(""[^"]|"[^"]|[^"])*"""'), string_format))

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6A9955"))  # VS Code Green for comments
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((QRegularExpression("#[^\n]*"), comment_format))  # CORRECTED

        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#b5cea8"))  # VS Code light green/number color
        self.highlighting_rules.append((QRegularExpression("\\b[0-9]+\\.?[0-9]*([eE][-+]?[0-9]+)?\\b"),
                                        number_format))  # CORRECTED (added exponent)

        decorator_format = QTextCharFormat()
        decorator_format.setForeground(QColor("#DCDCAA"))  # VS Code Yellow-ish for decorators
        self.highlighting_rules.append((QRegularExpression("@[A-Za-z0-9_.]+"),
                                        decorator_format))  # CORRECTED (allow dots for qualified decorators)

    def highlightBlock(self, text: str):
        for pattern, format_rule in self.highlighting_rules:
            # QRegularExpression.globalMatch is available in Qt6. For PySide6, QRegularExpressionMatchIterator is used.
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format_rule)
        # Reset block state if not handling multi-line constructs like block comments or strings
        self.setCurrentBlockState(0)


class CodeViewerWindow(QDialog):
    """
    A window/dialog for displaying generated or modified code.
    Features syntax highlighting, a file selector (dropdown),
    and options to copy or apply changes.
    """
    apply_change_requested = Signal(str, str, str, str)  # project_id, relative_filepath, new_content, focus_prefix

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Code Viewer / Editor")
        self.setObjectName("CodeViewerWindow")
        self.setMinimumSize(700, 500)
        self.setModal(False)

        self._event_bus = EventBus.get_instance()
        self._displayed_files: Dict[str, Dict[str, Any]] = {}
        self._current_file_key: Optional[str] = None
        self.highlighter: Optional[PythonSyntaxHighlighter] = None  # Store highlighter instance

        self._init_ui()
        self._connect_signals()
        logger.info("CodeViewerWindow initialized.")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        selector_layout = QHBoxLayout()
        self.file_selector_label = QLabel("Viewing:")
        self.file_selector_combo = QComboBox()
        self.file_selector_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        selector_layout.addWidget(self.file_selector_label)
        selector_layout.addWidget(self.file_selector_combo, 1)
        main_layout.addLayout(selector_layout)

        self.code_display_edit = QTextEdit()
        self.code_display_edit.setObjectName("CodeDisplayEdit")
        self.code_display_edit.setReadOnly(False)
        self.code_display_edit.setAcceptRichText(False)
        self.code_display_edit.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        font_family = getattr(constants, 'CHAT_FONT_FAMILY', "Consolas")
        font_size = getattr(constants, 'CHAT_FONT_SIZE', 10)
        self.code_display_edit.setFont(QFont(font_family, font_size))
        self.code_display_edit.setStyleSheet(
            "QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: 1px solid #3E3E3E; }"
        )
        self.highlighter = PythonSyntaxHighlighter(self.code_display_edit.document())

        main_layout.addWidget(self.code_display_edit, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.copy_button = QPushButton("Copy Code")
        self.copy_button.setIcon(self._get_icon("content_copy.svg"))
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.setIcon(self._get_icon("apply.svg"))
        self.save_as_button = QPushButton("Save As...")
        # self.close_button = QPushButton("Close") # QDialog has default ways to close

        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.save_as_button)
        button_layout.addWidget(self.apply_button)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def _get_icon(self, icon_filename: str) -> QIcon:
        assets_path = getattr(constants, "ASSETS_PATH", "./assets")
        icon_path = os.path.join(assets_path, icon_filename)
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        logger.warning(f"Icon not found: {icon_path}")
        return QIcon()

    def _connect_signals(self):
        self.file_selector_combo.currentIndexChanged.connect(self._on_file_selected)
        self.copy_button.clicked.connect(self._on_copy_code)
        self.apply_button.clicked.connect(self._on_apply_changes)
        self.save_as_button.clicked.connect(self._on_save_as)
        save_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Save), self)  # Ctrl+S or Cmd+S
        save_shortcut.activated.connect(self._on_apply_changes)

    @Slot(str, str, str, str)  # Adding project_id and focus_prefix to match DialogService call
    def display_new_file_content(self, filename: str, content: str,
                                 project_id: Optional[str] = None,
                                 focus_prefix: Optional[str] = None):
        logger.info(
            f"CodeViewer: Displaying/updating file '{filename}'. Project ID: {project_id}, Focus Prefix: {focus_prefix}")
        file_key = f"{project_id or 'no_project'}:{filename}"  # Make key more unique

        self._displayed_files[file_key] = {
            "content": content, "original_content": content,
            "project_id": project_id, "focus_prefix": focus_prefix,
            "display_name": filename  # Store original filename for display
        }

        found_in_combo = False
        for i in range(self.file_selector_combo.count()):
            if self.file_selector_combo.itemData(i) == file_key:
                found_in_combo = True
                if self.file_selector_combo.currentIndex() == i:
                    self._load_file_content(file_key)
                else:
                    self.file_selector_combo.setCurrentIndex(i)
                break
        if not found_in_combo:
            self.file_selector_combo.addItem(filename, userData=file_key)  # Display filename, store unique key
            self.file_selector_combo.setCurrentText(filename)

        if not self.isVisible(): self.show()
        self.activateWindow()

    @Slot(int)
    def _on_file_selected(self, index: int):
        if index < 0:
            self.code_display_edit.clear()
            self._current_file_key = None
            return

        file_key = self.file_selector_combo.itemData(index)
        if file_key and file_key in self._displayed_files:
            self._load_file_content(file_key)
        else:
            logger.warning(f"CodeViewer: Selected file key '{file_key}' not found.")
            self.code_display_edit.clear()
            self._current_file_key = None

    def _load_file_content(self, file_key: str):
        file_data = self._displayed_files.get(file_key)
        if file_data:
            self._current_file_key = file_key
            self.code_display_edit.setPlainText(file_data["content"])

            # Re-initialize highlighter for the new document content if needed
            # This is important if the document object changes or if you switch languages
            if self.highlighter:
                self.highlighter.setDocument(self.code_display_edit.document())
            else:  # Should not happen if initialized correctly
                self.highlighter = PythonSyntaxHighlighter(self.code_display_edit.document())

            # The highlighter will re-highlight automatically on text change or if rehighlight() is called.
            # Forcing a rehighlight can be useful if styles change or highlighter is re-assigned.
            if self.highlighter: self.highlighter.rehighlight()

            self.apply_button.setEnabled(True)
            self.copy_button.setEnabled(True)
            self.save_as_button.setEnabled(True)
            self.setWindowTitle(f"Code Viewer - {file_data.get('display_name', os.path.basename(file_key))}")
        else:
            self.code_display_edit.clear()
            self.apply_button.setEnabled(False)
            self.copy_button.setEnabled(False)
            self.save_as_button.setEnabled(False)
            self.setWindowTitle("Code Viewer")

    @Slot()
    def _on_copy_code(self):
        if self._current_file_key and self._current_file_key in self._displayed_files:
            QApplication.clipboard().setText(self.code_display_edit.toPlainText())
            self._event_bus.uiTextCopied.emit("Code copied to clipboard!", "#98c379")
        else:
            self._event_bus.uiTextCopied.emit("No code to copy.", "#e5c07b")

    @Slot()
    def _on_apply_changes(self):
        if not self._current_file_key or self._current_file_key not in self._displayed_files:
            QMessageBox.warning(self, "No File", "No file selected or content available to apply.")
            return

        file_data = self._displayed_files[self._current_file_key]
        current_content_in_editor = self.code_display_edit.toPlainText()

        project_id = file_data.get("project_id")
        # The 'display_name' is the relative path within its context for saving
        relative_filepath = file_data.get("display_name", self._current_file_key.split(":")[-1])  # Fallback
        focus_prefix = file_data.get("focus_prefix")

        if project_id is None or focus_prefix is None:
            logger.warning(
                f"Apply Changes: Missing context for file '{relative_filepath}'. Cannot determine save location.")
            self._on_save_as()
            return  # Offer Save As if context incomplete

        logger.info(
            f"Requesting apply/save: Proj='{project_id}', RelPath='{relative_filepath}', FocusPrefix='{focus_prefix}'")
        self.apply_change_requested.emit(project_id, relative_filepath, current_content_in_editor, focus_prefix)
        file_data["content"] = current_content_in_editor  # Optimistically update local cache
        self._event_bus.uiStatusUpdateGlobal.emit(f"Changes for '{relative_filepath}' sent for application.", "#61dafb",
                                                  True, 3000)

    @Slot()
    def _on_save_as(self):
        if not self._current_file_key or self._current_file_key not in self._displayed_files:
            QMessageBox.warning(self, "No Content", "No code content to save.")
            return

        current_content = self.code_display_edit.toPlainText()
        file_data = self._displayed_files[self._current_file_key]
        suggested_filename = file_data.get("display_name", os.path.basename(self._current_file_key.split(":")[-1]))

        default_dir = file_data.get("focus_prefix") or os.path.expanduser("~")
        if not os.path.isdir(default_dir): default_dir = os.path.expanduser("~")

        filePath, _ = QFileDialog.getSaveFileName(
            self, "Save Code As", os.path.join(default_dir, suggested_filename),
            "Python Files (*.py);;Text Files (*.txt);;All Files (*)"
        )
        if filePath:
            try:
                with open(filePath, 'w', encoding='utf-8') as f:
                    f.write(current_content)
                logger.info(f"Code saved to: {filePath}")
                self._event_bus.uiStatusUpdateGlobal.emit(f"Code saved to {os.path.basename(filePath)}", "#98c379",
                                                          True, 3000)
            except Exception as e:
                logger.error(f"Error saving file to {filePath}: {e}", exc_info=True)
                QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")

    def closeEvent(self, event: QCloseEvent):
        logger.debug("CodeViewerWindow close event.")
        super().closeEvent(event)