# app/ui/dialogs/code_viewer_window.py
import logging
import os
import re
from typing import Optional, Dict, List, Tuple, Any  # Added List, Tuple

from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QFont, QIcon, QSyntaxHighlighter, QTextCharFormat, QTextDocument, QColor, QTextOption, \
    QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTextEdit,
    QPushButton, QHBoxLayout, QComboBox, QLabel,
    QSizePolicy, QFileDialog, QApplication, QMessageBox
)

try:
    from utils import constants
    # EventBus for emitting apply changes or other actions
    from core.event_bus import EventBus  # If CodeViewer directly emits to EventBus
    # Pygments for syntax highlighting
    import pygments
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters.html import HtmlFormatter
    from pygments.util import ClassNotFound

    PYGMENTS_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning(
        "CodeViewerWindow: Pygments not found, syntax highlighting will be basic or disabled. "
        "Install with: pip install Pygments"
    )
    PYGMENTS_AVAILABLE = False
    # Fallback types if needed
    constants = type("constants", (object,),
                     {"CHAT_FONT_FAMILY": "Monospace", "CHAT_FONT_SIZE": 10, "ASSETS_PATH": "./assets"})  # type: ignore
    EventBus = type("EventBus", (object,), {})  # type: ignore
    # Define dummy pygments items if not available, for type hinting and preventing NameError
    pygments = None  # type: ignore
    get_lexer_by_name = lambda x, **kwargs: None  # type: ignore
    guess_lexer = lambda x, **kwargs: None  # type: ignore
    HtmlFormatter = type("HtmlFormatter", (object,), {})  # type: ignore
    ClassNotFound = type("ClassNotFound", (Exception,), {})  # type: ignore

logger = logging.getLogger(__name__)


# Basic Python Syntax Highlighter (Fallback if Pygments is not used for direct QTextEdit highlighting)
# This is a very rudimentary highlighter. Pygments via HTML is generally better for QTextEdit.
class BasicPythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent: QTextDocument):
        super().__init__(parent)
        self.highlighting_rules: List[Tuple[str, QTextCharFormat]] = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(Qt.GlobalColor.blue)
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = ["def", "class", "import", "from", "return", "if", "else", "elif",
                    "for", "while", "try", "except", "finally", "with", "as", "in",
                    "is", "and", "or", "not", "lambda", "yield", "async", "await", "pass",
                    "break", "continue", "global", "nonlocal", "assert", "del", "raise"]
        for word in keywords:
            pattern = rf"\b{word}\b"
            self.highlighting_rules.append((pattern, keyword_format))

        comment_format = QTextCharFormat()
        comment_format.setForeground(Qt.GlobalColor.darkGreen)
        self.highlighting_rules.append((r"#[^\n]*", comment_format))

        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#BA2121"))  # Dark red for strings
        self.highlighting_rules.append((r"\"[^\"]*\"", string_format))
        self.highlighting_rules.append((r"'[^']*'", string_format))

    def highlightBlock(self, text: str):
        for pattern, format_rule in self.highlighting_rules:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), format_rule)
        self.setCurrentBlockState(0)


class CodeViewerWindow(QMainWindow):  # Using QMainWindow for potential toolbars/statusbar
    """
    A window for displaying code snippets or full files, with syntax highlighting,
    copy functionality, and potentially saving or applying changes.
    """
    # Emitted when user wants to apply changes made in the viewer (if editing is enabled)
    # project_id, relative_filepath, new_content, original_focus_prefix (if any)
    apply_change_requested = Signal(str, str, str, str)

    # Signals for multi-project IDE integration (if this viewer becomes more advanced)
    projectFilesSaved = Signal(str, str, str)  # project_id, file_path, content
    focusSetOnFiles = Signal(str, list)  # project_id, file_paths

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Code Viewer")
        self.setObjectName("CodeViewerWindow")
        self.setMinimumSize(700, 500)

        self._event_bus = EventBus.get_instance()  # type: ignore
        self._current_files: Dict[str, Dict[
            str, Any]] = {}  # tab_id (filename) -> {"content": str, "project_id": str, "language": str, "is_ai_mod": bool, "original_content": str | None}
        self._active_tab_id: Optional[str] = None

        self._init_ui()
        self._connect_signals()

        logger.info("CodeViewerWindow initialized.")

    def _init_ui(self):
        """Initializes the UI components."""
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # --- Top Control Bar ---
        control_bar_layout = QHBoxLayout()
        self.file_tabs_combo = QComboBox()  # To switch between displayed files/tabs
        self.file_tabs_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.file_tabs_combo.setToolTip("Select file to view")

        self.language_label = QLabel("Language: Auto")  # Display detected/set language
        self.language_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.copy_button = QPushButton("Copy Code")
        self.copy_button.setIcon(self._get_icon("content_copy.svg"))
        self.copy_button.setToolTip("Copy all code in the current view")

        self.save_as_button = QPushButton("Save As...")
        self.save_as_button.setIcon(
            self._get_icon("save.svg", fallback_qta="fa5s.save"))  # Assuming you might add save.svg
        self.save_as_button.setToolTip("Save the current code to a new file")

        # Apply Changes Button (conditionally enabled)
        self.apply_changes_button = QPushButton("Apply Changes")
        self.apply_changes_button.setIcon(self._get_icon("apply.svg", fallback_qta="fa5s.check-circle"))
        self.apply_changes_button.setToolTip("Apply these changes to the original file/project context")
        self.apply_changes_button.setVisible(False)  # Initially hidden, shown if applicable

        control_bar_layout.addWidget(QLabel("File:"))
        control_bar_layout.addWidget(self.file_tabs_combo, 1)
        control_bar_layout.addSpacing(10)
        control_bar_layout.addWidget(self.language_label)
        control_bar_layout.addStretch(1)
        control_bar_layout.addWidget(self.copy_button)
        control_bar_layout.addWidget(self.save_as_button)
        control_bar_layout.addWidget(self.apply_changes_button)
        main_layout.addLayout(control_bar_layout)

        # --- Code Display Area ---
        self.code_display_edit = QTextEdit()
        self.code_display_edit.setObjectName("CodeDisplayAreaTextEdit")
        self.code_display_edit.setReadOnly(False)  # Allow editing for "Apply Changes"
        self.code_display_edit.setAcceptRichText(True)  # Needed for HTML-based syntax highlighting
        self.code_display_edit.setWordWrapMode(QTextOption.WrapMode.NoWrap)  # Code usually doesn't wrap
        font_family = getattr(constants, 'CHAT_FONT_FAMILY', "Monospace")
        font_size = getattr(constants, 'CHAT_FONT_SIZE', 10)
        self.code_display_edit.setFont(QFont(font_family, font_size))
        # Basic dark theme for text edit if no QSS is applied
        self.code_display_edit.setStyleSheet(
            "background-color: #161B22; color: #C9D1D9; border: 1px solid #30363D; font-family: Consolas, Courier New, monospace;")

        main_layout.addWidget(self.code_display_edit, 1)

        # Attempt to set up basic Python highlighter if Pygments fails or is not used for direct highlighting
        # Note: Using HTML formatter with Pygments is generally preferred for QTextEdit.
        # self._highlighter = BasicPythonHighlighter(self.code_display_edit.document())

    def _get_icon(self, icon_filename: str, fallback_qta: Optional[str] = None) -> QIcon:
        assets_path = getattr(constants, "ASSETS_PATH", "./assets")
        icon_path = os.path.join(assets_path, icon_filename)
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        elif fallback_qta and PYGMENTS_AVAILABLE:  # Pygments check is wrong here, should be QTAWESOME_AVAILABLE
            try:
                if qta: return qta.icon(fallback_qta, color="#C9D1D9")  # type: ignore
            except:
                pass
        return QIcon()

    def _connect_signals(self):
        self.file_tabs_combo.currentIndexChanged.connect(self._on_tab_selected)
        self.copy_button.clicked.connect(self._copy_code_to_clipboard)
        self.save_as_button.clicked.connect(self._save_code_as)
        self.apply_changes_button.clicked.connect(self._request_apply_changes)

    def add_generated_file(self, filename: str, content: str, project_id: Optional[str] = None,
                           language: Optional[str] = None, is_ai_modification: bool = True,
                           original_content: Optional[str] = None,
                           focus_prefix_for_apply: Optional[str] = None):  # New param
        """
        Adds a new file (or updates an existing one) to be displayed in the code viewer.
        Each file is treated as a 'tab'.
        """
        tab_id = filename  # Use filename as a simple tab identifier

        if not language:
            language = self._guess_language_from_filename(filename)

        self._current_files[tab_id] = {
            "content": content,
            "project_id": project_id,
            "language": language,
            "is_ai_modification": is_ai_modification,
            "original_content": original_content,
            "focus_prefix_for_apply": focus_prefix_for_apply,  # Store for apply action
            "relative_filepath": filename  # Assuming filename can be used as relative path
        }

        # Update combo box
        found_item = False
        for i in range(self.file_tabs_combo.count()):
            if self.file_tabs_combo.itemData(i) == tab_id:
                self.file_tabs_combo.setCurrentIndex(i)
                found_item = True
                break
        if not found_item:
            self.file_tabs_combo.addItem(os.path.basename(filename), userData=tab_id)
            self.file_tabs_combo.setCurrentIndex(self.file_tabs_combo.count() - 1)

        # _on_tab_selected will be called, which calls _display_code
        # If it was already selected, force display update:
        if self._active_tab_id == tab_id:
            self._display_code(tab_id)

    @Slot(int)
    def _on_tab_selected(self, index: int):
        if index < 0:
            self._clear_display()
            return
        tab_id = self.file_tabs_combo.itemData(index)
        if tab_id and tab_id in self._current_files:
            self._active_tab_id = tab_id
            self._display_code(tab_id)
        else:
            self._clear_display()

    def _display_code(self, tab_id: str):
        if tab_id not in self._current_files:
            self._clear_display()
            return

        file_info = self._current_files[tab_id]
        content = file_info["content"]
        language = file_info["language"]

        self.language_label.setText(f"Language: {language.title()}")

        if PYGMENTS_AVAILABLE and pygments:
            try:
                lexer = get_lexer_by_name(language) if language else guess_lexer(content)
                # Using a dark theme like 'material' or 'monokai'
                # Ensure CSS is minimal and compatible with QTextEdit's HTML subset
                formatter = HtmlFormatter(style='material', noclasses=True, wrapcode=True)
                html_code = pygments.highlight(content, lexer, formatter)
                # Wrap in basic HTML structure for QTextEdit
                full_html = f"""
                <html><head>
                <style>
                    body {{ background-color: #161B22; color: #C9D1D9; font-family: Consolas, 'Courier New', monospace; font-size: {self.code_display_edit.font().pointSize()}pt; }}
                    pre {{ margin: 0; white-space: pre-wrap; word-wrap: break-word; }}
                </style>
                </head><body><pre>{html_code}</pre></body></html>
                """
                self.code_display_edit.setHtml(full_html)
            except ClassNotFound:
                logger.warning(f"Pygments lexer not found for language '{language}'. Displaying plain text.")
                self.code_display_edit.setPlainText(content)
            except Exception as e_pyg:
                logger.error(f"Error during Pygments highlighting: {e_pyg}")
                self.code_display_edit.setPlainText(content)  # Fallback
        else:
            self.code_display_edit.setPlainText(content)  # Fallback to plain text

        # Show/Hide Apply Changes button
        can_apply = file_info.get("project_id") and file_info.get("relative_filepath")
        self.apply_changes_button.setVisible(bool(can_apply))

    def _clear_display(self):
        self.code_display_edit.clear()
        self.language_label.setText("Language: N/A")
        self._active_tab_id = None
        self.apply_changes_button.setVisible(False)

    @Slot()
    def _copy_code_to_clipboard(self):
        if self._active_tab_id and self._active_tab_id in self._current_files:
            # Get current text from editor, in case user edited it
            current_code_in_editor = self.code_display_edit.toPlainText()
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(current_code_in_editor)
                self._event_bus.uiTextCopied.emit("Code copied to clipboard!", "#98c379")  # type: ignore
                logger.info(f"Copied code from tab '{self._active_tab_id}' to clipboard.")
        else:
            self._event_bus.uiTextCopied.emit("No code selected to copy.", "#e5c07b")  # type: ignore

    @Slot()
    def _save_code_as(self):
        if self._active_tab_id and self._active_tab_id in self._current_files:
            current_code_in_editor = self.code_display_edit.toPlainText()
            default_filename = os.path.basename(self._active_tab_id)  # Use tab_id (original filename) as default

            filePath, _ = QFileDialog.getSaveFileName(self, "Save Code As", default_filename,
                                                      "Python Files (*.py);;Text Files (*.txt);;All Files (*)")
            if filePath:
                try:
                    with open(filePath, 'w', encoding='utf-8') as f:
                        f.write(current_code_in_editor)
                    logger.info(f"Code from tab '{self._active_tab_id}' saved to: {filePath}")
                    self._event_bus.uiStatusUpdateGlobal.emit(f"Code saved to {os.path.basename(filePath)}", "#4ade80",
                                                              True, 3000)  # type: ignore
                except Exception as e:
                    logger.error(f"Error saving code to file '{filePath}': {e}", exc_info=True)
                    QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")
        else:
            QMessageBox.information(self, "No Code", "No code is currently displayed to save.")

    @Slot()
    def _request_apply_changes(self):
        if self._active_tab_id and self._active_tab_id in self._current_files:
            file_info = self._current_files[self._active_tab_id]
            project_id = file_info.get("project_id")
            relative_filepath = file_info.get("relative_filepath")  # This should be the original filename/path
            focus_prefix = file_info.get("focus_prefix_for_apply")  # The original base directory for this file

            if project_id and relative_filepath:
                current_content_in_editor = self.code_display_edit.toPlainText()
                logger.info(
                    f"Requesting to apply changes for P:{project_id}, File:{relative_filepath}, FocusPrefix: {focus_prefix}")
                # Emit signal for ApplicationOrchestrator to handle the file system write
                self.apply_change_requested.emit(project_id, relative_filepath, current_content_in_editor,
                                                 focus_prefix or "")  # type: ignore
                # DialogService might listen and then emit to EventBus, or this can emit directly.
                # Let's assume this emits directly to EventBus for now if DialogService doesn't mediate this specific signal.
                # self._event_bus.applyFileChangeRequested.emit(project_id, relative_filepath, current_content_in_editor, focus_prefix or "")
            else:
                logger.warning(
                    f"Cannot apply changes for tab '{self._active_tab_id}': missing project_id or relative_filepath.")
                QMessageBox.warning(self, "Apply Error",
                                    "Cannot apply changes: Missing project context or original filepath.")
        else:
            logger.warning("Apply changes clicked but no active file or file info.")

    def _guess_language_from_filename(self, filename: str) -> str:
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        # Simple mapping, can be expanded
        if ext == ".py": return "python"
        if ext == ".js": return "javascript"
        if ext == ".ts": return "typescript"
        if ext == ".java": return "java"
        if ext == ".cs": return "csharp"
        if ext == ".html": return "html"
        if ext == ".css": return "css"
        if ext == ".json": return "json"
        if ext == ".yaml" or ext == ".yml": return "yaml"
        if ext == ".xml": return "xml"
        if ext == ".sql": return "sql"
        if ext == ".md": return "markdown"
        return "plaintext"  # Default

    def closeEvent(self, event: QCloseEvent):
        logger.debug("CodeViewerWindow closeEvent triggered.")
        # Clean up any resources if necessary (e.g., disconnect signals if not parented)
        super().closeEvent(event)