# app/ui/dialogs/code_viewer_dialog.py
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, List

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import (
    QFont, QSyntaxHighlighter, QTextCharFormat, QTextDocument,
    QColor, QKeySequence, QAction
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTabWidget, QTextEdit,
    QPushButton, QLabel, QMenu, QStatusBar,
    QFileDialog, QMessageBox, QGroupBox, QProgressBar
)

try:
    from utils import constants
    from core.event_bus import EventBus

    # For syntax highlighting
    try:
        import pygments
        from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
        from pygments.formatters.html import HtmlFormatter
        from pygments.util import ClassNotFound

        PYGMENTS_AVAILABLE = True
    except ImportError:
        PYGMENTS_AVAILABLE = False

except ImportError as e:
    logging.getLogger(__name__).error(f"Import error in CodeViewerDialog: {e}")
    constants = type("constants", (object,), {})()
    EventBus = type("EventBus", (object,), {})
    PYGMENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileChangeStatus(Enum):
    """Status of files in the code viewer"""
    UNCHANGED = auto()
    MODIFIED = auto()
    NEW = auto()
    STAGED = auto()


@dataclass
class FileInfo:
    """Information about a file in the code viewer"""
    path: str
    content: str
    original_content: Optional[str] = None
    status: FileChangeStatus = FileChangeStatus.UNCHANGED
    project_id: Optional[str] = None
    is_ai_generated: bool = False
    language: Optional[str] = None


class FileTreeWidget(QTreeWidget):
    """Enhanced file tree widget for project navigation"""

    fileSelected = Signal(str)  # Emit file path when selected
    directoryFocused = Signal(str)  # Emit directory path when focused

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabel("Project Files")
        self.setObjectName("CodeViewerFileTree")
        self.itemClicked.connect(self._on_item_clicked)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def load_project_structure(self, project_path: str):
        """Load project directory structure into tree"""
        self.clear()
        if not os.path.exists(project_path):
            return

        root_item = QTreeWidgetItem(self)
        root_item.setText(0, os.path.basename(project_path))
        root_item.setData(0, Qt.UserRole, project_path)
        root_item.setIcon(0, self.style().standardIcon(self.style().SP_DirIcon))

        self._populate_tree(root_item, project_path)
        self.expandItem(root_item)

    def _populate_tree(self, parent_item: QTreeWidgetItem, directory_path: str):
        """Recursively populate tree with files and directories"""
        try:
            for item in sorted(os.listdir(directory_path)):
                if item.startswith('.'):  # Skip hidden files
                    continue

                item_path = os.path.join(directory_path, item)
                tree_item = QTreeWidgetItem(parent_item)
                tree_item.setText(0, item)
                tree_item.setData(0, Qt.UserRole, item_path)

                if os.path.isdir(item_path):
                    tree_item.setIcon(0, self.style().standardIcon(self.style().SP_DirIcon))
                    self._populate_tree(tree_item, item_path)
                else:
                    tree_item.setIcon(0, self.style().standardIcon(self.style().SP_FileIcon))

        except PermissionError:
            logging.warning(f"Permission denied accessing {directory_path}")

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item selection"""
        file_path = item.data(0, Qt.UserRole)
        if os.path.isfile(file_path):
            self.fileSelected.emit(file_path)
        elif os.path.isdir(file_path):
            self.directoryFocused.emit(file_path)

    def _show_context_menu(self, position):
        """Show context menu for tree items"""
        item = self.itemAt(position)
        if not item:
            return

        menu = QMenu(self)

        file_path = item.data(0, Qt.UserRole)
        if os.path.isfile(file_path):
            menu.addAction("Open", lambda: self.fileSelected.emit(file_path))
            menu.addAction("Open in External Editor", lambda: self._open_external(file_path))
        elif os.path.isdir(file_path):
            menu.addAction("Set as Focus Directory", lambda: self.directoryFocused.emit(file_path))

        menu.exec(self.mapToGlobal(position))

    def _open_external(self, file_path: str):
        """Open file in external editor"""
        try:
            os.startfile(file_path)  # Windows
        except AttributeError:
            os.system(f'open "{file_path}"')  # macOS
        except:
            os.system(f'xdg-open "{file_path}"')  # Linux


class CodeEditorTab(QTextEdit):
    """Enhanced code editor with syntax highlighting and change tracking"""

    contentChanged = Signal(str, str)  # file_path, content
    modificationStateChanged = Signal(str, bool)  # file_path, is_modified

    def __init__(self, file_info: FileInfo, parent=None):
        super().__init__(parent)
        self.file_info = file_info
        self.is_modified = False

        # Set up font and basic styling
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)

        # Set up syntax highlighting if available
        self._setup_syntax_highlighting()

        # Connect text change signal
        self.textChanged.connect(self._on_text_changed)

        # Load content
        self.setPlainText(file_info.content)
        self.is_modified = False

    def _setup_syntax_highlighting(self):
        """Set up syntax highlighting based on file language"""
        if not PYGMENTS_AVAILABLE:
            return

        try:
            if self.file_info.language:
                lexer = get_lexer_by_name(self.file_info.language)
            else:
                lexer = guess_lexer_for_filename(self.file_info.path, self.file_info.content)

            # For now, use basic Python highlighting
            # In a full implementation, you'd create language-specific highlighters
            if 'python' in lexer.name.lower():
                self.highlighter = PythonSyntaxHighlighter(self.document())

        except Exception as e:
            logger.debug(f"Could not set up syntax highlighting: {e}")

    def _on_text_changed(self):
        """Handle text changes"""
        current_content = self.toPlainText()
        was_modified = self.is_modified
        self.is_modified = (current_content != self.file_info.original_content)

        if was_modified != self.is_modified:
            self.modificationStateChanged.emit(self.file_info.path, self.is_modified)

        self.contentChanged.emit(self.file_info.path, current_content)

    def get_content(self) -> str:
        """Get current content"""
        return self.toPlainText()

    def save_content(self):
        """Save current content to file info"""
        self.file_info.content = self.toPlainText()
        self.file_info.original_content = self.file_info.content
        self.is_modified = False
        self.modificationStateChanged.emit(self.file_info.path, False)


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Basic Python syntax highlighter"""

    def __init__(self, parent: QTextDocument):
        super().__init__(parent)
        self.highlighting_rules = []

        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#C586C0"))
        keyword_format.setFontWeight(QFont.Weight.Bold)

        keywords = [
            "def", "class", "import", "from", "return", "if", "elif", "else",
            "for", "while", "try", "except", "finally", "with", "as", "pass",
            "break", "continue", "yield", "and", "or", "not", "in", "is",
            "True", "False", "None", "lambda", "async", "await"
        ]

        for word in keywords:
            pattern = f"\\b{word}\\b"
            self.highlighting_rules.append((pattern, keyword_format))

        # Strings
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))
        self.highlighting_rules.append((r'"[^"\\]*(\\.[^"\\]*)*"', string_format))
        self.highlighting_rules.append((r"'[^'\\]*(\\.[^'\\]*)*'", string_format))

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6A9955"))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((r"#[^\n]*", comment_format))

    def highlightBlock(self, text: str):
        import re
        for pattern, format_rule in self.highlighting_rules:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), format_rule)


class StreamingProgressWidget(QWidget):
    """Widget to show streaming progress for micro-task workflow"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StreamingProgressWidget")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #98c379; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.current_task_label = QLabel("")
        self.current_task_label.setStyleSheet("color: #61AFEF; font-size: 11px;")
        layout.addWidget(self.current_task_label)

    def set_status(self, message: str, color: str = "#98c379"):
        """Update status message"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def show_progress(self, current: int, total: int, task: str = ""):
        """Show progress for current operation"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setVisible(True)
        self.current_task_label.setText(task)

    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.setVisible(False)
        self.current_task_label.setText("")


class CodeViewerWindow(QMainWindow):
    """
    Main code viewer/IDE window for viewing and editing generated code.
    Supports file tree navigation, tabbed editing, and integration with the micro-task workflow.
    """

    # Signals expected by DialogService
    apply_change_requested = Signal(str, str, str, str)  # project_id, relative_filepath, new_content, focus_prefix
    projectFilesSaved = Signal(str, str, str)  # project_id, file_path, content
    focusSetOnFiles = Signal(str, list)  # project_id, file_paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ava Code Viewer")
        self.setObjectName("CodeViewerWindow")
        self.setMinimumSize(1200, 800)

        # Track open files and state
        self.open_files: Dict[str, FileInfo] = {}
        self.current_project_path: Optional[str] = None
        self.current_project_id: Optional[str] = None
        self.focus_prefix: Optional[str] = None

        # Get event bus instance
        try:
            self._event_bus = EventBus.get_instance()
        except:
            self._event_bus = None
            logger.warning("Could not get EventBus instance")

        self._setup_ui()
        self._setup_connections()
        self._setup_menu_bar()
        self._setup_tool_bar()
        self._setup_status_bar()

    def _setup_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel (file tree + controls)
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel (editor tabs + streaming progress)
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([300, 900])

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with file tree and controls"""
        panel = QWidget()
        panel.setObjectName("LeftPanel")
        panel.setMaximumWidth(400)

        layout = QVBoxLayout(panel)

        # File tree
        tree_group = QGroupBox("Project Files")
        tree_layout = QVBoxLayout(tree_group)

        self.file_tree = FileTreeWidget()
        tree_layout.addWidget(self.file_tree)

        # File tree controls
        tree_controls = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_project)
        tree_controls.addWidget(self.refresh_btn)

        self.focus_btn = QPushButton("Set Focus")
        self.focus_btn.clicked.connect(self.set_focus_directory)
        tree_controls.addWidget(self.focus_btn)

        tree_layout.addLayout(tree_controls)
        layout.addWidget(tree_group)

        # Micro-task controls
        task_group = QGroupBox("Micro-Task Workflow")
        task_layout = QVBoxLayout(task_group)

        self.streaming_progress = StreamingProgressWidget()
        task_layout.addWidget(self.streaming_progress)

        # Task control buttons
        task_controls = QVBoxLayout()

        self.start_planning_btn = QPushButton("Start Planning")
        self.start_planning_btn.clicked.connect(self._start_planning)
        task_controls.addWidget(self.start_planning_btn)

        self.approve_plan_btn = QPushButton("Approve Plan")
        self.approve_plan_btn.setEnabled(False)
        task_controls.addWidget(self.approve_plan_btn)

        self.generate_code_btn = QPushButton("Generate Code")
        self.generate_code_btn.setEnabled(False)
        task_controls.addWidget(self.generate_code_btn)

        task_layout.addLayout(task_controls)
        layout.addWidget(task_group)

        layout.addStretch()
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with editor tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Editor tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        layout.addWidget(self.tab_widget)

        return panel

    def _setup_connections(self):
        """Setup signal connections"""
        # File tree connections
        self.file_tree.fileSelected.connect(self.open_file)
        self.file_tree.directoryFocused.connect(self.set_focus_directory)

    def _setup_menu_bar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_project_action = QAction("Open Project", self)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        file_menu.addSeparator()

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_current_file)
        file_menu.addAction(save_action)

        save_all_action = QAction("Save All", self)
        save_all_action.setShortcut("Ctrl+Shift+S")
        save_all_action.triggered.connect(self.save_all_files)
        file_menu.addAction(save_all_action)

        # View menu
        view_menu = menubar.addMenu("View")

        refresh_action = QAction("Refresh Project", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_project)
        view_menu.addAction(refresh_action)

    def _setup_tool_bar(self):
        """Setup the tool bar"""
        toolbar = self.addToolBar("Main")

        # Project info
        self.project_label = QLabel("No project loaded")
        toolbar.addWidget(self.project_label)

        toolbar.addSeparator()

        # Quick actions
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_current_file)
        toolbar.addWidget(save_btn)

        apply_btn = QPushButton("Apply Changes")
        apply_btn.clicked.connect(self.apply_current_changes)
        toolbar.addWidget(apply_btn)

    def _setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # === Public Interface Methods (Expected by DialogService) ===

    def add_generated_file(self, filename: str, content: str, project_id: Optional[str] = None):
        """Add a newly generated file to the viewer (expected by DialogService)"""
        # Determine full path
        if self.focus_prefix and not os.path.isabs(filename):
            file_path = os.path.join(self.focus_prefix, filename)
        else:
            file_path = filename

        # Create file info
        file_info = FileInfo(
            path=file_path,
            content=content,
            original_content=content,
            status=FileChangeStatus.NEW,
            project_id=project_id or self.current_project_id,
            is_ai_generated=True,
            language=self._guess_language(filename)
        )

        # Add to tracking
        self.open_files[file_path] = file_info

        # Create the file on disk if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Open in editor
        self.open_file(file_path)

        # Refresh project tree if needed
        if self.current_project_path and file_path.startswith(self.current_project_path):
            self.refresh_project()

        logger.info(f"Added generated file: {filename}")

    def update_or_add_file(self, filename: str, content: str, is_ai_modification: bool = True,
                           original_content: Optional[str] = None, project_id_for_apply: Optional[str] = None,
                           focus_prefix_for_apply: Optional[str] = None):
        """Update or add file (fallback method expected by DialogService)"""
        self.add_generated_file(filename, content, project_id_for_apply)

    # === File Management Methods ===

    @Slot(str)
    def open_project(self, project_path: str = ""):
        """Open a project directory"""
        if not project_path:
            project_path = QFileDialog.getExistingDirectory(
                self, "Select Project Directory"
            )

        if project_path and os.path.exists(project_path):
            self.current_project_path = project_path
            self.focus_prefix = project_path
            self.file_tree.load_project_structure(project_path)
            self.project_label.setText(f"Project: {os.path.basename(project_path)}")
            self.status_bar.showMessage(f"Loaded project: {project_path}")

    @Slot(str)
    def open_file(self, file_path: str):
        """Open a file in the editor"""
        # Check if already open
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditorTab) and widget.file_info.path == file_path:
                self.tab_widget.setCurrentIndex(i)
                return

        # Create new file info if not tracked
        if file_path not in self.open_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                file_info = FileInfo(
                    path=file_path,
                    content=content,
                    original_content=content,
                    status=FileChangeStatus.UNCHANGED,
                    project_id=self.current_project_id,
                    language=self._guess_language(file_path)
                )
                self.open_files[file_path] = file_info
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open file: {e}")
                return

        # Create editor tab
        file_info = self.open_files[file_path]
        editor = CodeEditorTab(file_info)
        editor.contentChanged.connect(self._on_file_content_changed)
        editor.modificationStateChanged.connect(self._on_file_modification_changed)

        # Add tab
        file_name = os.path.basename(file_path)
        index = self.tab_widget.addTab(editor, file_name)
        self.tab_widget.setCurrentIndex(index)

        self.status_bar.showMessage(f"Opened: {file_path}")

    @Slot(int)
    def close_tab(self, index: int):
        """Close a tab"""
        if index < 0 or index >= self.tab_widget.count():
            return

        widget = self.tab_widget.widget(index)
        if isinstance(widget, CodeEditorTab):
            if widget.is_modified:
                reply = QMessageBox.question(
                    self, "Unsaved Changes",
                    f"File '{os.path.basename(widget.file_info.path)}' has unsaved changes. Save before closing?",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
                )

                if reply == QMessageBox.Save:
                    widget.save_content()
                elif reply == QMessageBox.Cancel:
                    return

        self.tab_widget.removeTab(index)

    @Slot()
    def save_current_file(self):
        """Save the currently active file"""
        current_widget = self.tab_widget.currentWidget()
        if isinstance(current_widget, CodeEditorTab):
            current_widget.save_content()

            # Write to disk
            try:
                with open(current_widget.file_info.path, 'w', encoding='utf-8') as f:
                    f.write(current_widget.file_info.content)

                # Update tab title to remove modified indicator
                index = self.tab_widget.currentIndex()
                file_name = os.path.basename(current_widget.file_info.path)
                self.tab_widget.setTabText(index, file_name)

                self.status_bar.showMessage(f"Saved: {current_widget.file_info.path}")

                # Emit signal for project tracking
                if self._event_bus:
                    self.projectFilesSaved.emit(
                        current_widget.file_info.project_id or "",
                        current_widget.file_info.path,
                        current_widget.file_info.content
                    )

            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save file: {e}")

    @Slot()
    def save_all_files(self):
        """Save all open files"""
        saved_count = 0
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditorTab) and widget.is_modified:
                widget.save_content()
                try:
                    with open(widget.file_info.path, 'w', encoding='utf-8') as f:
                        f.write(widget.file_info.content)
                    saved_count += 1

                    # Update tab title
                    file_name = os.path.basename(widget.file_info.path)
                    self.tab_widget.setTabText(i, file_name)

                except Exception as e:
                    QMessageBox.warning(self, "Save Error", f"Could not save {widget.file_info.path}: {e}")

        self.status_bar.showMessage(f"Saved {saved_count} files")

    @Slot()
    def apply_current_changes(self):
        """Apply changes from current file to the project"""
        current_widget = self.tab_widget.currentWidget()
        if isinstance(current_widget, CodeEditorTab):
            file_info = current_widget.file_info

            if file_info.project_id and self.focus_prefix:
                # Determine relative path
                if file_info.path.startswith(self.focus_prefix):
                    relative_path = os.path.relpath(file_info.path, self.focus_prefix)
                else:
                    relative_path = os.path.basename(file_info.path)

                # Emit apply signal
                self.apply_change_requested.emit(
                    file_info.project_id,
                    relative_path,
                    current_widget.get_content(),
                    self.focus_prefix or ""
                )

                self.status_bar.showMessage(f"Applied changes for: {relative_path}")
            else:
                QMessageBox.warning(self, "Apply Error", "Cannot apply changes: missing project context")

    @Slot()
    def refresh_project(self):
        """Refresh the project file tree"""
        if self.current_project_path:
            self.file_tree.load_project_structure(self.current_project_path)
            self.status_bar.showMessage("Project refreshed")

    @Slot(str)
    def set_focus_directory(self, directory_path: str = ""):
        """Set the focus directory for file operations"""
        if not directory_path:
            directory_path = QFileDialog.getExistingDirectory(
                self, "Select Focus Directory"
            )

        if directory_path and os.path.exists(directory_path):
            self.focus_prefix = directory_path
            self.status_bar.showMessage(f"Focus set to: {directory_path}")

            # Emit signal for broader application awareness
            if self._event_bus and self.current_project_id:
                self.focusSetOnFiles.emit(self.current_project_id, [directory_path])

    # === Micro-Task Workflow Methods ===

    @Slot()
    def _start_planning(self):
        """Start the micro-task planning process"""
        self.streaming_progress.set_status("Starting planning phase...", "#61AFEF")
        self.streaming_progress.show_progress(0, 100, "Initializing planner...")

        # Emit planning request to event bus
        if self._event_bus:
            # This would trigger your planner AI
            self._event_bus.requestMicroTaskPlanning.emit(self.current_project_id or "")

    def update_streaming_progress(self, phase: str, current: int, total: int, task: str = ""):
        """Update streaming progress display"""
        self.streaming_progress.set_status(f"Phase: {phase}")
        self.streaming_progress.show_progress(current, total, task)

    def add_streaming_file(self, filename: str, content_chunk: str, is_complete: bool = False):
        """Add or update a file being streamed from the micro-task workflow"""
        # This would be called as files are being generated
        if filename not in self.open_files:
            self.add_generated_file(filename, content_chunk)
        else:
            # Update existing file
            file_info = self.open_files[filename]
            file_info.content += content_chunk

            # Update editor if open
            for i in range(self.tab_widget.count()):
                widget = self.tab_widget.widget(i)
                if isinstance(widget, CodeEditorTab) and widget.file_info.path == filename:
                    widget.setPlainText(file_info.content)
                    break

        if is_complete:
            self.status_bar.showMessage(f"Completed: {filename}")

    # === Helper Methods ===

    def _guess_language(self, filename: str) -> str:
        """Guess programming language from filename"""
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bat': 'batch'
        }

        return language_map.get(ext, 'text')

    def _on_file_content_changed(self, file_path: str, content: str):
        """Handle file content changes"""
        if file_path in self.open_files:
            self.open_files[file_path].content = content

    def _on_file_modification_changed(self, file_path: str, is_modified: bool):
        """Handle file modification state changes"""
        # Update tab title to show modification state
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditorTab) and widget.file_info.path == file_path:
                file_name = os.path.basename(file_path)
                if is_modified:
                    file_name = f"*{file_name}"
                self.tab_widget.setTabText(i, file_name)
                break

    def get_open_files(self) -> List[str]:
        """Get list of currently open file paths"""
        return list(self.open_files.keys())

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a tracked file"""
        if file_path in self.open_files:
            return self.open_files[file_path].content
        return None