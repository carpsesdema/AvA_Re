# app/ui/dialogs/project_rag_dialog.py
import logging
import os
from typing import Optional, List, Tuple

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QDialogButtonBox, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QCheckBox, QGroupBox, QWidget, QMessageBox
)
from PySide6.QtGui import QFont, QIcon

try:
    from utils import constants  # For styling, paths, etc.
    from core.event_bus import EventBus  # To emit the request to process files
except ImportError as e_prd:
    logging.getLogger(__name__).critical(f"ProjectRagDialog: Critical import error: {e_prd}", exc_info=True)
    # Fallback types
    constants = type("constants", (object,), {})  # type: ignore
    EventBus = type("EventBus", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class ProjectRagDialog(QDialog):
    """
    A dialog for users to select files or directories to add to the
    knowledge base (RAG) for a specific project.
    """

    # Signal emitted when user confirms selections.
    # List of absolute file paths, project_id
    files_selected_for_project_rag = Signal(list, str)

    def __init__(self, project_id: str, project_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Add to Knowledge Base for '{project_name}'")
        self.setMinimumSize(550, 450)
        self.setObjectName("ProjectRagDialog")

        if not project_id or not project_name:
            logger.error("ProjectRagDialog initialized without valid project_id or project_name.")
            # Optionally, close the dialog or show an error
            QTimer.singleShot(0, self.reject)  # type: ignore
            return

        self._project_id = project_id
        self._project_name = project_name
        self._event_bus = EventBus.get_instance()  # type: ignore

        self._selected_files_list: Optional[QListWidget] = None
        self._recursive_checkbox: Optional[QCheckBox] = None  # For directory selection

        self._current_file_paths: List[str] = []

        self._init_ui()
        self._connect_signals()

        logger.info(f"ProjectRagDialog initialized for project: {project_name} ({project_id})")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # --- Title and Description ---
        title_label = QLabel(f"Manage Knowledge for Project: {self._project_name}")
        title_font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"),
                           getattr(constants, 'CHAT_FONT_SIZE', 10) + 2, QFont.Weight.Bold)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        description_label = QLabel(
            "Select files or a directory to add to this project's specific knowledge base. "
            "This information will be used by the AI to answer questions and generate content related to this project."
        )
        description_label.setWordWrap(True)
        description_label.setStyleSheet("color: #8B949E;")
        layout.addWidget(description_label)

        # --- File Selection Area ---
        selection_group = QGroupBox("Selected Items for RAG")
        selection_layout = QVBoxLayout(selection_group)

        self._selected_files_list = QListWidget()
        self._selected_files_list.setObjectName("RagSelectedFilesList")
        self._selected_files_list.setToolTip("Files and folders queued for adding to RAG.")
        selection_layout.addWidget(self._selected_files_list, 1)  # Expandable list

        # Buttons for adding/removing files/folders
        file_button_layout = QHBoxLayout()
        add_files_button = QPushButton("Add Files...")
        add_files_button.setIcon(self._get_icon("add_file.svg", "fa5s.file-medical-alt"))  # Placeholder icon
        add_files_button.clicked.connect(self._add_files)

        add_folder_button = QPushButton("Add Folder...")
        add_folder_button.setIcon(self._get_icon("add_folder.svg", "fa5s.folder-plus"))  # Placeholder icon
        add_folder_button.clicked.connect(self._add_folder)

        remove_selected_button = QPushButton("Remove Selected")
        remove_selected_button.setIcon(self._get_icon("remove.svg", "fa5s.trash-alt"))  # Placeholder icon
        remove_selected_button.clicked.connect(self._remove_selected_items)

        clear_all_button = QPushButton("Clear All")
        clear_all_button.clicked.connect(self._clear_all_items)

        file_button_layout.addWidget(add_files_button)
        file_button_layout.addWidget(add_folder_button)
        file_button_layout.addStretch(1)
        file_button_layout.addWidget(remove_selected_button)
        file_button_layout.addWidget(clear_all_button)
        selection_layout.addLayout(file_button_layout)

        self._recursive_checkbox = QCheckBox("Include subfolders when adding a folder")
        self._recursive_checkbox.setChecked(True)  # Default to recursive
        self._recursive_checkbox.setToolTip(
            "If checked, all supported files in subdirectories of a selected folder will also be included.")
        selection_layout.addWidget(self._recursive_checkbox)

        layout.addWidget(selection_group, 1)  # Selection group takes most space

        # --- Dialog Buttons (OK/Cancel) ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Add to Project RAG")  # type: ignore
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def _get_icon(self, icon_filename: str, fallback_qta: Optional[str] = None) -> QIcon:
        # Simplified icon getter, assuming icons are in assets path
        assets_path = getattr(constants, "ASSETS_PATH", "./assets")
        icon_path = os.path.join(assets_path, icon_filename)
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        elif fallback_qta and hasattr(constants,
                                      "QTAWESOME_AVAILABLE") and constants.QTAWESOME_AVAILABLE:  # type: ignore
            try:
                import qtawesome as qta  # type: ignore
                return qta.icon(fallback_qta, color="#C9D1D9")  # type: ignore
            except Exception:
                pass
        return QIcon()

    def _connect_signals(self):
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    @Slot()
    def _add_files(self):
        """Opens a file dialog to select multiple files."""
        # Use user's home directory or last used path as starting point
        start_dir = os.path.expanduser("~")
        # Construct a filter string from ALLOWED_TEXT_EXTENSIONS in constants
        allowed_ext_list = getattr(constants, 'ALLOWED_TEXT_EXTENSIONS', ['.txt', '.py', '.md'])
        filter_str = "Supported Files (" + " ".join([f"*{ext}" for ext in allowed_ext_list]) + ");;All Files (*)"

        files, _ = QFileDialog.getOpenFileNames(self, "Select Files to Add to Project RAG", start_dir, filter_str)
        if files:
            for file_path in files:
                if file_path not in self._current_file_paths:
                    self._current_file_paths.append(file_path)
                    item = QListWidgetItem(os.path.basename(file_path))
                    item.setData(Qt.ItemDataRole.UserRole, file_path)  # Store full path
                    item.setToolTip(file_path)
                    if self._selected_files_list: self._selected_files_list.addItem(item)
            logger.debug(f"Added files: {files}")

    @Slot()
    def _add_folder(self):
        """Opens a directory dialog to select a folder."""
        start_dir = os.path.expanduser("~")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Add to Project RAG", start_dir)
        if folder_path:
            # For folders, we add the folder path itself. The UploadService will handle recursion if specified.
            # We can display it differently in the list.
            if folder_path not in self._current_file_paths:  # Avoid duplicate folder entries
                self._current_file_paths.append(folder_path)  # Store the folder path
                display_text = f"[Folder] {os.path.basename(folder_path)}"
                if self._recursive_checkbox and self._recursive_checkbox.isChecked():
                    display_text += " (Recursive)"

                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, folder_path)  # Store full path
                item.setToolTip(folder_path)
                if self._selected_files_list: self._selected_files_list.addItem(item)
            logger.debug(f"Added folder: {folder_path}")

    @Slot()
    def _remove_selected_items(self):
        if not self._selected_files_list: return
        selected_items = self._selected_files_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select items from the list to remove.")
            return

        for item in selected_items:
            file_path_to_remove = item.data(Qt.ItemDataRole.UserRole)
            if file_path_to_remove in self._current_file_paths:
                self._current_file_paths.remove(file_path_to_remove)
            self._selected_files_list.takeItem(self._selected_files_list.row(item))
        logger.debug(f"Removed {len(selected_items)} items from selection.")

    @Slot()
    def _clear_all_items(self):
        if self._selected_files_list: self._selected_files_list.clear()
        self._current_file_paths.clear()
        logger.debug("Cleared all selected items for RAG.")

    def accept(self):
        """Handles the OK/ "Add to Project RAG" button click."""
        if not self._current_file_paths:
            QMessageBox.information(self, "No Items Selected", "Please add files or folders to include in the RAG.")
            return

        # The paths in _current_file_paths can be individual files or directories.
        # UploadService will need to handle directory scanning if a path is a directory
        # and if recursion is enabled.

        # We will emit a single signal with all collected paths.
        # The actual recursive scan of directories happens in UploadService.
        # Here, we just pass the top-level selected paths.

        # The `requestProjectFilesUpload` signal takes List[str] (file paths) and project_id
        # If a directory is in `self._current_file_paths`, UploadService needs to know to scan it.
        # For simplicity, UploadService's process_files_for_context can check os.path.isdir
        # and then call process_directory_for_context internally.
        # Or, we can have two separate lists/signals.
        # Let's assume process_files_for_context in UploadService is smart enough.

        logger.info(f"Project RAG dialog accepted for project '{self._project_id}'. "
                    f"Processing {len(self._current_file_paths)} items (files/folders).")

        # Emit a signal that ChatManager (or ApplicationOrchestrator) will listen to.
        # This signal should then trigger UploadService.
        self._event_bus.requestProjectFilesUpload.emit(list(self._current_file_paths), self._project_id)  # type: ignore

        super().accept()  # Close the dialog

    def reject(self):
        """Handles the Cancel button click."""
        logger.info(f"Project RAG dialog cancelled for project '{self._project_id}'.")
        super().reject()

    def get_selected_paths_and_options(self) -> Tuple[List[str], bool]:
        """
        Returns the list of selected file/folder paths and recursion option.
        This method might be used if the dialog isn't directly emitting an EventBus signal.
        """
        is_recursive = self._recursive_checkbox.isChecked() if self._recursive_checkbox else True
        return list(self._current_file_paths), is_recursive