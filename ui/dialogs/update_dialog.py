# app/ui/dialogs/update_dialog.py
import datetime
import logging
import os
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QLabel, QProgressBar, QWidget
)

try:
    # For UpdateInfo type hint and constants
    from services.update_service import UpdateInfo
    from utils import constants
except ImportError as e_ud:
    logging.getLogger(__name__).critical(f"UpdateDialog: Critical import error: {e_ud}", exc_info=True)
    # Fallback types
    UpdateInfo = type("UpdateInfo", (object,), {})  # type: ignore
    constants = type("constants", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class UpdateDialog(QDialog):
    """
    Dialog to display information about an available application update
    and allow the user to initiate download and installation.
    """
    download_requested = Signal(object)  # Emits UpdateInfo object
    install_requested = Signal(str)  # Emits downloaded file path
    restart_requested = Signal()  # Emits when app should restart after install (not used by dialog itself)

    def __init__(self, update_info: UpdateInfo, parent: Optional[QWidget] = None):
        super().__init__(parent)

        if not isinstance(update_info, UpdateInfo):  # type: ignore
            logger.error("UpdateDialog initialized with invalid UpdateInfo object.")
            # Close immediately or show an error
            QTimer.singleShot(0, self.reject)  # type: ignore
            return

        self._update_info = update_info
        self._downloaded_file_path: Optional[str] = None  # Store path after successful download

        self.setWindowTitle(f"Update Available: {getattr(constants, 'APP_NAME', 'AvA')} v{self._update_info.version}")
        self.setMinimumSize(500, 400)
        self.setObjectName("UpdateDialog")

        self._init_ui()
        self._populate_info()
        self._connect_signals()

        logger.info(f"UpdateDialog shown for version {self._update_info.version}")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # --- Title ---
        title_label = QLabel(f"New Version Available: v{self._update_info.version}")
        title_font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"),
                           getattr(constants, 'CHAT_FONT_SIZE', 10) + 3, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # --- Sub-info (Build Date, Size) ---
        info_layout = QHBoxLayout()
        build_date_str = ""
        if self._update_info.build_date:
            try:
                dt = datetime.fromisoformat(self._update_info.build_date.replace("Z", "+00:00"))
                build_date_str = f"Released: {dt.strftime('%Y-%m-%d')}"
            except ValueError:
                build_date_str = f"Released: {self._update_info.build_date.split('T')[0]}"

        size_str = f"Size: {self._update_info.file_size_mb:.2f} MB" if self._update_info.file_size > 0 else "Size: Unknown"

        info_label_text = f"{build_date_str}  |  {size_str}"
        info_label = QLabel(info_label_text)
        info_label.setStyleSheet("color: #8B949E;")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        # --- Changelog ---
        changelog_label = QLabel("What's New:")
        changelog_label.setFont(
            QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), getattr(constants, 'CHAT_FONT_SIZE', 10),
                  QFont.Weight.DemiBold))
        layout.addWidget(changelog_label)

        self.changelog_text_edit = QTextEdit()
        self.changelog_text_edit.setObjectName("UpdateChangelogTextEdit")
        self.changelog_text_edit.setReadOnly(True)
        self.changelog_text_edit.setAcceptRichText(True)  # Changelog might be Markdown/HTML
        self.changelog_text_edit.setMinimumHeight(100)
        layout.addWidget(self.changelog_text_edit, 1)  # Expandable

        # --- Progress Bar & Status ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("UpdateProgressBar")
        self.progress_bar.setVisible(False)  # Initially hidden
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/%m MB")  # Show percentage and value/max
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setObjectName("UpdateStatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setVisible(False)  # Initially hidden
        layout.addWidget(self.status_label)

        # --- Action Buttons ---
        self.button_layout = QHBoxLayout()
        self.download_button = QPushButton("Download Update")
        self.download_button.setObjectName("UpdateDownloadButton")
        self.download_button.setIcon(self._get_icon("download.svg", "fa5s.download"))

        self.install_button = QPushButton("Install and Restart")
        self.install_button.setObjectName("UpdateInstallButton")
        self.install_button.setIcon(self._get_icon("apply.svg", "fa5s.sync-alt"))  # Or a restart icon
        self.install_button.setVisible(False)  # Initially hidden

        self.later_button = QPushButton("Later")
        self.later_button.setObjectName("UpdateLaterButton")

        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.later_button)
        self.button_layout.addWidget(self.download_button)
        self.button_layout.addWidget(self.install_button)
        self.button_layout.addStretch(1)
        layout.addLayout(self.button_layout)

        self.setLayout(layout)

    def _get_icon(self, icon_filename: str, fallback_qta: Optional[str] = None) -> QIcon:
        assets_path = getattr(constants, "ASSETS_PATH", "./assets")
        icon_path = os.path.join(assets_path, icon_filename)
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        elif fallback_qta and getattr(constants, "QTAWESOME_AVAILABLE", False):
            try:
                import qtawesome as qta  # type: ignore
                return qta.icon(fallback_qta, color="#C9D1D9")  # type: ignore
            except Exception:
                pass
        return QIcon()

    def _populate_info(self):
        """Populates the dialog with update information."""
        # Attempt to render changelog as HTML if it looks like Markdown
        changelog_html = self._update_info.changelog
        if "```" in changelog_html or "\n*" in changelog_html or "\n#" in changelog_html:  # Basic Markdown check
            try:
                import markdown
                changelog_html = markdown.markdown(self._update_info.changelog,
                                                   extensions=['extra', 'nl2br', 'sane_lists', 'fenced_code'])
            except ImportError:
                logger.warning("Markdown library not found for update changelog rendering.")
                changelog_html = self._update_info.changelog.replace("\n", "<br>")  # Simple newline to <br>
            except Exception as e_md:
                logger.error(f"Error rendering changelog markdown: {e_md}")
                changelog_html = self._update_info.changelog.replace("\n", "<br>")

        self.changelog_text_edit.setHtml(f"<body>{changelog_html}</body>")  # Wrap in body for styling

    def _connect_signals(self):
        self.download_button.clicked.connect(self._on_download_clicked)
        self.install_button.clicked.connect(self._on_install_clicked)
        self.later_button.clicked.connect(self.reject)  # Closes the dialog

    @Slot()
    def _on_download_clicked(self):
        self.download_button.setEnabled(False)
        self.later_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing download...")
        self.status_label.setVisible(True)
        self.download_requested.emit(self._update_info)

    @Slot()
    def _on_install_clicked(self):
        if self._downloaded_file_path:
            self.install_button.setEnabled(False)
            self.later_button.setEnabled(False)  # Also disable later once install starts
            self.status_label.setText("Preparing to install update...")
            self.install_requested.emit(self._downloaded_file_path)
            # The main application will handle the actual installation and restart.
            # This dialog might close or show a "Restarting..." message.
        else:
            logger.error("Install clicked, but no downloaded file path is set.")
            self.status_label.setText("Error: Downloaded file not found.")

    # --- Slots for UpdateService signals (connected by DialogService) ---
    @Slot(int)
    def update_progress(self, percentage: int):
        if percentage == -1:  # Indeterminate
            self.progress_bar.setFormat("Downloading...")
            self.progress_bar.setRange(0, 0)  # Makes it indeterminate
        else:
            self.progress_bar.setFormat("%p% - %v/%m MB")
            if self.progress_bar.minimum() == 0 and self.progress_bar.maximum() == 0:  # Switch back from indeterminate
                self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percentage)

            # Update max value of progress bar if file size is known
            if self._update_info.file_size > 0 and self.progress_bar.maximum() != self._update_info.file_size // (
                    1024 * 1024):  # In MB
                self.progress_bar.setMaximum(self._update_info.file_size // (1024 * 1024))
            if self._update_info.file_size > 0:
                current_mb = (self._update_info.file_size * (percentage / 100.0)) / (1024 * 1024)
                self.progress_bar.setValue(int(current_mb))

    @Slot(str)
    def update_status(self, status_message: str):
        self.status_label.setText(status_message)
        self.status_label.setVisible(True)

    @Slot(str)  # downloaded_file_path
    def download_completed(self, file_path: str):
        self._downloaded_file_path = file_path
        self.status_label.setText(f"Downloaded: {os.path.basename(file_path)}")
        self.progress_bar.setValue(self.progress_bar.maximum() if self.progress_bar.maximum() > 0 else 100)  # Show 100%
        self.download_button.setVisible(False)
        self.install_button.setVisible(True)
        self.install_button.setEnabled(True)
        self.later_button.setEnabled(True)  # Re-enable later button

    @Slot(str)  # error_message
    def download_failed(self, error_message: str):
        self.status_label.setText(f"Download Failed: {error_message}")
        self.status_label.setStyleSheet("color: #F85149;")  # Error color
        self.progress_bar.setVisible(False)
        self.download_button.setEnabled(True)  # Allow retry
        self.later_button.setEnabled(True)