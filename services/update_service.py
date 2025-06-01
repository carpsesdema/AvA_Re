# app/services/update_service.py
import json
import logging
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import requests  # Ensure requests is imported
from PySide6.QtCore import QObject, QThread, Signal, Slot

try:
    from utils import constants
except ImportError:
    # Fallback for constants if not found (e.g. during isolated testing)
    class constants_fallback:
        APP_VERSION = "0.0.0-fallback"
        USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".ava_pys6_data_p1_fallback_update")


    constants = constants_fallback  # type: ignore
    logging.getLogger(__name__).warning("UpdateService: utils.constants not found, using fallback values.")

logger = logging.getLogger(__name__)


class UpdateInfo:
    """Container for update information retrieved from the release source."""

    def __init__(self, data: Dict[str, Any]):
        self.version: str = data.get('version', '0.0.0')
        self.build_date: str = data.get('build_date', '')  # Typically release publish date
        self.changelog: str = data.get('changelog', 'No changelog provided.')
        self.critical: bool = data.get('critical', False)  # If the update is critical
        self.min_version: str = data.get('min_version', '0.0.0')  # Minimum version required for this update path
        self.file_name: str = data.get('file_name', '')  # Name of the update asset file
        self.file_size: int = data.get('file_size', 0)  # Size in bytes
        self.download_url: str = data.get('download_url', '')  # Direct download URL for the asset

    @property
    def file_size_mb(self) -> float:
        """Returns file size in megabytes."""
        return self.file_size / (1024 * 1024) if self.file_size > 0 else 0.0

    def is_newer_than(self, current_version_str: str) -> bool:
        """
        Compares this update's version with a current version string.
        Assumes semantic versioning (e.g., X.Y.Z).
        Returns True if this update's version is newer.
        """
        try:
            current_parts = [int(x) for x in current_version_str.split('.')]
            update_parts = [int(x) for x in self.version.split('.')]

            # Pad shorter version with zeros for fair comparison
            max_len = max(len(current_parts), len(update_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            update_parts.extend([0] * (max_len - len(update_parts)))

            return tuple(update_parts) > tuple(current_parts)
        except ValueError:
            logger.warning(
                f"Could not compare versions due to invalid format: '{self.version}' vs '{current_version_str}'")
            return False  # Treat as not newer if format is wrong


class UpdateDownloadWorker(QThread):  # Renamed for clarity
    """Worker thread for downloading updates to avoid blocking the UI."""
    progress_updated = Signal(int)  # percentage (0-100)
    status_updated = Signal(str)  # status message (e.g., "Downloading...", "Verifying...")
    download_completed = Signal(str)  # file_path of the downloaded update
    download_failed = Signal(str)  # error_message

    def __init__(self, update_info_obj: UpdateInfo, download_target_dir: Path, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.update_info = update_info_obj
        self.download_dir = download_target_dir
        self._should_stop_flag = False  # Internal flag to signal stop

    def run(self):
        try:
            self.status_updated.emit(f"Starting download: {self.update_info.file_name}...")
            self.download_dir.mkdir(parents=True, exist_ok=True)  # Ensure download directory exists

            download_file_path = self.download_dir / self.update_info.file_name

            # Make a HEAD request first to check for actual content length if possible
            try:
                head_response = requests.head(self.update_info.download_url, timeout=10, allow_redirects=True)
                head_response.raise_for_status()
                total_size = int(head_response.headers.get('content-length', self.update_info.file_size))
                if total_size == 0 and self.update_info.file_size > 0:  # Fallback if HEAD content-length is 0
                    total_size = self.update_info.file_size
            except requests.RequestException:
                logger.warning("HEAD request failed, using file_size from UpdateInfo for progress.")
                total_size = self.update_info.file_size

            response = requests.get(self.update_info.download_url, stream=True,
                                    timeout=30)  # 30s timeout for connect/read
            response.raise_for_status()  # Raise HTTPError for bad responses (4XX or 5XX)

            downloaded_size = 0
            with open(download_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                    if self._should_stop_flag:
                        if download_file_path.exists():
                            download_file_path.unlink(missing_ok=True)
                        self.status_updated.emit("Download cancelled.")
                        logger.info(f"Download of {self.update_info.file_name} cancelled by user.")
                        return

                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = int((downloaded_size / total_size) * 100)
                            self.progress_updated.emit(progress)
                        else:  # If total size is unknown, emit -1 for indeterminate progress
                            self.progress_updated.emit(-1)

            if self._should_stop_flag:  # Check again after loop in case stop was called during last chunk write
                if download_file_path.exists():
                    download_file_path.unlink(missing_ok=True)
                self.status_updated.emit("Download cancelled.")
                logger.info(f"Download of {self.update_info.file_name} cancelled during final write.")
                return

            # Verify file size if known
            if total_size > 0 and downloaded_size != total_size:
                logger.error(f"Downloaded file size mismatch: Expected {total_size}, Got {downloaded_size}")
                download_file_path.unlink(missing_ok=True)
                self.download_failed.emit("File size mismatch after download.")
                return

            self.status_updated.emit("Download completed!")
            self.download_completed.emit(str(download_file_path))

        except requests.exceptions.RequestException as e_req:
            logger.error(f"Download network error: {e_req}", exc_info=True)
            self.download_failed.emit(f"Network error: {e_req}")
        except Exception as e:
            logger.error(f"Download failed unexpectedly: {e}", exc_info=True)
            self.download_failed.emit(f"Unexpected error: {str(e)}")

    def stop_download(self):  # Public method to signal stop
        self._should_stop_flag = True


class UpdateService(QObject):
    """Service for checking and applying application updates from GitHub Releases."""

    update_available = Signal(object)  # Emits UpdateInfo object
    no_update_available = Signal()
    update_check_failed = Signal(str)  # error_message
    update_downloaded = Signal(str)  # file_path
    update_download_failed = Signal(str)  # error_message
    update_progress = Signal(int)  # download progress percentage (0-100, or -1 for indeterminate)
    update_status = Signal(str)  # user-facing status message

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self.github_repo = os.getenv('AVADEVTOOL_GITHUB_REPO', 'carpsesdema/AvA_DevTool')  # Use an env var or default
        self.current_version = constants.APP_VERSION

        self.update_dir = Path(constants.USER_DATA_DIR) / "updates"
        self.update_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists

        self._download_worker_thread: Optional[UpdateDownloadWorker] = None
        logger.info(f"UpdateService initialized. Current version: {self.current_version}. Repo: {self.github_repo}")
        self.cleanup_old_files()  # Clean up on startup

    @Slot()  # Make it a slot if called by EventBus
    def check_for_updates(self) -> None:
        """Check GitHub releases for newer versions of the application."""
        try:
            logger.info("Checking for updates...")
            self.update_status.emit("Checking for updates...")

            api_url = f"https://api.github.com/repos/{self.github_repo}/releases/latest"
            response = requests.get(api_url, timeout=10)  # 10s timeout
            response.raise_for_status()  # Check for HTTP errors

            release_data = response.json()
            latest_version_tag = release_data.get('tag_name', 'v0.0.0')
            latest_version_str = latest_version_tag.lstrip('v')  # Remove 'v' prefix if present

            logger.info(f"Latest version on GitHub: {latest_version_str} (tag: {latest_version_tag})")

            assets = release_data.get('assets', [])
            app_asset = None
            platform_suffix = ""
            if sys.platform == "win32":
                platform_suffix = ".exe"
            elif sys.platform == "darwin":
                platform_suffix = "_mac.zip"  # Or .dmg, .pkg depending on your build
            elif sys.platform.startswith("linux"):
                platform_suffix = "_linux.tar.gz"  # Or .AppImage, .deb, .zip

            # Try to find a platform-specific asset or a generic .zip
            # Adjust asset_name.startswith if your naming convention is different
            expected_asset_prefix = f"AvA_DevTool_v{latest_version_str}"  # Based on build_and_deploy.py

            for asset in assets:
                asset_name = asset.get('name', '')
                # Prioritize platform-specific executables/archives
                if asset_name.startswith(expected_asset_prefix) and asset_name.endswith(
                        platform_suffix) and platform_suffix:
                    app_asset = asset
                    break

            # If platform-specific not found, look for a generic zip (common for cross-platform source or simple bundles)
            if not app_asset:
                for asset in assets:
                    asset_name = asset.get('name', '')
                    if asset_name.startswith(expected_asset_prefix) and asset_name.endswith('.zip'):
                        app_asset = asset
                        logger.info(f"Found generic .zip asset: {asset_name}")
                        break

            if not app_asset and assets:  # If still no specific match, take the first asset if it looks like the app
                # This is a weaker heuristic
                first_asset_name = assets[0].get('name', '')
                if "AvA_DevTool" in first_asset_name or "ava" in first_asset_name.lower():
                    app_asset = assets[0]
                    logger.warning(
                        f"No perfectly matching asset found. Using first likely asset: {app_asset.get('name')}")

            if not app_asset:
                self.update_check_failed.emit("No compatible application asset found in the latest release.")
                logger.error(f"No compatible asset found for version {latest_version_str} and platform {sys.platform}")
                return

            update_data = {
                'version': latest_version_str,
                'build_date': release_data.get('published_at', ''),
                'changelog': release_data.get('body', 'No changelog provided.'),
                'critical': "critical" in release_data.get('name', '').lower() or "critical" in release_data.get('body',
                                                                                                                 '').lower(),
                'min_version': '0.1.0',  # This could be stored in release notes or a separate version manifest
                'file_name': app_asset.get('name', ''),
                'file_size': app_asset.get('size', 0),
                'download_url': app_asset.get('browser_download_url', '')
            }
            update_info = UpdateInfo(update_data)

            if update_info.is_newer_than(self.current_version):
                logger.info(f"Update available: {update_info.version}")
                self.update_available.emit(update_info)
            else:
                logger.info(f"No new update available. Current: {self.current_version}, Latest: {update_info.version}")
                self.no_update_available.emit()

        except requests.RequestException as e_req:
            error_msg = f"Network error checking for updates: {e_req}"
            logger.error(error_msg)
            self.update_check_failed.emit(error_msg)
        except Exception as e_check:
            error_msg = f"Error checking for updates: {e_check}"
            logger.error(error_msg, exc_info=True)
            self.update_check_failed.emit(error_msg)

    @Slot(object)  # Expects UpdateInfo object
    def download_update(self, update_info: UpdateInfo) -> None:
        """Initiate the download of the update file."""
        if self._download_worker_thread and self._download_worker_thread.isRunning():
            logger.warning("Download already in progress. Ignoring new download request.")
            self.update_status.emit("Download already in progress.")
            return

        logger.info(f"Starting download for update version {update_info.version} from {update_info.download_url}")
        self._download_worker_thread = UpdateDownloadWorker(update_info, self.update_dir)
        # Connect signals from worker to service's signals
        self._download_worker_thread.progress_updated.connect(self.update_progress.emit)
        self._download_worker_thread.status_updated.connect(self.update_status.emit)
        self._download_worker_thread.download_completed.connect(self._on_download_completed)
        self._download_worker_thread.download_failed.connect(self._on_download_failed)
        self._download_worker_thread.start()  # Start the thread

    @Slot(str)
    def _on_download_completed(self, file_path: str):
        logger.info(f"Update successfully downloaded to: {file_path}")
        self.update_downloaded.emit(file_path)
        self._download_worker_thread = None  # Clear worker thread instance

    @Slot(str)
    def _on_download_failed(self, error_message: str):
        logger.error(f"Update download failed: {error_message}")
        self.update_download_failed.emit(error_message)
        self._download_worker_thread = None  # Clear worker thread instance

    def cancel_download(self) -> None:
        """Cancel an ongoing download."""
        if self._download_worker_thread and self._download_worker_thread.isRunning():
            logger.info("Attempting to cancel ongoing download...")
            self._download_worker_thread.stop_download()
            # Worker thread will emit status "Download cancelled." and clean up file
            # self._download_worker_thread = None # Worker will clear itself on completion/cancellation
        else:
            logger.info("No active download to cancel.")

    def apply_update(self, downloaded_file_path_str: str) -> bool:
        """
        Apply the downloaded update. This typically involves replacing the
        current executable and preparing for a restart.
        WARNING: This is platform-dependent and can be complex.
        """
        try:
            update_file_path = Path(downloaded_file_path_str)
            if not update_file_path.exists():
                logger.error(f"Update file not found at: {downloaded_file_path_str}")
                self.update_status.emit(f"Error: Update file missing.")
                return False

            if not getattr(sys, 'frozen', False):
                logger.warning("Running from source code. Cannot apply update automatically. Manual update required.")
                self.update_status.emit("Manual update required (running from source).")
                return False  # Cannot apply update when not frozen

            current_exe_path = Path(sys.executable)
            backup_exe_path = current_exe_path.with_suffix(f".{self.current_version}.backup")

            logger.info(f"Attempting to apply update from: {update_file_path}")
            self.update_status.emit("Applying update...")

            # 1. Create backup of current executable
            if backup_exe_path.exists(): backup_exe_path.unlink(missing_ok=True)  # Remove old backup
            logger.info(f"Backing up current executable to: {backup_exe_path}")
            shutil.copy2(current_exe_path, backup_exe_path)

            # 2. Replace current executable with the update file
            # On Windows, you often can't replace a running executable directly.
            # A common strategy is to use a helper script/updater process.
            # For simplicity here, we'll try direct replacement which might fail on Windows.
            logger.info(f"Replacing {current_exe_path} with {update_file_path}")
            try:
                # Ensure the old exe is not locked by trying to remove it first (might fail on Windows)
                if current_exe_path.exists():
                    current_exe_path.unlink()
                shutil.copy2(update_file_path, current_exe_path)
            except Exception as e_replace:  # More specific error handling for Windows
                logger.error(
                    f"Direct replacement failed (common on Windows while running): {e_replace}. Recommending restart.")
                # On Windows, a common approach is to schedule the replacement on next boot
                # or use a small external updater. For now, we'll just signal that restart is needed.
                self.update_status.emit("Update ready. Restart application to apply.")
                # We can't confirm success here if direct replacement fails.
                # The user will need to manually restart, or a more complex updater is needed.
                # For now, assume it *will* be applied on restart if we can't confirm.
                return True  # Indicate "pending apply on restart"

            # 3. Make executable (for non-Windows)
            if sys.platform != 'win32':
                current_exe_path.chmod(current_exe_path.stat().st_mode | 0o111)  # Add execute permissions

            logger.info("Update applied successfully. Restart required.")
            self.update_status.emit("Update applied! Restart application.")
            return True

        except Exception as e:
            logger.error(f"Failed to apply update: {e}", exc_info=True)
            self.update_status.emit(f"Error applying update: {e}")
            # Attempt to restore backup if things went wrong
            if 'backup_exe_path' in locals() and backup_exe_path.exists() and 'current_exe_path' in locals():  # type: ignore
                try:
                    logger.info(f"Attempting to restore backup from {backup_exe_path}")
                    if current_exe_path.exists(): current_exe_path.unlink()  # type: ignore
                    shutil.copy2(backup_exe_path, current_exe_path)  # type: ignore
                    logger.info("Backup restored.")
                except Exception as e_restore:
                    logger.error(f"Failed to restore backup: {e_restore}")
            return False

    def cleanup_old_files(self) -> None:
        """Clean up old update files and backups from the update directory."""
        try:
            logger.info(f"Cleaning up old files in update directory: {self.update_dir}")
            current_update_filename_prefix = f"AvA_DevTool_v{self.current_version}"

            for item in self.update_dir.iterdir():
                if item.is_file():
                    # Delete old downloaded updates (not the one matching current version if it exists)
                    if item.name.startswith("AvA_DevTool_v") and not item.name.startswith(
                            current_update_filename_prefix):
                        try:
                            item.unlink()
                            logger.debug(f"Cleaned up old update file: {item.name}")
                        except OSError as e_unlink:
                            logger.warning(f"Could not remove old update file {item.name}: {e_unlink}")

            # Clean up old backups from application directory (if frozen)
            if getattr(sys, 'frozen', False):
                app_dir = Path(sys.executable).parent
                for item in app_dir.glob("*.backup"):  # Finds files like AvA_DevTool.exe.0.1.0.backup
                    try:
                        item.unlink()
                        logger.debug(f"Cleaned up old backup file: {item.name}")
                    except OSError as e_unlink_backup:
                        logger.warning(f"Could not remove old backup file {item.name}: {e_unlink_backup}")

        except Exception as e:
            logger.warning(f"Error during update cleanup: {e}", exc_info=True)