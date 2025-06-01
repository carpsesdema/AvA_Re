# app/core/event_bus.py
import logging
from typing import Optional
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class EventBus(QObject):
    _instance: Optional['EventBus'] = None

    # User Actions
    userMessageSubmitted = Signal(str, list)  # text, image_data_list
    newChatRequested = Signal()  # UI requests CM to signal orchestrator for new session
    chatLlmPersonalitySubmitted = Signal(str, str)  # new_prompt, backend_id_for_persona
    chatLlmSelectionChanged = Signal(str, str)  # backend_id, model_name
    specializedLlmSelectionChanged = Signal(str, str)  # backend_id, model_name

    # RAG Signals
    requestRagScanDirectory = Signal(str)  # dir_path (targets GLOBAL RAG)
    requestProjectFilesUpload = Signal(list, str)  # file_paths: List[str], project_id: str

    # Orchestrator-level request for new session
    createNewSessionForProjectRequested = Signal(str)  # project_id (ChatManager emits this)

    # UI Navigation / Dialog Triggers section
    showLlmLogWindowRequested = Signal()
    chatLlmPersonalityEditRequested = Signal()
    viewCodeViewerRequested = Signal()
    showProjectRagDialogRequested = Signal()
    createNewProjectRequested = Signal(str, str)  # project_name, project_description
    openProjectSelectorRequested = Signal()
    renameCurrentSessionRequested = Signal(str)  # new_session_name

    # Loading Overlay Controls
    showLoader = Signal(str)  # message: str
    hideLoader = Signal()
    updateLoaderMessage = Signal(str)  # message: str

    # Backend & LLM Communication
    backendConfigurationChanged = Signal(str, str, bool, list) # backend_id, model_name, is_configured, available_models
    llmRequestSent = Signal(str, str) # backend_id, request_id
    llmStreamStarted = Signal(str) # request_id
    llmStreamChunkReceived = Signal(str, str) # request_id, chunk_text
    llmResponseCompleted = Signal(str, object, dict) # request_id, chat_message_obj, usage_stats_dict
    llmResponseError = Signal(str, str) # request_id, error_message_str

    # Chat History & Session Management
    newMessageAddedToHistory = Signal(str, str, object)  # project_id, session_id, chat_message_obj
    activeSessionHistoryCleared = Signal(str, str)  # project_id, session_id
    activeSessionHistoryLoaded = Signal(str, str, list)  # project_id, session_id, history_list
    messageChunkReceivedForSession = Signal(str, str, str, str)  # project_id, session_id, request_id, chunk_text
    messageFinalizedForSession = Signal(str, str, str, object, dict,
                                        bool)  # project_id, session_id, request_id, message_obj, usage_dict, is_error

    # Terminal Command Execution
    terminalCommandRequested = Signal(str, str, str)  # command, working_directory, command_id
    terminalCommandStarted = Signal(str, str)  # command_id, command
    terminalCommandOutput = Signal(str, str, str)  # command_id, output_type, content
    terminalCommandCompleted = Signal(str, int, float)  # command_id, exit_code, execution_time
    terminalCommandError = Signal(str, str)  # command_id, error_message

    # Global UI Updates
    uiStatusUpdateGlobal = Signal(str, str, bool, int) # message, color_hex, is_temporary, duration_ms
    uiErrorGlobal = Signal(str, bool) # error_message, is_critical
    uiTextCopied = Signal(str, str) # message, color_hex
    uiInputBarBusyStateChanged = Signal(bool) # is_busy
    backendBusyStateChanged = Signal(bool) # is_busy
    ragStatusChanged = Signal(bool, str, str) # is_ready, status_text, status_color

    # Code Generation / Modification Flow
    modificationFileReadyForDisplay = Signal(str, str) # filename, content
    applyFileChangeRequested = Signal(str, str, str, str) # project_id, relative_filepath, new_content, focus_prefix

    # Multi-Project IDE Signals
    projectFilesSaved = Signal(str, str, str)  # project_id, file_path, content
    projectLoaded = Signal(str, str)  # project_id, project_path
    projectUnloaded = Signal(str)  # project_id
    focusSetOnFiles = Signal(str, list)  # project_id, file_paths_list

    # Code Viewer IDE Signals
    codeViewerProjectLoaded = Signal(str, str, str)  # project_name, project_path, project_id
    codeViewerFileSaved = Signal(str, str, str)  # project_id, file_path, content
    codeViewerFocusRequested = Signal(str, list)  # project_id, file_paths

    # RAG Sync Signals
    ragProjectSyncRequested = Signal(str, str, str)  # project_id, file_path, content
    ragProjectSyncCompleted = Signal(str, str, bool)  # project_id, file_path, success
    ragProjectInitializationRequested = Signal(str, str)  # project_id, project_path

    # File System Integration
    fileSystemWatchRequested = Signal(str, str)  # project_id, project_path
    fileSystemChangeDetected = Signal(str, str, str)  # project_id, file_path, change_type

    # Live Code Intelligence Signals
    codeContextUpdated = Signal(str, str)  # file_path, context_summary
    fileStateChanged = Signal(str, object)  # file_path, FileState
    activeFileChanged = Signal(str)  # file_path
    liveCodeAnalysisReady = Signal(str, dict)  # project_id, analysis_overview

    # Update System Signals
    checkForUpdatesRequested = Signal()
    updateAvailable = Signal(object)  # UpdateInfo
    noUpdateAvailable = Signal()
    updateCheckFailed = Signal(str)
    updateDownloadRequested = Signal(object)
    updateDownloaded = Signal(str)
    updateDownloadFailed = Signal(str)
    updateInstallRequested = Signal(str)
    updateProgress = Signal(int)
    updateStatusChanged = Signal(str)
    applicationRestartRequested = Signal()

    # Autonomous coding
    forcePlanAndCodeGenerationRequested = Signal()


    @staticmethod
    def get_instance() -> 'EventBus':
        if EventBus._instance is None:
            EventBus._instance = EventBus()
            logger.info(f"EventBus singleton instance created: {id(EventBus._instance)}")
        return EventBus._instance

    def __init__(self, parent: Optional[QObject] = None):
        # This check ensures that even if someone accidentally calls EventBus() multiple times,
        # they get the same instance, preserving the singleton pattern.
        # However, the primary way to get it should be EventBus.get_instance().
        if EventBus._instance is not None and id(self) != id(EventBus._instance):
            logger.warning(
                f"EventBus re-instantiated (ID: {id(self)}). This might lead to issues. "
                f"Always use EventBus.get_instance(). Singleton ID: {id(EventBus._instance)}."
            )
            # To strictly enforce singleton, one might raise an exception here,
            # or simply make the __init__ private (e.g. _EventBus__init__) and only allow creation via get_instance.
            # For now, we'll allow it but log a warning.
            super().__init__(parent)
            # Do NOT reassign EventBus._instance = self here if it's not the first one.
            # The first one created via get_instance() is the true singleton.
        else:
            super().__init__(parent)
            if EventBus._instance is None: # This will be true for the first call, typically from get_instance()
                EventBus._instance = self
                logger.info(f"EventBus instance {id(self)} initialized (Parent: {parent}). This is the primary instance.")
            # If id(self) == id(EventBus._instance), it means get_instance() called __init__ on itself, which is fine.

        # The signals are defined at class level, so they are shared across all instances,
        # but only connections to the singleton instance's signals will be effective globally.
        logger.debug(f"EventBus instance {id(self)} signals defined and ready.")