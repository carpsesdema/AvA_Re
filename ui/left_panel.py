# app/ui/left_panel.py
import logging
import os
from typing import Optional, List

from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy,
    QComboBox, QGroupBox, QListWidget, QListWidgetItem, QHBoxLayout,
    QInputDialog, QMessageBox, QFileDialog, QSlider
)

try:
    import qtawesome as qta

    QTAWESOME_AVAILABLE = True
except ImportError:
    QTAWESOME_AVAILABLE = False
    qta = None  # type: ignore
    logging.getLogger(__name__).warning("LeftControlPanel: qtawesome library not found. Icons will be limited.")

try:
    from utils import constants
    from core.event_bus import EventBus
    # ChatManager is used to get initial state and backend info
    from core.chat_manager import ChatManager
    # Project and ChatSession models for list widgets
    from models.project_models import Project, ChatSession
except ImportError as e_lp:
    logging.getLogger(__name__).critical(f"Critical import error in LeftPanel: {e_lp}", exc_info=True)
    # Fallback types
    constants = type("constants", (object,), {})  # type: ignore
    EventBus = type("EventBus", (object,), {})  # type: ignore
    ChatManager = type("ChatManager", (object,), {})  # type: ignore
    Project = type("Project", (object,), {})  # type: ignore
    ChatSession = type("ChatSession", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class LeftControlPanel(QWidget):
    """
    The left-hand control panel for the application.
    Contains LLM selection, chat actions, project/session management,
    RAG controls, and other settings.
    """
    # Roles for storing data in QComboBox and QListWidget items
    MODEL_CONFIG_DATA_ROLE = Qt.ItemDataRole.UserRole + 2  # For QComboBox backend/model data
    PROJECT_ID_ROLE = Qt.ItemDataRole.UserRole + 3  # For QListWidget project items
    SESSION_ID_ROLE = Qt.ItemDataRole.UserRole + 4  # For QListWidget session items

    # Display details for backend prefixes in dropdowns
    BACKEND_DISPLAY_DETAILS = {
        constants.DEFAULT_CHAT_BACKEND_ID: {"prefix": "Gemini",
                                            "default_models": [constants.DEFAULT_GEMINI_CHAT_MODEL]},  # type: ignore
        "ollama_chat_default": {"prefix": "Ollama (Chat)", "default_models": [constants.DEFAULT_OLLAMA_CHAT_MODEL]},
        # type: ignore
        "gpt_chat_default": {"prefix": "GPT", "default_models": [constants.DEFAULT_GPT_CHAT_MODEL]},  # type: ignore
        constants.GENERATOR_BACKEND_ID: {"prefix": "Ollama (Gen)",
                                         "default_models": [constants.DEFAULT_OLLAMA_GENERATOR_MODEL]}  # type: ignore
    }

    def __init__(self, chat_manager: ChatManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LeftControlPanel")

        if not isinstance(chat_manager, ChatManager):  # type: ignore
            logger.critical("LeftControlPanel requires a valid ChatManager instance.")
            raise TypeError("LeftControlPanel requires a valid ChatManager instance.")

        self.chat_manager = chat_manager
        # Get ProjectManager via ChatManager's BackendConfigManager or directly if CM exposes it.
        # Assuming ChatManager provides access to ProjectManager via ApplicationOrchestrator.
        self._project_manager = self.chat_manager.get_project_manager()  # type: ignore
        self._event_bus = EventBus.get_instance()  # type: ignore

        # Flags to prevent signal loops during programmatic UI updates
        self._is_programmatic_model_change: bool = False
        self._is_programmatic_list_selection_change: bool = False  # For project/session lists

        self._init_ui_elements()
        self._init_layout()
        self._connect_ui_signals_to_event_bus()
        self._connect_event_bus_to_ui_updates()

        self._load_initial_settings_and_data()

        logger.info("LeftControlPanel initialized.")

    def _get_qta_icon(self, icon_name: str, color: str = "#00CFE8") -> QIcon:
        """Helper to get a qtawesome icon, with fallback."""
        if QTAWESOME_AVAILABLE and qta:
            try:
                return qta.icon(icon_name, color=color)
            except Exception as e_qta:
                logger.warning(f"qtawesome icon '{icon_name}' not found or error: {e_qta}")
        return QIcon()  # Return empty icon as fallback

    def _init_ui_elements(self):
        """Initializes all UI widgets for the left panel."""
        font_size = getattr(constants, 'CHAT_FONT_SIZE', 10)
        self.button_font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), font_size - 1)
        self.label_font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), font_size - 1)
        self.group_box_font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), font_size - 1,
                                    QFont.Weight.Bold)
        self.button_style_sheet = "QPushButton { text-align: left; padding: 6px 8px; }"
        self.button_icon_size = QSize(16, 16)

        # LLM Configuration Group
        self.llm_config_group = QGroupBox("LLM Configuration")
        self.llm_config_group.setFont(self.group_box_font)
        self.chat_llm_label = QLabel("Chat LLM:")
        self.chat_llm_label.setFont(self.label_font)
        self.chat_llm_combo_box = QComboBox()
        self.chat_llm_combo_box.setFont(self.button_font)
        self.chat_llm_combo_box.setObjectName("ChatLlmComboBox")
        self.chat_llm_combo_box.setToolTip("Select the primary AI model for chat conversations.")
        self.chat_llm_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.specialized_llm_label = QLabel("Specialized LLM (Code Gen):")
        self.specialized_llm_label.setFont(self.label_font)
        self.specialized_llm_combo_box = QComboBox()
        self.specialized_llm_combo_box.setFont(self.button_font)
        self.specialized_llm_combo_box.setObjectName("SpecializedLlmComboBox")
        self.specialized_llm_combo_box.setToolTip("Select AI for code generation and specialized tasks.")
        self.specialized_llm_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.temperature_label = QLabel("Temperature (Chat):")
        self.temperature_label.setFont(self.label_font)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(0, 200)  # Represents 0.0 to 2.0
        self.temperature_slider.setToolTip("Adjust AI creativity for chat (0.0 = focused, 2.0 = creative).")
        self.temperature_slider.setObjectName("TemperatureSlider")
        self.temperature_value_label = QLabel("0.70")  # Default display
        self.temperature_value_label.setFont(QFont(self.label_font.family(), font_size - 2))
        self.temperature_value_label.setStyleSheet("color: #61AFEF;")
        self.temperature_value_label.setMinimumWidth(35)  # Ensure space for "1.00"

        self.configure_ai_personality_button = QPushButton(" Configure Persona")
        self.configure_ai_personality_button.setFont(self.button_font)
        self.configure_ai_personality_button.setIcon(self._get_qta_icon('fa5s.user-cog', color="#DAA520"))
        self.configure_ai_personality_button.setToolTip("Customize chat AI personality / system prompt (Ctrl+P).")
        self.configure_ai_personality_button.setStyleSheet(self.button_style_sheet)
        self.configure_ai_personality_button.setIconSize(self.button_icon_size)

        # Chat Actions Group
        self.actions_group = QGroupBox("Chat Actions")
        self.actions_group.setFont(self.group_box_font)
        self.new_chat_button = QPushButton(" New Session")
        self.new_chat_button.setFont(self.button_font)
        self.new_chat_button.setIcon(self._get_qta_icon('fa5s.comment-dots', color="#61AFEF"))
        self.new_chat_button.setToolTip("Start a new chat session in the current project (Ctrl+N).")
        self.new_chat_button.setStyleSheet(self.button_style_sheet)
        self.new_chat_button.setIconSize(self.button_icon_size)

        self.view_llm_terminal_button = QPushButton(" View LLM Log")
        self.view_llm_terminal_button.setFont(self.button_font)
        self.view_llm_terminal_button.setIcon(self._get_qta_icon('fa5s.terminal', color="#98C379"))
        self.view_llm_terminal_button.setToolTip("Show LLM communication log (Ctrl+L).")
        self.view_llm_terminal_button.setStyleSheet(self.button_style_sheet)
        self.view_llm_terminal_button.setIconSize(self.button_icon_size)

        self.view_generated_code_button = QPushButton(" View Generated Code")
        self.view_generated_code_button.setFont(self.button_font)
        self.view_generated_code_button.setIcon(self._get_qta_icon('fa5s.code', color="#ABB2BF"))
        self.view_generated_code_button.setToolTip("Open or focus the generated code viewer window.")
        self.view_generated_code_button.setStyleSheet(self.button_style_sheet)
        self.view_generated_code_button.setIconSize(self.button_icon_size)

        self.force_code_generation_button = QPushButton(" Force Code Gen")
        self.force_code_generation_button.setFont(self.button_font)
        self.force_code_generation_button.setIcon(self._get_qta_icon('fa5s.cogs', color="#D19A66"))
        self.force_code_generation_button.setToolTip(
            "Force plan-and-code sequence to proceed to code generation (use if stuck).")
        self.force_code_generation_button.setStyleSheet(self.button_style_sheet)
        self.force_code_generation_button.setIconSize(self.button_icon_size)

        self.check_updates_button = QPushButton(" Check for Updates")
        self.check_updates_button.setFont(self.button_font)
        self.check_updates_button.setIcon(self._get_qta_icon('fa5s.download', color="#E5C07B"))
        self.check_updates_button.setToolTip("Check for application updates.")
        self.check_updates_button.setStyleSheet(self.button_style_sheet)
        self.check_updates_button.setIconSize(self.button_icon_size)

        # Projects & Sessions Group
        self.projects_group = QGroupBox("Projects & Sessions")
        self.projects_group.setFont(self.group_box_font)
        self.projects_list_widget = QListWidget()
        self.projects_list_widget.setFont(self.button_font)
        self.projects_list_widget.setToolTip("Select a project to work on.")
        self.projects_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.projects_list_widget.setMaximumHeight(150)  # Limit height for project list

        self.sessions_list_widget = QListWidget()
        self.sessions_list_widget.setFont(self.button_font)
        self.sessions_list_widget.setToolTip("Select a chat session within the current project.")
        self.sessions_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.new_project_button = QPushButton(" New Project")
        self.new_project_button.setFont(self.button_font)
        self.new_project_button.setIcon(self._get_qta_icon('fa5s.folder-plus', color="#56B6C2"))
        self.new_project_button.setToolTip("Create a new project.")
        self.new_project_button.setStyleSheet(self.button_style_sheet)
        self.new_project_button.setIconSize(self.button_icon_size)

        # RAG Group
        self.rag_group = QGroupBox("Knowledge Base (RAG)")
        self.rag_group.setFont(self.group_box_font)
        self.scan_global_rag_directory_button = QPushButton(" Scan Directory (Global)")
        self.scan_global_rag_directory_button.setFont(self.button_font)
        self.scan_global_rag_directory_button.setIcon(self._get_qta_icon('fa5s.globe-americas', color="#E0B6FF"))
        self.scan_global_rag_directory_button.setToolTip(
            "Scan a directory to add its files to the GLOBAL knowledge base.")
        self.scan_global_rag_directory_button.setStyleSheet(self.button_style_sheet)
        self.scan_global_rag_directory_button.setIconSize(self.button_icon_size)

        self.add_project_files_button = QPushButton(" Add Files (Project)")
        self.add_project_files_button.setFont(self.button_font)
        self.add_project_files_button.setIcon(self._get_qta_icon('fa5s.file-medical', color="#61AFEF"))  # Changed icon
        self.add_project_files_button.setToolTip("Add specific files to the CURRENT project's knowledge base.")
        self.add_project_files_button.setStyleSheet(self.button_style_sheet)
        self.add_project_files_button.setIconSize(self.button_icon_size)

        self.rag_status_label = QLabel("RAG Status: Initializing...")
        self.rag_status_label.setFont(QFont(self.label_font.family(), font_size - 2))
        self.rag_status_label.setObjectName("RagStatusLabel")  # For styling
        self.rag_status_label.setWordWrap(True)

    def _init_layout(self):
        """Arranges widgets in layouts."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)  # Overall spacing between groups

        # Projects & Sessions Layout
        project_session_layout = QVBoxLayout(self.projects_group)
        project_session_layout.setSpacing(5)
        project_session_layout.addWidget(QLabel("Projects:"))
        project_session_layout.addWidget(self.projects_list_widget)
        project_session_layout.addWidget(self.new_project_button)
        project_session_layout.addSpacing(8)
        project_session_layout.addWidget(QLabel("Sessions (Current Project):"))
        project_session_layout.addWidget(self.sessions_list_widget)
        main_layout.addWidget(self.projects_group)

        # LLM Config Layout
        llm_config_layout = QVBoxLayout(self.llm_config_group)
        llm_config_layout.setSpacing(5)
        llm_config_layout.addWidget(self.chat_llm_label)
        llm_config_layout.addWidget(self.chat_llm_combo_box)
        llm_config_layout.addWidget(self.specialized_llm_label)
        llm_config_layout.addWidget(self.specialized_llm_combo_box)
        temp_h_layout = QHBoxLayout()  # For temperature slider and label
        temp_h_layout.addWidget(self.temperature_label)
        temp_h_layout.addWidget(self.temperature_slider, 1)  # Slider takes more space
        temp_h_layout.addWidget(self.temperature_value_label)
        llm_config_layout.addLayout(temp_h_layout)
        llm_config_layout.addWidget(self.configure_ai_personality_button)
        main_layout.addWidget(self.llm_config_group)

        # RAG Actions Layout
        rag_actions_layout = QVBoxLayout(self.rag_group)
        rag_actions_layout.setSpacing(5)
        rag_actions_layout.addWidget(self.scan_global_rag_directory_button)
        rag_actions_layout.addWidget(self.add_project_files_button)
        rag_actions_layout.addWidget(self.rag_status_label)
        main_layout.addWidget(self.rag_group)

        # Chat Actions Layout
        actions_layout = QVBoxLayout(self.actions_group)
        actions_layout.setSpacing(5)
        actions_layout.addWidget(self.new_chat_button)
        actions_layout.addWidget(self.view_llm_terminal_button)
        actions_layout.addWidget(self.view_generated_code_button)
        actions_layout.addWidget(self.force_code_generation_button)
        actions_layout.addWidget(self.check_updates_button)
        main_layout.addWidget(self.actions_group)

        main_layout.addStretch(1)  # Push everything up
        self.setLayout(main_layout)

    def _connect_ui_signals_to_event_bus(self):
        """Connects UI widget signals to emit corresponding EventBus signals."""
        self.new_chat_button.clicked.connect(lambda: self._event_bus.newChatRequested.emit())
        self.configure_ai_personality_button.clicked.connect(
            lambda: self._event_bus.chatLlmPersonalityEditRequested.emit())
        self.view_llm_terminal_button.clicked.connect(lambda: self._event_bus.showLlmLogWindowRequested.emit())
        self.view_generated_code_button.clicked.connect(lambda: self._event_bus.viewCodeViewerRequested.emit())
        self.force_code_generation_button.clicked.connect(
            lambda: self._event_bus.forcePlanAndCodeGenerationRequested.emit())
        self.check_updates_button.clicked.connect(lambda: self._event_bus.checkForUpdatesRequested.emit())

        self.chat_llm_combo_box.currentIndexChanged.connect(self._on_chat_llm_selection_changed_by_user)
        self.specialized_llm_combo_box.currentIndexChanged.connect(self._on_specialized_llm_selection_changed_by_user)
        self.temperature_slider.valueChanged.connect(self._on_temperature_slider_changed)

        self.new_project_button.clicked.connect(self._on_new_project_button_clicked)
        self.projects_list_widget.currentItemChanged.connect(self._on_project_list_selection_changed)
        self.sessions_list_widget.currentItemChanged.connect(self._on_session_list_selection_changed)

        self.scan_global_rag_directory_button.clicked.connect(self._on_scan_global_rag_button_clicked)
        self.add_project_files_button.clicked.connect(self._on_add_project_files_button_clicked)

    def _connect_event_bus_to_ui_updates(self):
        """Connects EventBus signals (from core logic) to update UI elements in this panel."""
        self._event_bus.backendConfigurationChanged.connect(self._update_llm_combobox_on_config_change)
        self._event_bus.backendBusyStateChanged.connect(self._update_ui_on_busy_state_change)
        self._event_bus.ragStatusChanged.connect(self._update_rag_status_display)

        # Project/Session list updates from ProjectManager (which emits these to EventBus or directly)
        if self._project_manager:
            self._project_manager.projectsLoaded.connect(self.populate_projects_list)
            self._project_manager.projectCreated.connect(self._handle_project_list_changed)  # Repopulate or add item
            self._project_manager.projectDeleted.connect(self._handle_project_list_changed)  # Repopulate
            self._project_manager.projectSwitched.connect(self._handle_project_switched_by_logic)
            self._project_manager.sessionCreated.connect(self._handle_session_list_changed_for_current_project)
            self._project_manager.sessionSwitched.connect(self._handle_session_switched_by_logic)
            self._project_manager.sessionRenamed.connect(
                self._handle_session_list_changed_for_current_project)  # Repopulate sessions

    def _load_initial_settings_and_data(self):
        """Loads initial settings from ChatManager and populates lists."""
        logger.debug("LCP: Loading initial model settings and project/session data.")
        self._is_programmatic_model_change = True  # Prevent signals during initial population

        # Set initial temperature from ChatManager (via BackendConfigManager)
        initial_temp = self.chat_manager.get_current_chat_temperature()
        self.temperature_slider.setValue(int(initial_temp * 100))
        self.temperature_value_label.setText(f"{initial_temp:.2f}")

        # Populate LLM combo boxes (will use cached models first, then async update)
        self._populate_llm_combobox(self.chat_llm_combo_box, self._get_chat_backend_ids_for_dropdown())
        self._populate_llm_combobox(self.specialized_llm_combo_box, self._get_specialized_backend_ids_for_dropdown())

        # Set current selections
        active_chat_be = self.chat_manager.get_current_active_chat_backend_id()
        active_chat_mdl = self.chat_manager.get_model_for_backend(active_chat_be)
        self._set_combobox_selection(self.chat_llm_combo_box, active_chat_be, active_chat_mdl)

        active_spec_be = self.chat_manager.get_current_active_specialized_backend_id()
        active_spec_mdl = self.chat_manager.get_model_for_backend(active_spec_be)
        self._set_combobox_selection(self.specialized_llm_combo_box, active_spec_be, active_spec_mdl)

        self.update_personality_button_tooltip()

        self._is_programmatic_model_change = False

        # Load projects and sessions
        self.load_initial_projects_and_sessions()  # This will handle its own programmatic flags

        # Initial RAG status update
        self._check_initial_rag_status()
        self.set_panel_enabled_state(not self.chat_manager.is_overall_busy())

    # --- Slot Implementations for UI Element Interactions ---
    @Slot(int)
    def _on_chat_llm_selection_changed_by_user(self, index: int):
        if self._is_programmatic_model_change or index < 0: return
        data = self.chat_llm_combo_box.itemData(index)
        if isinstance(data, dict) and not data.get("is_placeholder"):
            self._event_bus.chatLlmSelectionChanged.emit(data["backend_id"], data["model_name"])

    @Slot(int)
    def _on_specialized_llm_selection_changed_by_user(self, index: int):
        if self._is_programmatic_model_change or index < 0: return
        data = self.specialized_llm_combo_box.itemData(index)
        if isinstance(data, dict) and not data.get("is_placeholder"):
            self._event_bus.specializedLlmSelectionChanged.emit(data["backend_id"], data["model_name"])

    @Slot(int)
    def _on_temperature_slider_changed(self, value: int):
        temperature = value / 100.0
        self.temperature_value_label.setText(f"{temperature:.2f}")
        self.chat_manager.set_chat_temperature(temperature)  # This now goes to BackendConfigManager

    @Slot()
    def _on_new_project_button_clicked(self):
        project_name, ok = QInputDialog.getText(self, "Create New Project", "Enter project name:")
        if ok and project_name.strip():
            self._event_bus.createNewProjectRequested.emit(project_name.strip(), "")  # Empty description for now
        elif ok:  # Empty name provided
            QMessageBox.warning(self, "Invalid Name", "Project name cannot be empty.")

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_project_list_selection_changed(self, current: Optional[QListWidgetItem],
                                           previous: Optional[QListWidgetItem]):
        if self._is_programmatic_list_selection_change or not current: return
        project_id = current.data(self.PROJECT_ID_ROLE)
        if project_id and (not previous or project_id != previous.data(self.PROJECT_ID_ROLE)):
            logger.info(f"LCP: Project selected by user: {project_id}")
            if self._project_manager: self._project_manager.switch_to_project(project_id)
            # Session list will update via _handle_project_switched_by_logic

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_session_list_selection_changed(self, current: Optional[QListWidgetItem],
                                           previous: Optional[QListWidgetItem]):
        if self._is_programmatic_list_selection_change or not current: return
        session_id = current.data(self.SESSION_ID_ROLE)
        if session_id and self._project_manager and self._project_manager.get_current_project() and \
                (not previous or session_id != previous.data(self.SESSION_ID_ROLE)):
            logger.info(f"LCP: Session selected by user: {session_id}")
            self._project_manager.switch_to_session(session_id)

    @Slot()
    def _on_scan_global_rag_button_clicked(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Global RAG Import",
                                                     os.path.expanduser("~"))
        if directory: self._event_bus.requestRagScanDirectory.emit(directory)

    @Slot()
    def _on_add_project_files_button_clicked(self):
        current_project = self._project_manager.get_current_project() if self._project_manager else None  # type: ignore
        if not current_project:
            QMessageBox.warning(self, "No Project", "Please select or create a project first to add files to its RAG.")
            return
        # This will trigger DialogService to show the ProjectRagDialog
        self._event_bus.showProjectRagDialogRequested.emit()

    # --- Slot Implementations for EventBus Signals ---
    @Slot(str, str, bool, list)
    def _update_llm_combobox_on_config_change(self, backend_id: str, model_name: str, is_configured: bool,
                                              available_models: List[str]):
        logger.debug(
            f"LCP: Received backendConfigurationChanged for {backend_id}/{model_name}, Configured: {is_configured}")
        self._is_programmatic_model_change = True
        # Determine which combobox and its associated "purpose" backends need update
        chat_backends = self._get_chat_backend_ids_for_dropdown()
        spec_backends = self._get_specialized_backend_ids_for_dropdown()

        if backend_id in chat_backends:
            self._populate_llm_combobox(self.chat_llm_combo_box, chat_backends, force_refresh_backend=backend_id)
            active_chat_be = self.chat_manager.get_current_active_chat_backend_id()  # From BackendConfigManager
            active_chat_mdl = self.chat_manager.get_model_for_backend(active_chat_be)
            self._set_combobox_selection(self.chat_llm_combo_box, active_chat_be, active_chat_mdl)

        if backend_id in spec_backends:
            self._populate_llm_combobox(self.specialized_llm_combo_box, spec_backends, force_refresh_backend=backend_id)
            active_spec_be = self.chat_manager.get_current_active_specialized_backend_id()  # From BackendConfigManager
            active_spec_mdl = self.chat_manager.get_model_for_backend(active_spec_be)
            self._set_combobox_selection(self.specialized_llm_combo_box, active_spec_be, active_spec_mdl)

        self._is_programmatic_model_change = False
        self.update_personality_button_tooltip()
        self.set_panel_enabled_state(not self.chat_manager.is_overall_busy())

    @Slot(bool)
    def _update_ui_on_busy_state_change(self, is_busy: bool):
        self.set_panel_enabled_state(not is_busy)

    @Slot(bool, str, str)
    def _update_rag_status_display(self, is_ready: bool, status_text: str, status_color: str):
        self.rag_status_label.setText(status_text)
        self.rag_status_label.setStyleSheet(f"QLabel#RagStatusLabel {{ color: {status_color}; }}")
        self.add_project_files_button.setEnabled(
            is_ready and self._project_manager and self._project_manager.get_current_project() is not None)  # type: ignore
        self.scan_global_rag_directory_button.setEnabled(is_ready)

    @Slot(list)
    def populate_projects_list(self, projects: List[Project]):  # type: ignore
        logger.debug(f"LCP: Populating projects list with {len(projects)} projects.")
        self._is_programmatic_list_selection_change = True
        self.projects_list_widget.clear()
        for project in projects:
            item = QListWidgetItem(project.name)
            item.setData(self.PROJECT_ID_ROLE, project.id)
            item.setToolTip(project.description or project.name)  # Show description on hover
            self.projects_list_widget.addItem(item)
        self._is_programmatic_list_selection_change = False
        # After populating, ensure current project is selected if one exists
        current_pm_project = self._project_manager.get_current_project() if self._project_manager else None  # type: ignore
        if current_pm_project:
            self._select_project_in_list_widget(current_pm_project.id)

    @Slot(str)  # project_id
    def _handle_project_list_changed(self,
                                     project_id: Optional[str] = None):  # project_id can be for created or deleted
        """Repopulates the projects list. Called on create or delete."""
        logger.debug("LCP: Project list changed, repopulating.")
        if self._project_manager:
            self.populate_projects_list(self._project_manager.get_all_projects())

    @Slot(str)  # project_id
    def _handle_project_switched_by_logic(self, project_id: str):
        logger.debug(f"LCP: Project switched by logic to {project_id}. Updating UI selection.")
        self._select_project_in_list_widget(project_id)
        self._update_sessions_list_for_project(project_id)  # Update sessions for the new project

    @Slot(str, str)  # project_id, session_id
    def _handle_session_list_changed_for_current_project(self, project_id: str, session_id: Optional[str] = None,
                                                         new_name: Optional[str] = None):
        """Repopulates session list if change is for current project."""
        current_project = self._project_manager.get_current_project() if self._project_manager else None  # type: ignore
        if current_project and current_project.id == project_id:
            logger.debug(f"LCP: Session list changed for current project {project_id}. Repopulating sessions.")
            self._update_sessions_list_for_project(project_id)
            if session_id and new_name is None:  # If it was a creation, select the new session
                self._select_session_in_list_widget(session_id)

    @Slot(str, str)  # project_id, session_id
    def _handle_session_switched_by_logic(self, project_id: str, session_id: str):
        logger.debug(f"LCP: Session switched by logic to S:{session_id} in P:{project_id}. Updating UI selection.")
        current_project = self._project_manager.get_current_project() if self._project_manager else None  # type: ignore
        if current_project and current_project.id == project_id:  # Ensure session switch is for current project
            self._select_session_in_list_widget(session_id)

    # --- Helper Methods for UI Updates ---
    def _get_chat_backend_ids_for_dropdown(self) -> List[str]:
        # Define which backends are suitable for general chat
        return [be_id for be_id in self.chat_manager.get_all_available_backend_ids() if "generator" not in be_id]

    def _get_specialized_backend_ids_for_dropdown(self) -> List[str]:
        # All backends can potentially be specialized
        return self.chat_manager.get_all_available_backend_ids()

    def _populate_llm_combobox(self, combo_box: QComboBox, backend_ids_to_show: List[str],
                               force_refresh_backend: Optional[str] = None):
        combo_box.clear()
        any_model_added = False
        for backend_id in backend_ids_to_show:
            display_details = self.BACKEND_DISPLAY_DETAILS.get(backend_id,
                                                               {"prefix": backend_id.replace("_default", "").title(),
                                                                "default_models": []})
            prefix = display_details["prefix"]

            # If forcing refresh for a specific backend, or if its models aren't cached/are stale.
            # The get_available_models_for_backend in BackendCoordinator now handles scheduling async refresh.
            available_models = self._backend_coordinator.get_available_models_for_backend(backend_id)  # type: ignore

            if not available_models:  # Use default models from display_details if API fetch yielded none
                available_models = display_details.get("default_models", [])

            if available_models:
                for model_name_str in available_models:
                    model_disp_name = model_name_str.replace("models/",
                                                             "") if "gemini" in backend_id else model_name_str
                    combo_box.addItem(f"{prefix}: {model_disp_name}",
                                      userData={"backend_id": backend_id, "model_name": model_name_str})
                    any_model_added = True
            else:  # Still no models, show placeholder
                is_conf = self._backend_coordinator.is_backend_configured(backend_id)  # type: ignore
                last_err = self._backend_coordinator.get_last_error_for_backend(backend_id)  # type: ignore
                ph_text = f"[{prefix}: No models]"
                if not is_conf:
                    ph_text = f"[{prefix}: Not Configured]"
                elif last_err:
                    ph_text = f"[{prefix}: Error]"
                combo_box.addItem(ph_text,
                                  userData={"backend_id": backend_id, "model_name": None, "is_placeholder": True})
                any_model_added = True  # Placeholder is an item

        if not any_model_added:
            combo_box.addItem("No LLMs Available", userData={"is_placeholder": True}); combo_box.setEnabled(False)
        else:
            combo_box.setEnabled(True)

    def _set_combobox_selection(self, combo_box: QComboBox, target_backend_id: Optional[str],
                                target_model_name: Optional[str]):
        if not target_backend_id or not target_model_name:  # If target is not fully specified, try to select first valid
            if combo_box.count() > 0:
                first_valid_idx = -1
                for i in range(combo_box.count()):
                    if not combo_box.itemData(i).get("is_placeholder"): first_valid_idx = i; break
                if first_valid_idx != -1:
                    combo_box.setCurrentIndex(first_valid_idx)
                elif combo_box.count() > 0:
                    combo_box.setCurrentIndex(0)  # Select placeholder if nothing else
            return

        for i in range(combo_box.count()):
            item_data = combo_box.itemData(i)
            if isinstance(item_data, dict) and item_data.get("backend_id") == target_backend_id and item_data.get(
                    "model_name") == target_model_name:
                if combo_box.currentIndex() != i: combo_box.setCurrentIndex(i)
                return
        # If exact match not found, try to select first model of the target backend
        for i in range(combo_box.count()):
            item_data = combo_box.itemData(i)
            if isinstance(item_data, dict) and item_data.get("backend_id") == target_backend_id and not item_data.get(
                    "is_placeholder"):
                if combo_box.currentIndex() != i: combo_box.setCurrentIndex(i)
                # Emit change if this auto-selection changed the model for the backend
                if item_data.get("model_name") != target_model_name:
                    if combo_box == self.chat_llm_combo_box:
                        self._event_bus.chatLlmSelectionChanged.emit(target_backend_id, item_data.get("model_name"))
                    elif combo_box == self.specialized_llm_combo_box:
                        self._event_bus.specializedLlmSelectionChanged.emit(target_backend_id,
                                                                            item_data.get("model_name"))
                return
        # Fallback if no match at all
        if combo_box.count() > 0 and combo_box.currentIndex() == -1: combo_box.setCurrentIndex(0)

    def _select_project_in_list_widget(self, project_id: str):
        self._is_programmatic_list_selection_change = True
        for i in range(self.projects_list_widget.count()):
            item = self.projects_list_widget.item(i)
            if item and item.data(self.PROJECT_ID_ROLE) == project_id:
                if self.projects_list_widget.currentItem() != item: self.projects_list_widget.setCurrentItem(item)
                break
        self._is_programmatic_list_selection_change = False

    def _select_session_in_list_widget(self, session_id: str):
        self._is_programmatic_list_selection_change = True
        for i in range(self.sessions_list_widget.count()):
            item = self.sessions_list_widget.item(i)
            if item and item.data(self.SESSION_ID_ROLE) == session_id:
                if self.sessions_list_widget.currentItem() != item: self.sessions_list_widget.setCurrentItem(item)
                break
        self._is_programmatic_list_selection_change = False

    def _update_sessions_list_for_project(self, project_id: str):
        logger.debug(f"LCP: Updating sessions list UI for project ID: {project_id}")
        self._is_programmatic_list_selection_change = True
        self.sessions_list_widget.clear()
        if self._project_manager:
            sessions = self._project_manager.get_project_sessions(project_id)
            for session in sessions:
                item = QListWidgetItem(session.name)
                item.setData(self.SESSION_ID_ROLE, session.id)
                self.sessions_list_widget.addItem(item)

            current_pm_session = self._project_manager.get_current_session()
            if current_pm_session and current_pm_session.project_id == project_id:
                self._select_session_in_list_widget(current_pm_session.id)
            elif sessions:  # If no current session set in PM for this project, select the first one
                self._select_session_in_list_widget(sessions[0].id)
                # Optionally, tell PM to make this the current session for the project
                # self._project_manager.switch_to_session(sessions[0].id) # Be careful of signal loops
        self._is_programmatic_list_selection_change = False

    def update_personality_button_tooltip(self):
        current_persona = self.chat_manager.get_current_chat_personality()  # From BackendConfigManager
        tooltip_base = "Customize chat AI personality / system prompt (Ctrl+P)."
        status = "(Custom Persona Active)" if current_persona and current_persona != "You are Ava, a bubbly, enthusiastic, and incredibly helpful AI assistant!" else "(Default Persona)"
        self.configure_ai_personality_button.setToolTip(f"{tooltip_base}\nStatus: {status}")

    def _check_initial_rag_status(self):
        # Called once at init to get initial RAG status
        # ChatManager._check_rag_readiness_and_emit_status will do the actual check and emit
        # This just ensures it's called after UI is ready.
        # No, ChatManager.initialize() should call it. This method is redundant here.
        pass

    def set_panel_enabled_state(self, panel_is_enabled: bool):
        """Enables or disables interactive elements on the panel."""
        # This is a general enable/disable, typically based on overall app busy state.
        # Individual button states (like RAG buttons based on RAG readiness) are handled separately.
        self.llm_config_group.setEnabled(panel_is_enabled)
        self.actions_group.setEnabled(panel_is_enabled)
        self.projects_group.setEnabled(panel_is_enabled)
        # RAG group might depend on both panel_is_enabled AND RAG readiness.
        # Let _update_rag_status_display handle specific RAG button states.
        self.rag_group.setEnabled(panel_is_enabled)

        # Re-evaluate specific button states within groups if panel is enabled
        if panel_is_enabled:
            is_project_active = self._project_manager and self._project_manager.get_current_project() is not None  # type: ignore
            self.new_chat_button.setEnabled(is_project_active)
            self.sessions_list_widget.setEnabled(is_project_active)

            # RAG button states depend on RAG readiness, handled by _update_rag_status_display
            # For now, just ensure they are not disabled if panel is enabled and RAG is ready
            if self.chat_manager and self.chat_manager.is_rag_ready():
                self.scan_global_rag_directory_button.setEnabled(True)
                self.add_project_files_button.setEnabled(is_project_active)
            else:
                self.scan_global_rag_directory_button.setEnabled(False)
                self.add_project_files_button.setEnabled(False)
        else:  # Panel disabled, disable everything
            self.new_chat_button.setEnabled(False)
            self.sessions_list_widget.setEnabled(False)
            self.scan_global_rag_directory_button.setEnabled(False)
            self.add_project_files_button.setEnabled(False)