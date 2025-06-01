# app/core/plan_and_code_coordinator.py
import logging
import uuid
import asyncio
import os
import re
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import ast  # For parsing plan parts like lists

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    from app.models.chat_message import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE  # Corrected path
    from app.models.message_enums import MessageLoadingState  # Corrected path
    from app.llm.backend_coordinator import BackendCoordinator  # Corrected path
    from app.services.llm_communication_logger import LlmCommunicationLogger  # Corrected path
    from app.core.code_output_processor import CodeOutputProcessor, CodeQualityLevel  # Corrected path
    # Prompts are now in app.llm.prompts
    from app.llm import prompts as llm_prompts
    from utils import constants
except ImportError as e_pacc:
    logging.getLogger(__name__).critical(f"PlanAndCodeCoordinator: Critical import error: {e_pacc}", exc_info=True)
    # Define fallback types for type hinting
    EventBus = type("EventBus", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MessageLoadingState = type("MessageLoadingState", (object,), {})  # type: ignore
    BackendCoordinator = type("BackendCoordinator", (object,), {})
    LlmCommunicationLogger = type("LlmCommunicationLogger", (object,), {})  # type: ignore
    CodeOutputProcessor = type("CodeOutputProcessor", (object,), {})  # type: ignore
    CodeQualityLevel = type("CodeQualityLevel", (object,), {})  # type: ignore
    llm_prompts = type("llm_prompts", (object,), {})  # type: ignore
    constants = type("constants", (object,), {})  # type: ignore
    USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "model", "system", "error"
    raise

logger = logging.getLogger(__name__)


class SequencePhase(Enum):
    IDLE = auto()
    PLANNING = auto()
    DEPENDENCY_ANALYSIS = auto()  # Optional intermediate step
    AWAITING_PLAN_CONFIRMATION = auto()
    CODE_GENERATION = auto()
    VALIDATION_AND_REFINEMENT = auto()  # Optional self-critique/refinement loop
    FINALIZATION = auto()
    ERROR = auto()


class GenerationStrategy(Enum):
    SEQUENTIAL_STRICT = auto()  # One file at a time, strictly following defined order
    BATCHED_BY_DEPENDENCY = auto()  # Small batches based on dependency levels
    # PARALLEL_ALL = auto()  # All at once (more risky, might skip for now)


@dataclass
class FileGenerationTask:
    filename: str
    instructions: str  # Detailed instructions for this specific file
    task_type: str  # e.g., 'api', 'ui', 'data_processing', 'utility', 'core'
    dependencies: List[str] = field(default_factory=list)  # List of other filenames this file depends on
    dependents: List[str] = field(default_factory=list)  # List of other filenames that depend on this one
    generation_order: int = 0  # Determined by dependency analysis
    purpose: str = ""  # Short description of the file's role
    file_type: str = "general"  # More specific type, e.g. 'model', 'controller', 'view'
    priority: int = 1  # Lower number means higher priority within a batch

    # Generation state
    request_id: Optional[str] = None  # LLM request ID for generating this file
    generated_code: Optional[str] = None
    code_quality: Optional[CodeQualityLevel] = None
    processing_notes: List[str] = field(default_factory=list)  # Notes from CodeOutputProcessor
    validation_passed: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2  # Max retries for generation (1 initial + 2 retries)


@dataclass
class GenerationBatch:
    """A batch of files that can be generated concurrently (no direct dependencies within the batch)."""
    files: List[FileGenerationTask]
    batch_number: int
    completed_count: int = 0
    failed_count: int = 0
    is_active: bool = False


class PlanAndCodeCoordinator(QObject):
    """
    Coordinates the multi-step process of planning a multi-file project
    and then generating code for each file based on the plan.
    Uses an "enhanced quality" approach with potential for self-critique and retries.
    """

    def __init__(self,
                 backend_coordinator: BackendCoordinator,
                 event_bus: EventBus,
                 llm_comm_logger: Optional[LlmCommunicationLogger],  # type: ignore
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(backend_coordinator, BackendCoordinator):  # type: ignore
            raise TypeError("PlanAndCodeCoordinator requires a valid BackendCoordinator.")
        if not isinstance(event_bus, EventBus):  # type: ignore
            raise TypeError("PlanAndCodeCoordinator requires a valid EventBus.")

        self._backend_coordinator = backend_coordinator
        self._event_bus = event_bus
        self._llm_comm_logger = llm_comm_logger
        self._code_processor = CodeOutputProcessor()  # type: ignore

        # Sequence State
        self._current_phase = SequencePhase.IDLE
        self._sequence_id: Optional[str] = None
        self._original_query: Optional[str] = None
        self._project_context: Dict[str, Any] = {}  # Stores project_dir, project_id, session_id, model choices etc.
        self._plan_text: Optional[str] = None  # Raw text of the generated plan
        self._file_tasks: List[FileGenerationTask] = []
        self._generation_batches: List[GenerationBatch] = []
        self._current_batch_index: int = 0
        self._generated_files_context: Dict[str, str] = {}  # filename -> generated_code_summary or full_code
        self._active_generation_tasks: Dict[str, FileGenerationTask] = {}  # request_id -> FileGenerationTask

        # Configuration
        self._generation_strategy = GenerationStrategy.BATCHED_BY_DEPENDENCY
        self._max_concurrent_generations = 2  # Max files to generate in parallel within a batch
        self._max_retries_per_file = 1  # Initial attempt + 1 retry with self-critique
        self._enable_self_critique = True

        self._connect_event_bus_handlers()
        self._event_bus.forcePlanAndCodeGenerationRequested.connect(self._handle_force_proceed_to_generation)

        logger.info("PlanAndCodeCoordinator initialized with enhanced quality focus.")

    def _connect_event_bus_handlers(self):
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_completion)
        self._event_bus.llmResponseError.connect(self._handle_llm_error)

    def start_autonomous_coding(self,
                                user_query: str,
                                planner_backend: str,
                                planner_model: str,
                                coder_backend: str,
                                coder_model: str,
                                project_dir: str,  # Absolute path to project's root directory
                                project_id: Optional[str] = None,  # Associated project ID from ProjectManager
                                session_id: Optional[str] = None,  # UI session ID for updates
                                task_type: Optional[str] = None  # e.g. 'api', 'ui', 'data_processing'
                                ) -> bool:
        """Initiates the autonomous coding sequence: Plan -> [Confirm] -> Code."""
        if self._current_phase != SequencePhase.IDLE:
            logger.warning(
                f"PACC: Sequence '{self._sequence_id}' already active in phase {self._current_phase.name}. Ignoring new request for '{user_query[:30]}...'.")
            self._emit_status_update("Autonomous coding sequence already in progress.", "#e5c07b", True, 3000)
            return False

        self._reset_sequence_state()  # Ensure clean state
        self._sequence_id = f"pacc_seq_{uuid.uuid4().hex[:8]}"
        self._original_query = user_query
        self._project_context = {
            'project_dir': project_dir,
            'project_id': project_id,
            'session_id': session_id,
            'task_type': task_type or 'general',
            'planner_backend': planner_backend,
            'planner_model': planner_model,
            'coder_backend': coder_backend,
            'coder_model': coder_model
        }

        logger.info(
            f"PACC ({self._sequence_id}): Starting sequence for query: '{user_query[:50]}...'. Project Dir: {project_dir}")
        self._log_communication("SEQ_START", f"Query: {user_query[:100]}... Project Dir: {project_dir}")
        self._emit_chat_message_to_ui(
            f"[System: Starting autonomous coding for '{user_query[:40]}...'. Planning project structure...]")
        self._event_bus.uiInputBarBusyStateChanged.emit(True)  # Block user input

        self._current_phase = SequencePhase.PLANNING
        return self._initiate_planning_phase()

    def _initiate_planning_phase(self) -> bool:
        """Sends a request to the planning LLM to generate the project plan."""
        if not self._backend_coordinator or not self._sequence_id or not self._original_query: return False

        planning_request_id = f"plan_{self._sequence_id}"
        self._active_generation_tasks[planning_request_id] = FileGenerationTask(filename="<PLANNING_PHASE>",
                                                                                instructions="",
                                                                                task_type="planning")  # Dummy task for tracking

        self._emit_status_update(f"Planning project with {self._project_context['planner_model']}...", "#61afef", False)

        planning_prompt_text = self._construct_planning_prompt()
        history = [ChatMessage(role=USER_ROLE, parts=[planning_prompt_text])]  # type: ignore

        self._backend_coordinator.start_llm_streaming_task(
            request_id=planning_request_id,
            target_backend_id=self._project_context['planner_backend'],
            history_to_send=history,
            is_modification_response_expected=True,  # The plan is a modification of the prompt
            options={"temperature": 0.15},  # Low temperature for structured planning
            request_metadata={
                "purpose": "autonomous_planning",
                "sequence_id": self._sequence_id,
                "project_id": self._project_context.get('project_id'),
                "session_id": self._project_context.get('session_id')
            }
        )
        logger.info(f"PACC ({self._sequence_id}): Planning request ({planning_request_id}) sent.")
        return True

    def _construct_planning_prompt(self) -> str:
        """Constructs the detailed prompt for the planning phase."""
        # Use the enhanced planning prompt from llm_prompts
        base_prompt = getattr(llm_prompts, 'ENHANCED_PLANNING_SYSTEM_PROMPT',
                              "Error: ENHANCED_PLANNING_SYSTEM_PROMPT not found.")

        # Gather context about existing files if project_dir exists
        existing_structure_info = self._analyze_existing_project_structure()
        task_specific_guidance = self._get_task_specific_planning_guidance()

        context_section = (
            f"\n\n## Project Context & Constraints\n"
            f"**User's Core Request**: {self._original_query}\n"
            f"**Target Project Directory**: {self._project_context.get('project_dir', 'N/A')}\n"
            f"**Primary Task Type**: {self._project_context.get('task_type', 'general')}\n"
            f"**Selected Coder LLM**: {self._project_context.get('coder_model', 'Default Coder')}\n"
            f"**Existing Project Structure (if any, summarize key files/folders)**:\n{existing_structure_info}\n"
            f"{task_specific_guidance}\n"
            f"Focus on creating a robust, well-structured plan that anticipates dependencies and ensures high-quality code generation. "
            f"The generated code will be implemented by '{self._project_context.get('coder_model', 'Default Coder')}' which excels at Python."
        )
        return base_prompt + context_section

    def _analyze_existing_project_structure(self) -> str:
        project_dir = self._project_context.get('project_dir')
        if not project_dir or not os.path.isdir(project_dir):
            return "This is a new project, no existing files to consider beyond the user's request."

        file_tree_lines = []
        max_files_to_list = 20  # Limit how many files we list to keep prompt size manageable
        files_listed_count = 0

        for root, dirs, files in os.walk(project_dir):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if
                       d.lower() not in constants.DEFAULT_IGNORED_DIRS and not d.startswith('.')]  # type: ignore

            level = root.replace(project_dir, '').count(os.sep)
            if level > 3:  # Limit depth
                dirs[:] = []  # Don't go deeper
                continue

            indent = '  ' * level
            file_tree_lines.append(f"{indent}{os.path.basename(root)}/")

            for file_name in files:
                if files_listed_count >= max_files_to_list: break
                file_ext = os.path.splitext(file_name)[1].lower()
                if file_ext in constants.ALLOWED_TEXT_EXTENSIONS:  # type: ignore
                    file_tree_lines.append(f"{indent}  {file_name}")
                    files_listed_count += 1
            if files_listed_count >= max_files_to_list: break

        if not file_tree_lines: return "No relevant existing files found in the project directory."
        return "Brief overview of existing relevant files:\n" + "\n".join(file_tree_lines)

    def _get_task_specific_planning_guidance(self) -> str:
        task_type = self._project_context.get('task_type', 'general')
        # These could also be moved to llm_prompts.py if they become very large
        guidance = {
            'api': "For API projects, prioritize clear endpoints, request/response models (Pydantic), database interaction modules, and a main FastAPI/Flask app file. Ensure authentication and error handling are planned.",
            'ui': "For UI projects (PySide6), plan for main window, custom widgets, dialogs, models/controllers, and potentially a styling/theme file. Event handling and state management are key.",
            'data_processing': "For data processing, plan for data loading, validation, transformation steps, output modules, and a main orchestration script. Consider error logging and data quality checks.",
            'utility': "For utility libraries, focus on well-defined functions/classes, clear API contracts, configuration, and robust error handling. Ensure minimal external dependencies unless specified.",
            'general': "For general projects, ensure a logical separation of concerns. Common modules include config, utils, core logic, and a main entry point. Plan for clear interfaces between modules."
        }
        return f"\n**Guidance for Task Type '{task_type.upper()}'**:\n{guidance.get(task_type, guidance['general'])}"

    @Slot(str)
    def _handle_force_proceed_to_generation(self):
        logger.info(
            f"PACC ({self._sequence_id}): Force proceed to code generation requested by user. Current phase: {self._current_phase.name}")
        if self._current_phase == SequencePhase.AWAITING_PLAN_CONFIRMATION:
            if self._file_tasks and self._generation_batches:
                self.confirm_plan_and_proceed_with_generation()
            else:
                logger.error(
                    f"PACC ({self._sequence_id}): Cannot force proceed from AWAITING_PLAN_CONFIRMATION as plan/tasks are not ready.")
                self._emit_chat_message_to_ui(
                    "[System Error: Plan data is missing, cannot force code generation. Please restart the process.]",
                    is_error=True)
                self._reset_sequence_state()
        elif self._current_phase == SequencePhase.PLANNING:
            logger.warning(
                f"PACC ({self._sequence_id}): Forcing generation from PLANNING phase. This might use an incomplete or default plan.")
            # Attempt to parse whatever plan_text we might have, or create a fallback
            if self._plan_text:
                self._process_valid_plan(self._plan_text)  # Try to use partially received plan
            if not self._file_tasks:  # If plan parsing failed or no plan_text yet
                self._create_emergency_fallback_plan()

            if self._file_tasks:
                self.confirm_plan_and_proceed_with_generation()
            else:
                self._emit_chat_message_to_ui("[System Error: Could not create even a fallback plan. Cannot proceed.]",
                                              is_error=True)
                self._reset_sequence_state()
        else:
            logger.info(
                f"PACC ({self._sequence_id}): Force proceed not applicable for phase {self._current_phase.name}.")
            self._emit_status_update(f"Cannot force generation from current phase: {self._current_phase.name}",
                                     "#e06c75", True, 3000)

    def _create_emergency_fallback_plan(self):
        """Creates a very basic plan if forced to proceed without a proper LLM plan."""
        logger.warning(
            f"PACC ({self._sequence_id}): Creating emergency fallback plan for query: '{self._original_query[:50]}...'")
        filename = "main.py"  # Default filename
        # Try to infer a better filename
        if self._original_query:
            query_words = self._original_query.lower().split()
            if "api" in query_words:
                filename = "api_server.py"
            elif "script" in query_words:
                filename = "script.py"
            elif "tool" in query_words:
                filename = "utility_tool.py"

        instructions = (
            f"Create a complete Python implementation for the user's request: '{self._original_query}'.\n"
            f"The file should be named '{filename}'.\n"
            "Ensure the code is production-ready, includes error handling, logging, type hints, and comprehensive docstrings.\n"
            "Generate all necessary imports and structure the code logically."
        )
        fallback_task = FileGenerationTask(
            filename=filename,
            instructions=instructions,
            task_type=self._project_context.get('task_type', 'general'),
            purpose=f"Core implementation for: {self._original_query[:60]}...",
            generation_order=0
        )
        self._file_tasks = [fallback_task]
        self._build_generation_batches()
        self._plan_text = f"## Emergency Fallback Plan\nFILES_LIST: ['{filename}']\nGENERATION_ORDER: ['{filename}']\n### {filename}\nPURPOSE: {fallback_task.purpose}\nDETAILED_REQUIREMENTS:\n- {instructions}"
        logger.info(f"PACC ({self._sequence_id}): Emergency fallback plan created with 1 task: {filename}")
        self._emit_chat_message_to_ui(f"[System: Using fallback plan to generate {filename} based on your request.]")

    @Slot(str, ChatMessage, dict)  # type: ignore
    def _handle_llm_completion(self, request_id: str, message: ChatMessage, metadata: dict):  # type: ignore
        if not self._sequence_id or metadata.get("sequence_id") != self._sequence_id:
            return  # Not for the current active sequence

        purpose = metadata.get("purpose")

        # Remove from active tasks now that we have a response (success or error)
        task_details = self._active_generation_tasks.pop(request_id, None)

        if purpose == "autonomous_planning":
            if task_details:  # Should be the dummy planning task
                logger.info(f"PACC ({self._sequence_id}): Received plan from LLM (ReqID: {request_id}).")
                self._process_valid_plan(message.text)  # type: ignore
            else:
                logger.error(
                    f"PACC ({self._sequence_id}): Received planning completion for unknown request ID {request_id}.")

        elif purpose == "autonomous_coding":
            if task_details:  # Should be a FileGenerationTask
                filename = metadata.get("filename", task_details.filename)  # Prefer metadata if available
                logger.info(
                    f"PACC ({self._sequence_id}): Received generated code for '{filename}' (ReqID: {request_id}).")
                self._handle_generated_code_for_file(task_details, message.text)  # type: ignore
            else:
                logger.error(
                    f"PACC ({self._sequence_id}): Received code generation for unknown request ID {request_id} or task details missing.")
        else:
            logger.debug(
                f"PACC ({self._sequence_id}): Received LLM completion for unhandled purpose '{purpose}' (ReqID: {request_id}).")

    def _process_valid_plan(self, plan_text: str):
        """Parses the plan text, sets up tasks, and transitions to confirmation or coding."""
        self._plan_text = plan_text
        try:
            self._file_tasks = self._parse_llm_plan_to_tasks(plan_text)
            if not self._file_tasks:
                raise ValueError("Plan parsing resulted in no file tasks.")

            self._build_generation_batches()  # Based on dependencies and strategy
            if not self._generation_batches:
                raise ValueError("Could not form generation batches from file tasks.")

            self._current_phase = SequencePhase.AWAITING_PLAN_CONFIRMATION
            plan_summary = self._create_plan_summary_for_ui()
            self._emit_chat_message_to_ui(
                f"[System: Project Plan Created]\nI've drafted the following plan:\n{plan_summary}\n\n"
                f"Review the plan. Type 'yes', 'proceed', or 'generate files' to start coding, or provide feedback for revisions."
            )
            self._emit_status_update(f"Plan ready ({len(self._file_tasks)} files). Confirm to generate.", "#e5c07b",
                                     False)
            self._event_bus.uiInputBarBusyStateChanged.emit(False)  # Allow user to confirm/give feedback
            logger.info(
                f"PACC ({self._sequence_id}): Plan processed. {len(self._file_tasks)} tasks in {len(self._generation_batches)} batches. Awaiting confirmation.")

        except ValueError as e_parse:
            logger.error(f"PACC ({self._sequence_id}): Failed to parse LLM plan: {e_parse}", exc_info=True)
            self._emit_chat_message_to_ui(
                f"[System Error: The generated plan was malformed and could not be understood. Error: {e_parse}. Please try rephrasing your request or try again.]",
                is_error=True)
            self._reset_sequence_state(error_occurred=True)
        except Exception as e_plan_proc:
            logger.error(f"PACC ({self._sequence_id}): Unexpected error processing plan: {e_plan_proc}", exc_info=True)
            self._emit_chat_message_to_ui(
                f"[System Error: An unexpected error occurred while processing the plan. Details: {e_plan_proc}]",
                is_error=True)
            self._reset_sequence_state(error_occurred=True)

    def confirm_plan_and_proceed_with_generation(self):
        """Called by ChatManager when user confirms the plan."""
        if self._current_phase != SequencePhase.AWAITING_PLAN_CONFIRMATION:
            logger.warning(
                f"PACC ({self._sequence_id}): Plan confirmation received but not in AWAITING_PLAN_CONFIRMATION phase (current: {self._current_phase.name}).")
            return

        if not self._file_tasks or not self._generation_batches:
            logger.error(
                f"PACC ({self._sequence_id}): Plan confirmed, but file tasks or batches are empty. Aborting generation.")
            self._emit_chat_message_to_ui("[System Error: Plan data is missing. Cannot proceed with code generation.]",
                                          is_error=True)
            self._reset_sequence_state(error_occurred=True)
            return

        logger.info(f"PACC ({self._sequence_id}): User confirmed plan. Proceeding to code generation.")
        self._emit_chat_message_to_ui(
            "[System: Plan confirmed! Starting code generation... This may take some time. 🚀]")
        self._event_bus.uiInputBarBusyStateChanged.emit(True)
        self._current_phase = SequencePhase.CODE_GENERATION
        self._current_batch_index = 0  # Start from the first batch
        asyncio.create_task(self._process_current_generation_batch())

    def _parse_llm_plan_to_tasks(self, plan_text: str) -> List[FileGenerationTask]:
        """ Parses the structured plan text from LLM into FileGenerationTask objects."""
        # This parsing logic needs to be robust to handle variations in LLM output
        # while strictly expecting the defined format (FILES_LIST, GENERATION_ORDER, ### filename sections)
        tasks_dict: Dict[str, FileGenerationTask] = {}

        files_list_match = re.search(r"FILES_LIST:\s*(\[.*?\])", plan_text, re.DOTALL | re.IGNORECASE)
        generation_order_match = re.search(r"GENERATION_ORDER:\s*(\[.*?\])", plan_text, re.DOTALL | re.IGNORECASE)

        if not files_list_match:
            raise ValueError("Plan missing mandatory 'FILES_LIST: [...]' section.")

        try:
            files_list_str = files_list_match.group(1)
            # Use ast.literal_eval for safe evaluation of Python list string
            parsed_files_list = ast.literal_eval(files_list_str)
            if not isinstance(parsed_files_list, list) or not all(isinstance(f, str) for f in parsed_files_list):
                raise ValueError(f"FILES_LIST format error. Expected list of strings, got: {files_list_str}")
        except (ValueError, SyntaxError) as e_eval:
            raise ValueError(f"Error parsing FILES_LIST: {e_eval}. Content: '{files_list_match.group(1)}'")

        parsed_generation_order: Optional[List[str]] = None
        if generation_order_match:
            try:
                order_list_str = generation_order_match.group(1)
                parsed_generation_order = ast.literal_eval(order_list_str)
                if not isinstance(parsed_generation_order, list) or not all(
                        isinstance(f, str) for f in parsed_generation_order):
                    logger.warning(
                        f"GENERATION_ORDER format error. Expected list of strings, got: {order_list_str}. Will attempt to infer.")
                    parsed_generation_order = None
            except (ValueError, SyntaxError) as e_eval_order:
                logger.warning(
                    f"Error parsing GENERATION_ORDER: {e_eval_order}. Content: '{generation_order_match.group(1)}'. Will attempt to infer.")
                parsed_generation_order = None

        if parsed_generation_order and set(parsed_files_list) != set(parsed_generation_order):
            logger.warning(
                "Mismatch between FILES_LIST and GENERATION_ORDER. Using GENERATION_ORDER for sequence, but all files from FILES_LIST will be processed.")
            # Ensure all files from files_list are considered, even if not in order list
            all_files_to_process = list(dict.fromkeys(
                parsed_generation_order + [f for f in parsed_files_list if f not in parsed_generation_order]))
        elif parsed_generation_order:
            all_files_to_process = parsed_generation_order
        else:  # No valid generation order, use files_list and infer order later if needed
            all_files_to_process = parsed_files_list

        # Extract details for each file
        file_sections = re.split(r'\n###\s+', '\n' + plan_text)[1:]  # Split by "### " but keep filename part

        for filename in all_files_to_process:
            found_section = False
            for section in file_sections:
                if section.lower().startswith(filename.lower()):
                    section_content = section[len(filename):].strip()  # Remove filename part

                    purpose = self._extract_detail_from_section(section_content,
                                                                "PURPOSE") or f"Implementation for {filename}"
                    dependencies_str = self._extract_detail_from_section(section_content, "DEPENDENCIES", r"\[.*?\]")
                    dependents_str = self._extract_detail_from_section(section_content, "DEPENDENTS", r"\[.*?\]")
                    priority_str = self._extract_detail_from_section(section_content, "PRIORITY", r"\d+")
                    file_type = self._extract_detail_from_section(section_content,
                                                                  "TYPE") or self._infer_task_type_from_filename(
                        filename)
                    core_components = self._extract_multiline_detail_from_section(section_content, "CORE_COMPONENTS")
                    detailed_reqs = self._extract_multiline_detail_from_section(section_content,
                                                                                "DETAILED_REQUIREMENTS")
                    api_contract = self._extract_code_block_from_section(section_content, "API_CONTRACT")
                    integration_notes = self._extract_multiline_detail_from_section(section_content,
                                                                                    "INTEGRATION_NOTES")

                    dependencies = self._parse_list_string(dependencies_str) if dependencies_str else []
                    dependents = self._parse_list_string(dependents_str) if dependents_str else []
                    priority = int(priority_str) if priority_str and priority_str.isdigit() else 99  # Default priority

                    # Construct comprehensive instructions for the coder LLM
                    instructions = (
                        f"**File**: {filename}\n"
                        f"**Purpose**: {purpose}\n"
                        f"**Type**: {file_type}\n"
                        f"**Dependencies**: {', '.join(dependencies) or 'None'}\n"
                        f"**Core Components to Implement**:\n{core_components or 'As per detailed requirements.'}\n"
                        f"**Detailed Requirements**:\n{detailed_reqs or 'Implement based on overall project goal and purpose.'}\n"
                    )
                    if api_contract: instructions += f"\n**API Contract (interfaces this file must expose)**:\n```python\n{api_contract}\n```\n"
                    if integration_notes: instructions += f"\n**Integration Notes**:\n{integration_notes}\n"

                    tasks_dict[filename] = FileGenerationTask(
                        filename=filename,
                        instructions=instructions,
                        task_type=file_type,  # Using file_type as task_type here
                        dependencies=dependencies,
                        dependents=dependents,  # Will be populated later if not in plan
                        generation_order=all_files_to_process.index(
                            filename) if filename in all_files_to_process else 999,
                        purpose=purpose,
                        file_type=file_type,
                        priority=priority
                    )
                    found_section = True
                    break
            if not found_section:
                logger.warning(
                    f"PACC ({self._sequence_id}): Details section for file '{filename}' not found in plan. Creating with default instructions.")
                tasks_dict[filename] = FileGenerationTask(
                    filename=filename,
                    instructions=f"Generate the complete Python code for the file '{filename}' as part of the project: {self._original_query}. "
                                 f"Ensure it integrates with other planned files and meets high quality standards.",
                    task_type=self._infer_task_type_from_filename(filename),
                    generation_order=all_files_to_process.index(filename) if filename in all_files_to_process else 999,
                    purpose=f"Core functionality for {filename}",
                    file_type=self._infer_task_type_from_filename(filename)
                )

        # Post-process to fill in dependents if not explicitly provided by LLM
        all_task_filenames = set(tasks_dict.keys())
        for task_filename, task in tasks_dict.items():
            for dep_filename in task.dependencies:
                if dep_filename in tasks_dict and task_filename not in tasks_dict[dep_filename].dependents:
                    tasks_dict[dep_filename].dependents.append(task_filename)
            # Filter out dependencies not in the tasks_dict (e.g. external libraries)
            task.dependencies = [dep for dep in task.dependencies if dep in all_task_filenames]

        # Sort tasks by generation_order
        final_tasks_list = sorted(list(tasks_dict.values()), key=lambda t: t.generation_order)
        # Re-assign generation_order based on the final sorted list for consistency
        for i, task in enumerate(final_tasks_list):
            task.generation_order = i

        if not final_tasks_list:
            raise ValueError("No valid file tasks could be parsed from the plan.")
        return final_tasks_list

    def _extract_detail_from_section(self, section_content: str, detail_key: str,
                                     pattern_override: Optional[str] = None) -> Optional[str]:
        """Helper to extract a single-line detail from a file section."""
        # Match "KEY: value" or "KEY**: value" (markdown bold)
        pattern = re.compile(rf"^\*\*{re.escape(detail_key)}\*\*:\s*(.+)$| ^{re.escape(detail_key)}:\s*(.+)$",
                             re.MULTILINE | re.IGNORECASE)
        if pattern_override:  # For lists or specific formats
            pattern_str = rf"^\*\*{re.escape(detail_key)}\*\*:\s*({pattern_override})$| ^{re.escape(detail_key)}:\s*({pattern_override})$"
            pattern = re.compile(pattern_str, re.MULTILINE | re.IGNORECASE | re.DOTALL)

        match = pattern.search(section_content)
        if match:
            # Match.groups() will return (value_from_bold, None) or (None, value_from_normal)
            # Or for pattern_override, it might be (value_bold_override, None, None, None) or (None, None, value_normal_override, None)
            # We need to find the first non-None group that captured content.
            for group in match.groups():
                if group is not None:
                    return group.strip()
        return None

    def _extract_multiline_detail_from_section(self, section_content: str, detail_key: str) -> Optional[str]:
        """Helper to extract a multi-line detail (e.g., requirements list) from a file section."""
        # Pattern looks for "KEY:" or "**KEY**:" then captures everything until the next key (e.g., "**OTHER_KEY**:"),
        # or a double newline, or end of section.
        pattern_str = rf"^\*\*{re.escape(detail_key)}\*\*:\s*\n(.*?)(?=\n\s*(\*\*[\w\s]+?\*\*:\s*|$))| ^{re.escape(detail_key)}:\s*\n(.*?)(?=\n\s*(\*\*[\w\s]+?\*\*:\s*|$))"
        pattern = re.compile(pattern_str, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        match = pattern.search(section_content)
        if match:
            # Groups would be (content_after_bold_key, next_bold_key_or_eos, content_after_normal_key, next_normal_key_or_eos)
            content = match.group(1) if match.group(1) is not None else match.group(3)
            if content:
                # Clean up list items if they start with "- "
                lines = [line.strip() for line in content.strip().split('\n')]
                cleaned_lines = []
                for line in lines:
                    if line.startswith("- "):
                        cleaned_lines.append(line)
                    elif line and cleaned_lines and cleaned_lines[-1].startswith(
                            "- "):  # Continuation of previous list item
                        cleaned_lines[-1] += " " + line
                    elif line:  # Not a list item, but part of the section
                        cleaned_lines.append(line)
                return "\n".join(cleaned_lines)
        return None

    def _extract_code_block_from_section(self, section_content: str, detail_key: str) -> Optional[str]:
        """Helper to extract a fenced code block following a key, e.g., API_CONTRACT."""
        # Pattern looks for "KEY:" or "**KEY**:", then some optional text, then a code block.
        pattern_str = rf"^\*\*{re.escape(detail_key)}\*\*:\s*.*?\n```(?:python|py)?\s*\n(.*?)\n?\s*```| ^{re.escape(detail_key)}:\s*.*?\n```(?:python|py)?\s*\n(.*?)\n?\s*```"
        pattern = re.compile(pattern_str, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        match = pattern.search(section_content)
        if match:
            code_content = match.group(1) if match.group(1) is not None else match.group(2)
            return code_content.strip() if code_content else None
        return None

    def _parse_list_string(self, list_str: Optional[str]) -> List[str]:
        """Safely parses a string like "['item1', 'item2']" into a list of strings."""
        if not list_str: return []
        try:
            evaluated_list = ast.literal_eval(list_str)
            if isinstance(evaluated_list, list) and all(isinstance(item, str) for item in evaluated_list):
                return evaluated_list
            logger.warning(
                f"PACC: Parsed list string '{list_str}' but result was not a list of strings: {type(evaluated_list)}")
        except (ValueError, SyntaxError) as e:
            logger.warning(
                f"PACC: Could not parse list string '{list_str}' with ast.literal_eval: {e}. Attempting regex fallback.")
            # Regex fallback for simple cases if ast fails (e.g. slightly malformed string)
            items = re.findall(r"['\"]([^'\"]+)['\"]", list_str)
            if items: return items
            logger.error(f"PACC: Failed to parse list string '{list_str}' even with regex fallback.")
        return []

    def _infer_task_type_from_filename(self, filename: str) -> str:
        fn_lower = filename.lower()
        if "main" in fn_lower or "app" in fn_lower or "server" in fn_lower: return "core"
        if "util" in fn_lower or "helper" in fn_lower: return "utility"
        if "config" in fn_lower or "setting" in fn_lower: return "config"
        if "model" in fn_lower or "schema" in fn_lower or "database" in fn_lower: return "data"
        if "ui" in fn_lower or "view" in fn_lower or "widget" in fn_lower or "dialog" in fn_lower: return "ui"
        if "test" in fn_lower: return "test"
        return "general"

    def _create_plan_summary_for_ui(self) -> str:
        """Creates a concise summary of the plan for display to the user."""
        if not self._file_tasks: return "The plan is empty or could not be parsed."

        summary_lines = [f"Project plan involves {len(self._file_tasks)} file(s):"]
        for i, task in enumerate(self._file_tasks):
            # Using task.purpose if available and concise, else filename.
            purpose_preview = task.purpose if task.purpose and len(task.purpose) < 80 else task.filename
            if len(purpose_preview) > 80: purpose_preview = purpose_preview[:77] + "..."

            line = f"  {i + 1}. **{task.filename}**: {purpose_preview}"
            if task.dependencies:
                deps_preview = ", ".join(task.dependencies[:3])
                if len(task.dependencies) > 3: deps_preview += ", ..."
                line += f" (depends on: {deps_preview})"
            summary_lines.append(line)

        if len(summary_lines) > 10:  # If too many files, truncate summary
            summary_lines = summary_lines[:10] + ["  ... and more files."]

        return "\n".join(summary_lines)

    def _build_generation_batches(self):
        """
        Builds batches of files for generation based on dependencies and selected strategy.
        This version implements a topological sort based approach for batching.
        """
        self._generation_batches = []
        if not self._file_tasks:
            logger.warning(f"PACC ({self._sequence_id}): No file tasks available to build batches.")
            return

        # Create a graph representation: filename -> set of dependencies
        graph: Dict[str, Set[str]] = {task.filename: set(task.dependencies) for task in self._file_tasks}
        # In-degree: filename -> number of incoming edges (dependencies it has on other files in the plan)
        in_degree: Dict[str, int] = {filename: 0 for filename in graph}
        # Adjacency list for dependents: filename -> list of files that depend on it
        adj: Dict[str, List[str]] = {filename: [] for filename in graph}

        for filename, dependencies in graph.items():
            for dep in dependencies:
                if dep in graph:  # Ensure dependency is part of the current plan
                    adj[dep].append(filename)
                    in_degree[filename] += 1

        # Initialize queue with nodes having in-degree 0 (no local dependencies)
        queue: List[str] = [filename for filename, degree in in_degree.items() if degree == 0]

        batch_number = 0
        processed_files_count = 0

        while processed_files_count < len(self._file_tasks):
            if not queue:  # Should not happen in a valid DAG if not all files processed
                logger.error(
                    f"PACC ({self._sequence_id}): Dependency cycle detected or error in batching. Remaining files may not be processable.")
                # Add remaining unprocessed files to a final error batch if any
                remaining_unprocessed = [task for task in self._file_tasks if
                                         task.filename not in graph or in_degree.get(task.filename,
                                                                                     -1) != -100]  # -100 marks processed
                if remaining_unprocessed:
                    self._generation_batches.append(
                        GenerationBatch(files=remaining_unprocessed, batch_number=batch_number + 1))
                break

            current_batch_filenames = list(queue)  # Files in this batch can be processed concurrently
            queue = []  # Clear queue for next level
            batch_number += 1

            batch_tasks = [task for task in self._file_tasks if task.filename in current_batch_filenames]
            # Sort tasks within a batch by their original priority/order if needed, though concurrency implies less strict order within batch
            batch_tasks.sort(key=lambda t: t.priority if hasattr(t, 'priority') else t.generation_order)

            if batch_tasks:
                self._generation_batches.append(GenerationBatch(files=batch_tasks, batch_number=batch_number))
                processed_files_count += len(batch_tasks)

            for filename_done in current_batch_filenames:
                in_degree[filename_done] = -100  # Mark as processed to avoid cycles in error case
                for neighbor in adj.get(filename_done, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        logger.info(
            f"PACC ({self._sequence_id}): Built {len(self._generation_batches)} generation batches for {len(self._file_tasks)} files using topological sort.")

    async def _process_current_generation_batch(self):
        """Processes the current batch of files for code generation."""
        if self._current_batch_index >= len(self._generation_batches):
            self._finalize_sequence_if_all_done()
            return

        current_batch = self._generation_batches[self._current_batch_index]
        current_batch.is_active = True
        logger.info(
            f"PACC ({self._sequence_id}): Processing Batch {current_batch.batch_number}/{len(self._generation_batches)} with {len(current_batch.files)} file(s).")
        self._emit_status_update(
            f"Generating Batch {current_batch.batch_number}/{len(self._generation_batches)} ({len(current_batch.files)} files)...",
            "#61afef", False
        )

        # Generate files in the current batch concurrently (up to _max_concurrent_generations)
        # For simplicity, we'll use asyncio.gather for the whole batch here.
        # A more complex version could use a semaphore to limit true concurrency.
        generation_coroutines = [self._generate_code_for_single_file(task) for task in current_batch.files]

        # Wait for all tasks in the current batch to complete (or error out)
        # Results will contain the FileGenerationTask objects (or exceptions)
        await asyncio.gather(*generation_coroutines,
                             return_exceptions=True)  # Allow exceptions so one failure doesn't stop others in batch

        # After batch completion (successful or partial), check status
        current_batch.is_active = False
        all_successful_in_batch = all(task.generated_code and task.validation_passed for task in current_batch.files)

        if all_successful_in_batch:
            logger.info(f"PACC ({self._sequence_id}): Batch {current_batch.batch_number} completed successfully.")
            self._current_batch_index += 1
            asyncio.create_task(self._process_current_generation_batch())  # Move to next batch
        else:
            failed_tasks_in_batch = [task.filename for task in current_batch.files if
                                     not (task.generated_code and task.validation_passed)]
            logger.error(
                f"PACC ({self._sequence_id}): Batch {current_batch.batch_number} had failures for files: {', '.join(failed_tasks_in_batch)}. Sequence will halt.")
            self._emit_chat_message_to_ui(
                f"[System Error: Code generation failed for files in batch {current_batch.batch_number}: {', '.join(failed_tasks_in_batch)}. Cannot proceed automatically.]",
                is_error=True)
            self._reset_sequence_state(error_occurred=True,
                                       message=f"Generation failed for: {', '.join(failed_tasks_in_batch)}")
            # No automatic retry across batches for now, user might need to intervene or restart.

    async def _generate_code_for_single_file(self, task: FileGenerationTask):
        """Generates code for a single file, with retries and self-critique if enabled."""
        if not self._backend_coordinator or not self._sequence_id: return

        task.request_id = f"code_{self._sequence_id}_{task.filename.replace('.', '_').replace(os.sep, '_')}"
        self._active_generation_tasks[task.request_id] = task

        for attempt in range(task.max_retries + 1):
            task.retry_count = attempt
            logger.info(
                f"PACC ({self._sequence_id}): Generating '{task.filename}' (Attempt {attempt + 1}/{task.max_retries + 1}). Request ID: {task.request_id}")
            self._emit_chat_message_to_ui(f"[System: Generating code for {task.filename} (Attempt {attempt + 1})...]")

            current_instructions = task.instructions
            if attempt > 0 and self._enable_self_critique and task.generated_code and not task.validation_passed:
                # If retrying after a failed validation and self-critique is on
                critique_prompt = self._create_self_critique_prompt_for_task(task)
                current_instructions = critique_prompt  # Use critique prompt for retry
                logger.info(f"PACC ({self._sequence_id}): Applying self-critique for '{task.filename}'.")
                self._emit_chat_message_to_ui(
                    f"[System: Refining code for {task.filename} based on internal review...]")

            # Add context from already generated dependent files
            context_from_dependencies = self._gather_context_from_dependencies(task.dependencies)
            final_coding_prompt_text = self._construct_final_coding_prompt(task, current_instructions,
                                                                           context_from_dependencies)
            history = [ChatMessage(role=USER_ROLE, parts=[final_coding_prompt_text])]  # type: ignore

            # Create a future for this specific attempt
            attempt_future = asyncio.Future()
            self._active_generation_tasks[task.request_id] = task  # Re-assign to ensure it's the current one

            # This will emit llmResponseCompleted or llmResponseError, handled by _handle_llm_completion/_handle_llm_error
            # which will then call _handle_generated_code_for_file that sets the future's result/exception.
            self._backend_coordinator.start_llm_streaming_task(
                request_id=task.request_id,  # Unique request_id for each attempt is good
                target_backend_id=self._project_context['coder_backend'],
                history_to_send=history,
                is_modification_response_expected=True,
                options={"temperature": 0.05 + (attempt * 0.05)},  # Slightly increase temp on retries
                request_metadata={
                    "purpose": "autonomous_coding",
                    "sequence_id": self._sequence_id,
                    "filename": task.filename,
                    "attempt_number": attempt + 1,
                    "project_id": self._project_context.get('project_id'),
                    "session_id": self._project_context.get('session_id')
                }
            )

            try:
                # Wait for the LLM response for this attempt
                # The result of the future will be set by _handle_generated_code_for_file
                # This is a simplified way to "await" the external signal.
                # A more robust system might use asyncio.Event or a Queue per request.
                # For now, we assume _handle_generated_code_for_file will set the result/exception on task.
                # Let's use a timeout here to prevent indefinite blocking if something goes wrong with signals
                # The _active_generation_tasks being popped in _handle_llm_completion is key.
                # This needs a rethink. The signal will call _handle_generated_code_for_file.
                # This async function should just make the call and then let the signal handler do its work.
                # The overall batch processing will handle waiting.
                # This function's job is just to *initiate* the request.
                # The state of `task.validation_passed` will be checked by the batch processor.
                return  # Let the signal handler manage the outcome of this specific LLM call.

            except asyncio.TimeoutError:
                task.error_message = "LLM response timeout during code generation."
                logger.error(
                    f"PACC ({self._sequence_id}): Timeout waiting for code generation response for '{task.filename}'.")
                # self._active_generation_tasks.pop(task.request_id, None) # Clean up from active tasks
                # No break here, let the loop attempt retries if configured
            except Exception as e_gen:
                task.error_message = f"LLM call error: {e_gen}"
                logger.error(
                    f"PACC ({self._sequence_id}): Exception during code generation for '{task.filename}': {e_gen}",
                    exc_info=True)
                # self._active_generation_tasks.pop(task.request_id, None)
                # No break, allow retries

            if task.validation_passed:  # Set by _handle_generated_code_for_file
                break  # Successful generation

            if attempt < task.max_retries:
                logger.info(f"PACC ({self._sequence_id}): Will retry '{task.filename}' (attempt {attempt + 2}).")
                await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff for retries
            else:
                logger.error(
                    f"PACC ({self._sequence_id}): All {task.max_retries + 1} attempts failed for '{task.filename}'. Last error: {task.error_message}")
                # Final failure for this task
                break

        # If still no valid code after all retries for this task.
        if not task.validation_passed:
            self._emit_chat_message_to_ui(
                f"[System Error: Failed to generate valid code for {task.filename} after {task.max_retries + 1} attempts. Last error: {task.error_message or 'Unknown'}]",
                is_error=True)

    def _gather_context_from_dependencies(self, dependencies: List[str]) -> str:
        """Gathers summaries or API contracts of already generated dependency files."""
        context_parts = []
        for dep_filename in dependencies:
            if dep_filename in self._generated_files_context:
                # Prioritize API contract if available from plan, else generated code summary
                dep_task = next((t for t in self._file_tasks if t.filename == dep_filename), None)
                api_contract_in_plan = ""
                if dep_task:
                    api_contract_match = re.search(r"\*\*API_CONTRACT\*\*:\s*```python\n(.*?)\n```",
                                                   dep_task.instructions, re.DOTALL | re.IGNORECASE)
                    if api_contract_match:
                        api_contract_in_plan = api_contract_match.group(1).strip()

                if api_contract_in_plan:
                    context_parts.append(
                        f"--- Context from {dep_filename} (API Contract from Plan) ---\n```python\n{api_contract_in_plan}\n```")
                else:
                    # Fallback to summarizing the generated code
                    code_summary = self._summarize_generated_code(self._generated_files_context[dep_filename])
                    context_parts.append(
                        f"--- Context from {dep_filename} (Generated Code Summary) ---\n{code_summary}")
            else:
                context_parts.append(
                    f"--- Note: Dependency '{dep_filename}' has not been generated yet or its code is unavailable. Proceed with defined interfaces if possible. ---")
        return "\n\n".join(
            context_parts) if context_parts else "No prior generated code context from dependencies for this file."

    def _summarize_generated_code(self, code_text: str) -> str:
        """Creates a brief summary of generated code (e.g., public functions/classes)."""
        if not code_text: return "No code content to summarize."
        try:
            tree = ast.parse(code_text)
            summary = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    args = [arg.arg for arg in node.args.args]
                    summary.append(f"def {node.name}({', '.join(args)})")
                elif isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_"):
                    args = [arg.arg for arg in node.args.args]
                    summary.append(f"async def {node.name}({', '.join(args)})")
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    summary.append(f"class {node.name}:")
                    # Optionally list public methods
                    # for sub_node in node.body:
                    #     if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub_node.name.startswith("_"):
                    #         summary.append(f"  def {sub_node.name}(...)")
            if not summary: return "Contains general statements or private members."
            return "\n".join(summary[:10])  # Limit summary length
        except SyntaxError:
            return "Code has syntax errors, cannot summarize structure."
        except Exception:
            return "Could not generate structural summary."

    def _construct_final_coding_prompt(self, task: FileGenerationTask, instructions: str,
                                       dependency_context: str) -> str:
        """Constructs the full prompt for the coder LLM for a single file."""
        # Use the enhanced coding prompt from llm_prompts
        base_prompt = getattr(llm_prompts, 'ENHANCED_CODING_SYSTEM_PROMPT',
                              "Error: ENHANCED_CODING_SYSTEM_PROMPT not found")

        # Task-specific prompt part based on file type/task type
        task_type_prompt_key = f"ENHANCED_{task.task_type.upper()}_DEVELOPMENT_PROMPT"
        task_specific_header = getattr(llm_prompts, task_type_prompt_key, "")
        if not task_specific_header and task.task_type not in ['general', 'core', 'utility',
                                                               'config']:  # Avoid double general prompt
            task_specific_header = getattr(llm_prompts, 'GENERAL_CODING_PROMPT', "")

        file_spec_prompt = (
            f"{task_specific_header}\n\n"
            f"**Current File for Generation**: `{task.filename}`\n"
            f"**Overall Project Goal**: {self._original_query}\n\n"
            f"**Specific Instructions for `{task.filename}`**:\n{instructions}\n\n"
        )
        if dependency_context and dependency_context != "No prior generated code context from dependencies for this file.":
            file_spec_prompt += f"**Context from Dependent Files (API Contracts / Summaries)**:\n{dependency_context}\n\n"

        file_spec_prompt += (
            f"**Reminder of Quality Standards**: Refer to the initial system prompt for comprehensive quality requirements including PEP8, type hints, docstrings, error handling, logging, and production-readiness.\n"
            f"Focus solely on generating the complete, executable Python code for `{task.filename}`. Adhere strictly to the output format:\n"
            f"```python\n# Your COMPLETE code for {task.filename} starts here\n```\n"
        )
        return base_prompt + "\n\n" + file_spec_prompt

    def _handle_generated_code_for_file(self, task: FileGenerationTask, raw_llm_response: str):
        """Processes the LLM response for a single file's code generation attempt."""
        logger.debug(
            f"PACC ({self._sequence_id}): Processing generated code for '{task.filename}'. Raw response length: {len(raw_llm_response)}")

        extracted_code, quality, notes = self._code_processor.process_llm_response(
            raw_llm_response, task.filename, "python"  # Assuming Python for now
        )

        task.generated_code = extracted_code
        task.code_quality = quality
        task.processing_notes = notes

        if extracted_code:
            is_syntactically_valid, syntax_error = self._code_processor._is_valid_python_code(
                extracted_code)  # type: ignore
            if is_syntactically_valid:
                task.validation_passed = True  # Basic syntax validation passed
                task.error_message = None
                formatted_code_for_disk = self._code_processor.clean_and_format_code(extracted_code)
                self._write_file_to_disk(task.filename, formatted_code_for_disk)
                self._generated_files_context[task.filename] = formatted_code_for_disk  # Store cleaned code
                self._event_bus.modificationFileReadyForDisplay.emit(task.filename, formatted_code_for_disk)
                self._emit_chat_message_to_ui(
                    f"[System: Successfully generated and validated syntax for {task.filename} (Quality: {quality.name if quality else 'N/A'}).]")
                logger.info(
                    f"PACC ({self._sequence_id}): Successfully generated and saved '{task.filename}'. Quality: {quality.name if quality else 'N/A'}")
            else:
                task.validation_passed = False
                task.error_message = f"Syntax error: {syntax_error}"
                logger.warning(
                    f"PACC ({self._sequence_id}): Generated code for '{task.filename}' has syntax errors: {syntax_error}")
                self._emit_chat_message_to_ui(
                    f"[System Error: Generated code for {task.filename} has syntax errors: {syntax_error}]",
                    is_error=True)
        else:
            task.validation_passed = False
            task.error_message = f"Code extraction failed. Notes: {', '.join(notes)}"
            logger.error(f"PACC ({self._sequence_id}): Failed to extract code for '{task.filename}'. Notes: {notes}")
            self._emit_chat_message_to_ui(
                f"[System Error: Could not extract code for {task.filename}. LLM response might be malformed.]",
                is_error=True)

        # This task attempt is now complete (either success or failure for this attempt)
        # The batch processor will decide if retries are needed or if the batch/sequence should halt.

    def _create_self_critique_prompt_for_task(self, task: FileGenerationTask) -> str:
        """Creates a prompt for the LLM to critique and improve its own generated code."""
        if not task.generated_code:
            return task.instructions  # Cannot critique if no code was generated

        error_info = f"Previous Error (if any): {task.error_message}\n" if task.error_message else ""
        quality_info = f"Assessed Quality: {task.code_quality.name if task.code_quality else 'Not Assessed'}\n"
        processing_notes_info = f"Processing Notes: {', '.join(task.processing_notes)}\n" if task.processing_notes else ""

        # Use the quality validation prompt as a basis for critique
        critique_guidance = getattr(llm_prompts, 'CODE_QUALITY_VALIDATION_PROMPT',
                                    "Review against high quality standards.")

        return (
            f"The previous attempt to generate code for `{task.filename}` had issues. Please review and improve.\n\n"
            f"**Original Instructions for `{task.filename}`**:\n{task.instructions}\n\n"
            f"**Previously Generated Code for `{task.filename}` (may contain errors)**:\n"
            f"```python\n{task.generated_code}\n```\n\n"
            f"**Issues from Previous Attempt**:\n{error_info}{quality_info}{processing_notes_info}\n"
            f"**Self-Critique and Improvement Guidelines (refer to these meticulously)**:\n{critique_guidance}\n\n"
            f"Your task is to regenerate the COMPLETE, CORRECTED, and IMPROVED Python code for `{task.filename}`. "
            f"Address all identified issues and strictly adhere to all quality standards mentioned in the initial system prompt and the critique guidelines. "
            f"Output ONLY the full, corrected Python code in a single fenced code block."
        )

    def _write_file_to_disk(self, relative_filename: str, content: str):
        """Writes the generated file content to the project directory."""
        try:
            project_root = self._project_context.get('project_dir')
            if not project_root:
                logger.error(
                    f"PACC ({self._sequence_id}): Project directory not set. Cannot write file '{relative_filename}'.")
                raise IOError("Project directory context is missing.")

            # Ensure relative_filename does not try to escape project_root (e.g., "../../../etc/passwd")
            # os.path.join on its own doesn't prevent this if relative_filename starts with / or similar.
            # We need to ensure the final path is truly within project_root.
            abs_file_path = os.path.abspath(os.path.join(project_root, relative_filename))

            # Security check: ensure the resolved path is under the project root
            if not abs_file_path.startswith(os.path.abspath(project_root)):
                logger.error(
                    f"PACC ({self._sequence_id}): Security risk! Attempt to write file '{relative_filename}' outside project root '{project_root}'. Path resolved to '{abs_file_path}'. Blocked.")
                raise SecurityException(f"Attempt to write outside project root: {relative_filename}")  # type: ignore

            os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)

            # It's generally good practice to clean/format before writing
            # formatted_content = self._code_processor.clean_and_format_code(content) # Already done in handler

            with open(abs_file_path, 'w', encoding='utf-8') as f:
                f.write(content)  # Write the already formatted content
            logger.info(f"PACC ({self._sequence_id}): Successfully wrote file to: {abs_file_path}")
            self._log_communication("FILE_WROTE", f"File: {abs_file_path}, Size: {len(content)} bytes")

        except IOError as e_io:
            logger.error(
                f"PACC ({self._sequence_id}): IO Error writing file '{relative_filename}' to '{project_root}': {e_io}",
                exc_info=True)
            raise  # Re-raise to be handled by the calling generation step
        except Exception as e_write:
            logger.error(f"PACC ({self._sequence_id}): Unexpected error writing file '{relative_filename}': {e_write}",
                         exc_info=True)
            raise

    def _finalize_sequence_if_all_done(self):
        """Checks if all batches and tasks are complete, then finalizes the sequence."""
        if self._current_batch_index >= len(self._generation_batches):  # All batches processed
            all_tasks_successful = all(task.validation_passed for task in self._file_tasks)
            num_successful = sum(1 for task in self._file_tasks if task.validation_passed)
            num_failed = len(self._file_tasks) - num_successful

            if all_tasks_successful:
                final_message = f"[System: Autonomous coding sequence '{self._sequence_id}' completed successfully! All {len(self._file_tasks)} files generated and validated.]"
                self._emit_status_update(f"✅ Autonomous coding complete: {len(self._file_tasks)} files generated.",
                                         "#98c379", False)
                logger.info(f"PACC ({self._sequence_id}): Sequence completed successfully.")
            else:
                final_message = (f"[System: Autonomous coding sequence '{self._sequence_id}' completed with issues. "
                                 f"Successfully generated: {num_successful}/{len(self._file_tasks)}. "
                                 f"Failed: {num_failed}. Please review logs and generated files.]")
                self._emit_status_update(
                    f"⚠️ Autonomous coding finished: {num_successful}/{len(self._file_tasks)} files. Check issues.",
                    "#e5c07b", False)
                logger.warning(f"PACC ({self._sequence_id}): Sequence completed with {num_failed} failed task(s).")

            self._emit_chat_message_to_ui(final_message, is_error=(not all_tasks_successful))
            self._log_communication("SEQ_COMPLETE", f"Successful: {num_successful}, Failed: {num_failed}")
            self._reset_sequence_state(error_occurred=(not all_tasks_successful))

    @Slot(str, str)  # request_id, error_message
    def _handle_llm_error(self, request_id: str, error_message: str):
        if not self._sequence_id: return  # No active sequence

        task_details = self._active_generation_tasks.pop(request_id, None)
        if not task_details:
            logger.debug(
                f"PACC ({self._sequence_id}): Received LLM error for unknown or already handled request ID: {request_id}")
            return

        # Check if this error was for the planning phase or a code generation phase
        if task_details.filename == "<PLANNING_PHASE>":  # The dummy task for planning
            logger.error(f"PACC ({self._sequence_id}): Planning phase LLM error (ReqID: {request_id}): {error_message}")
            self._emit_chat_message_to_ui(
                f"[System Error: Failed to generate project plan. LLM Error: {error_message}]", is_error=True)
            self._reset_sequence_state(error_occurred=True, message=f"Planning failed: {error_message}")
        else:  # Error during code generation for a specific file
            task_details.error_message = error_message
            task_details.validation_passed = False  # Mark as failed validation
            logger.error(
                f"PACC ({self._sequence_id}): Code generation LLM error for '{task_details.filename}' (ReqID: {request_id}): {error_message}")
            # The batch processing logic (_process_current_generation_batch) will handle retries or halting.
            # We don't reset the whole sequence here, just this task's attempt.
            # If this was the last attempt for this task, the batch processor will eventually halt.

            # Check if this error occurred for a task in the currently active batch
            active_batch = None
            if 0 <= self._current_batch_index < len(self._generation_batches):
                active_batch = self._generation_batches[self._current_batch_index]

            if active_batch and active_batch.is_active and task_details in active_batch.files:
                # If error for a file in current active batch, this specific file attempt failed.
                # The gather() in _process_current_generation_batch will catch this exception if we re-raise or handle it by setting future.
                # For now, task.error_message is set. The batch processor will see validation_passed is False.
                pass
            else:  # Error for a task not in current batch (should not happen with current logic)
                logger.warning(
                    f"PACC ({self._sequence_id}): LLM error for task '{task_details.filename}' which is not in the current active batch. This is unexpected.")

    def _reset_sequence_state(self, error_occurred: bool = False, message: Optional[str] = None):
        phase_before_reset = self._current_phase
        logger.info(
            f"PACC ({self._sequence_id}): Resetting sequence state. Error occurred: {error_occurred}. Prev phase: {phase_before_reset.name}")

        self._current_phase = SequencePhase.IDLE
        # Don't clear sequence_id immediately if we want to log against it after reset is called.
        # seq_id_for_log = self._sequence_id
        self._sequence_id = None  # Mark as idle

        self._original_query = None
        self._project_context.clear()
        self._plan_text = None
        self._file_tasks.clear()
        self._generation_batches.clear()
        self._current_batch_index = 0
        self._generated_files_context.clear()

        # Cancel any outstanding LLM requests tied to this coordinator instance
        # This requires _backend_coordinator to have a way to cancel tasks by purpose/metadata if generic cancel is too broad.
        # For now, just clear our tracking. Specific task cancellation might be tricky if req_id was reused.
        # A better approach is that _backend_coordinator should manage its tasks and allow cancellation by request_id.
        # If an LLM request is in flight and its callback (_handle_llm_completion/_error) arrives after reset,
        # it should be ignored because self._sequence_id will be None.
        for req_id in list(self._active_generation_tasks.keys()):  # Iterate a copy
            # If backend_coordinator allows cancelling specific requests:
            # self._backend_coordinator.cancel_task_by_id(req_id) # Hypothetical
            self._active_generation_tasks.pop(req_id, None)

        self._event_bus.uiInputBarBusyStateChanged.emit(False)  # Release input bar
        if error_occurred:
            final_status = message if message else "Autonomous coding failed."
            self._emit_status_update(f"❌ {final_status}", "#e06c75", False)
        elif phase_before_reset not in [SequencePhase.IDLE,
                                        SequencePhase.FINALIZATION]:  # Don't show "ready" if it was just finalizing
            self._emit_status_update("Ready for new autonomous coding task.", "#98c379", False)

    def _log_communication(self, stage: str, message: str):
        if self._llm_comm_logger and self._sequence_id:
            self._llm_comm_logger.log_message(f"PACC:{self._sequence_id}:{stage}", message)
        else:
            logger.debug(f"PACC_LOG_FALLBACK ({self._sequence_id or 'NO_SEQ_ID'} - {stage}): {message[:100]}...")

    def _emit_status_update(self, message: str, color: str, is_temporary: bool, duration_ms: int = 0):
        self._event_bus.uiStatusUpdateGlobal.emit(message, color, is_temporary, duration_ms)

    def _emit_chat_message_to_ui(self, text: str, is_error: bool = False):
        project_id = self._project_context.get('project_id')
        session_id = self._project_context.get('session_id')
        if project_id and session_id:
            role = ERROR_ROLE if is_error else SYSTEM_ROLE
            # Use a unique ID for these system messages if they need to be targetable (e.g., for updates)
            msg_id = f"pacc_msg_{self._sequence_id}_{uuid.uuid4().hex[:4]}"
            chat_msg = ChatMessage(id=msg_id, role=role, parts=[text])  # type: ignore
            self._event_bus.newMessageAddedToHistory.emit(project_id, session_id, chat_msg)
        else:
            logger.warning(
                f"PACC ({self._sequence_id}): Cannot emit chat message - project_id or session_id missing from context.")

    def is_busy(self) -> bool:
        """Returns True if the coordinator is currently processing a sequence."""
        return self._current_phase != SequencePhase.IDLE

    # Placeholder for SecurityException if needed
    # class SecurityException(Exception): pass