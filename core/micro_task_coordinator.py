# app/core/micro_task_coordinator.py
import asyncio
import logging
import os
import uuid
import re  # For parsing LLM responses
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import ast  # For safely evaluating list strings from LLM

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    # Corrected paths for models and services
    from models.chat_message import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from models.message_enums import MessageLoadingState
    from llm.backend_coordinator import BackendCoordinator
    from services.llm_communication_logger import LlmCommunicationLogger
    from core.code_output_processor import CodeOutputProcessor, CodeQualityLevel
    from utils import constants
    # Assuming prompts will be in their own module, e.g., app.llm.prompts
    from app.llm import prompts as llm_prompts
except ImportError as e_mtc:
    logging.getLogger(__name__).critical(f"MicroTaskCoordinator: Critical import error: {e_mtc}", exc_info=True)
    # Fallback types
    EventBus = type("EventBus", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MessageLoadingState = type("MessageLoadingState", (object,), {})  # type: ignore
    BackendCoordinator = type("BackendCoordinator", (object,), {})
    LlmCommunicationLogger = type("LlmCommunicationLogger", (object,), {})  # type: ignore
    CodeOutputProcessor = type("CodeOutputProcessor", (object,), {})  # type: ignore
    CodeQualityLevel = type("CodeQualityLevel", (object,), {})  # type: ignore
    constants = type("constants", (object,), {})  # type: ignore
    llm_prompts = type("llm_prompts", (object,), {})  # type: ignore
    USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "model", "system", "error"
    raise

logger = logging.getLogger(__name__)


class MicroTaskPhase(Enum):
    IDLE = auto()
    INITIAL_DECOMPOSITION = auto()  # Planning model breaks down the main task
    FUNCTION_SPEC_VALIDATION = auto()  # Optional: LLM/User validates function specs
    FUNCTION_GENERATION = auto()  # Coding model generates individual functions
    FUNCTION_REVIEW_AND_RETRY = auto()  # Planning model reviews, requests fixes if needed
    ADAPTIVE_PLANNING = auto()  # Planning model adjusts plan based on generation progress/issues
    CODE_ASSEMBLY = auto()  # Assembling generated functions into files
    FINAL_VALIDATION = auto()  # Validating assembled files
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class FunctionSpecification:
    """Detailed specification for a single function to be generated."""
    id: str = field(default_factory=lambda: f"func_spec_{uuid.uuid4().hex[:8]}")
    function_name: str
    description: str  # Detailed purpose and logic
    parameters: List[Dict[str, str]]  # e.g., [{"name": "param1", "type": "int", "description": "..."}]
    return_type: str
    dependencies: List[str] = field(default_factory=list)  # Names of other functions this one depends on
    file_target: Optional[str] = None  # Suggested filename or module
    complexity_score: int = 1  # Estimated complexity (e.g., 1-5)
    priority: int = 1  # Generation priority

    # Generation State
    generated_code: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    review_feedback: Optional[str] = None  # Feedback from planning model
    generation_attempts: int = 0
    status: MicroTaskPhase = MicroTaskPhase.IDLE  # Tracks this specific function's generation status


@dataclass
class FileAssemblyInfo:
    """Tracks the assembly of functions into a complete file."""
    filename: str
    target_function_names: List[str]  # Names of functions intended for this file
    generated_functions: Dict[str, str] = field(default_factory=dict)  # name -> code
    imports_needed: Set[str] = field(default_factory=set)
    class_definitions: Dict[str, str] = field(default_factory=dict)  # class_name -> class_scaffold_code
    assembled_code: Optional[str] = None
    is_complete: bool = False


class MicroTaskCoordinator(QObject):
    """
    Coordinates the generation of code through micro-tasks.
    A planning LLM oversees the process, decomposing tasks and reviewing generated functions.
    A coding LLM generates individual functions, allowing for rapid iteration and quality control.
    """

    MAX_GENERATION_ATTEMPTS_PER_FUNCTION = 3

    def __init__(self,
                 backend_coordinator: BackendCoordinator,
                 event_bus: EventBus,
                 llm_comm_logger: Optional[LlmCommunicationLogger],  # type: ignore
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(backend_coordinator, BackendCoordinator):  # type: ignore
            raise TypeError("MicroTaskCoordinator requires a valid BackendCoordinator.")
        if not isinstance(event_bus, EventBus):  # type: ignore
            raise TypeError("MicroTaskCoordinator requires a valid EventBus.")

        self._backend_coordinator = backend_coordinator
        self._event_bus = event_bus
        self._llm_comm_logger = llm_comm_logger
        self._code_processor = CodeOutputProcessor()  # type: ignore

        # Sequence State
        self._current_phase = MicroTaskPhase.IDLE
        self._sequence_id: Optional[str] = None
        self._original_query: Optional[str] = None
        self._project_context: Dict[str, Any] = {}  # project_dir, project_id, session_id, etc.

        # Micro-task specific state
        self._function_specs_list: List[FunctionSpecification] = []
        self._file_assembly_map: Dict[str, FileAssemblyInfo] = {}  # filename -> FileAssemblyInfo
        self._current_spec_index_being_processed: int = 0
        self._active_llm_request_id: Optional[str] = None  # Tracks requests made by this coordinator

        self._connect_event_bus_handlers()
        logger.info("MicroTaskCoordinator initialized.")

    def _connect_event_bus_handlers(self):
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_completion)
        self._event_bus.llmResponseError.connect(self._handle_llm_error)

    def start_micro_task_generation(self,
                                    user_query: str,
                                    planning_backend: str,
                                    planning_model: str,
                                    coding_backend: str,
                                    coding_model: str,
                                    project_dir: str,
                                    project_id: Optional[str] = None,
                                    session_id: Optional[str] = None) -> bool:
        """Starts the micro-task generation sequence."""
        if self._current_phase != MicroTaskPhase.IDLE:
            logger.warning(
                f"MTC: Sequence '{self._sequence_id}' already active in phase {self._current_phase.name}. Ignoring new request.")
            self._emit_status_update("Micro-task generation already in progress.", "#e5c07b", True, 3000)
            return False

        self._reset_sequence_state()
        self._sequence_id = f"mtc_seq_{uuid.uuid4().hex[:8]}"
        self._original_query = user_query
        self._project_context = {
            'project_dir': project_dir, 'project_id': project_id, 'session_id': session_id,
            'planning_backend': planning_backend, 'planning_model': planning_model,
            'coding_backend': coding_backend, 'coding_model': coding_model
        }

        logger.info(f"MTC ({self._sequence_id}): Starting sequence for query: '{user_query[:50]}...'")
        self._log_communication("SEQ_START", f"Query: {user_query[:100]}...")
        self._emit_chat_message_to_ui(
            f"[System: Starting micro-task generation for '{user_query[:40]}...'. Decomposing task...]")
        self._event_bus.uiInputBarBusyStateChanged.emit(True)

        self._current_phase = MicroTaskPhase.INITIAL_DECOMPOSITION
        asyncio.create_task(self._request_initial_task_decomposition())
        return True

    async def _request_initial_task_decomposition(self):
        """Asks the planning LLM to break down the user's query into function specifications."""
        if not self._original_query or not self._sequence_id or not self._backend_coordinator: return

        self._active_llm_request_id = f"mtc_decompose_{self._sequence_id}"
        self._emit_status_update(f"Decomposing task with {self._project_context['planning_model']}...", "#61afef",
                                 False)

        # TODO: Use a well-defined prompt from llm_prompts for decomposition
        decomposition_prompt = (
            f"You are an expert software architect. Decompose the following user request into a list of precise Python function specifications. "
            f"For each function, provide: name, detailed description/purpose, parameters (name, type, description), return type, and any direct function dependencies (by name) from this list.\n"
            f"User Request: \"{self._original_query}\"\n\n"
            f"Output Format (Strictly follow, use JSON-like list of objects if possible, or very clear structured text):\n"
            f"FUNCTION_SPECS_START\n"
            f"Function: [function_name_1]\n"
            f"Description: [Detailed purpose and logic for function_name_1]\n"
            f"Parameters: (param1: type, # description; param2: type, # description)\n"  # Or list of dicts
            f"ReturnType: [return_type_1]\n"
            f"Dependencies: [function_name_x, function_name_y]\n"
            f"FileTarget: [optional_suggested_filename.py]\n"  # Added FileTarget
            f"---\n"
            f"Function: [function_name_2]\n"
            f"...\n"
            f"FUNCTION_SPECS_END"
        )
        history = [ChatMessage(role=USER_ROLE, parts=[decomposition_prompt])]  # type: ignore

        self._backend_coordinator.start_llm_streaming_task(
            request_id=self._active_llm_request_id,
            target_backend_id=self._project_context['planning_backend'],
            history_to_send=history,
            is_modification_response_expected=True,
            options={"temperature": 0.2},  # Low temp for structured output
            request_metadata={
                "purpose": "microtask_decomposition", "sequence_id": self._sequence_id,
                "project_id": self._project_context.get('project_id'),
                "session_id": self._project_context.get('session_id')
            }
        )

    @Slot(str, ChatMessage, dict)  # type: ignore
    def _handle_llm_completion(self, request_id: str, message: ChatMessage, metadata: dict):  # type: ignore
        if not self._sequence_id or metadata.get(
                "sequence_id") != self._sequence_id or self._active_llm_request_id != request_id:
            return  # Not for current MTC sequence or not the expected request

        purpose = metadata.get("purpose")
        logger.info(f"MTC ({self._sequence_id}): LLM completion for ReqID '{request_id}', Purpose: '{purpose}'")
        self._active_llm_request_id = None  # Clear active request ID

        if purpose == "microtask_decomposition":
            self._process_initial_decomposition_response(message.text)  # type: ignore
        elif purpose == "microtask_function_generation":
            spec_id = metadata.get("function_spec_id")
            self._process_generated_function_code(spec_id, message.text)  # type: ignore
        elif purpose == "microtask_function_review":
            spec_id = metadata.get("function_spec_id")
            self._process_function_review_response(spec_id, message.text)  # type: ignore
        elif purpose == "microtask_adaptive_planning":
            self._process_adaptive_planning_response(message.text)  # type: ignore
        elif purpose == "microtask_code_assembly":
            filename = metadata.get("filename")
            self._process_assembled_code_response(filename, message.text)  # type: ignore

    def _process_initial_decomposition_response(self, llm_response_text: str):
        try:
            self._function_specs_list = self._parse_function_specs_from_llm(llm_response_text)
            if not self._function_specs_list:
                raise ValueError("LLM decomposition response did not yield any function specifications.")

            self._total_functions_planned = len(self._function_specs_list)
            self._functions_generated = 0
            logger.info(
                f"MTC ({self._sequence_id}): Successfully decomposed task into {self._total_functions_planned} function specs.")
            self._emit_chat_message_to_ui(
                f"[System: Task decomposed into {self._total_functions_planned} micro-tasks (functions). Starting generation...]")

            # Prepare file assemblies based on FileTarget in specs
            self._prepare_file_assemblies()

            self._current_spec_index_being_processed = 0
            self._current_phase = MicroTaskPhase.FUNCTION_GENERATION
            asyncio.create_task(self._process_next_function_spec())

        except ValueError as e_parse:
            logger.error(
                f"MTC ({self._sequence_id}): Failed to parse decomposition response: {e_parse}\nResponse:\n{llm_response_text[:500]}",
                exc_info=True)
            self._emit_chat_message_to_ui(
                f"[System Error: Could not understand the task decomposition from AI. Error: {e_parse}]", is_error=True)
            self._reset_sequence_state(error_occurred=True)
        except Exception as e_proc_decomp:
            logger.error(f"MTC ({self._sequence_id}): Unexpected error processing decomposition: {e_proc_decomp}",
                         exc_info=True)
            self._emit_chat_message_to_ui(
                f"[System Error: Unexpected error during task decomposition. Details: {e_proc_decomp}]", is_error=True)
            self._reset_sequence_state(error_occurred=True)

    def _parse_function_specs_from_llm(self, response_text: str) -> List[FunctionSpecification]:
        """Parses the LLM's structured text output into FunctionSpecification objects."""
        specs = []
        # Look for the main block first
        main_block_match = re.search(r"FUNCTION_SPECS_START(.*?)FUNCTION_SPECS_END", response_text,
                                     re.DOTALL | re.IGNORECASE)
        content_to_parse = main_block_match.group(1) if main_block_match else response_text

        # Split into individual function sections based on "---" or "Function:"
        raw_function_blocks = re.split(r"\n---\n|\nFunction:", content_to_parse, flags=re.IGNORECASE)

        for block in raw_function_blocks:
            block = block.strip()
            if not block: continue

            name_match = re.search(r"^(?:Function:)?\s*([a-zA-Z_][a-zA-Z0-9_]*)", block, re.IGNORECASE)
            desc_match = re.search(r"Description:\s*(.+)", block, re.IGNORECASE)
            params_match = re.search(r"Parameters:\s*\((.*?)\)", block, re.IGNORECASE)  # Matches content inside ()
            # More robust param parsing: list of dicts or string
            params_list_match = re.search(r"Parameters:\s*(\[.*?\])", block, re.DOTALL | re.IGNORECASE)

            ret_match = re.search(r"ReturnType:\s*(.+)", block, re.IGNORECASE)
            deps_match = re.search(r"Dependencies:\s*(\[.*?\])", block, re.IGNORECASE)
            file_target_match = re.search(r"FileTarget:\s*([\w\-./]+\.py)", block, re.IGNORECASE)

            if not name_match or not desc_match:
                logger.warning(f"MTC: Skipping malformed function spec block: {block[:100]}...")
                continue

            name = name_match.group(1).strip()
            description = desc_match.group(1).strip()
            return_type = ret_match.group(1).strip() if ret_match else "Any"

            parameters = []
            if params_list_match:  # Prefer list of dicts if available
                try:
                    params_list_str = params_list_match.group(1)
                    parsed_params = ast.literal_eval(params_list_str)
                    if isinstance(parsed_params, list):
                        for p_dict in parsed_params:
                            if isinstance(p_dict, dict) and "name" in p_dict and "type" in p_dict:
                                parameters.append({
                                    "name": p_dict["name"],
                                    "type": p_dict["type"],
                                    "description": p_dict.get("description", "")
                                })
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"MTC: Could not parse Parameters list '{params_list_match.group(1)}': {e}")
            elif params_match:  # Fallback to string parsing
                params_str = params_match.group(1).strip()
                if params_str:
                    param_parts = params_str.split(';')
                    for part in param_parts:
                        match_param_detail = re.match(r"\s*(\w+)\s*:\s*([\w\[\],\.\|\s\w]+)(?:\s*#\s*(.*))?",
                                                      part.strip())
                        if match_param_detail:
                            p_name, p_type, p_desc = match_param_detail.groups()
                            parameters.append(
                                {"name": p_name.strip(), "type": p_type.strip(), "description": (p_desc or "").strip()})

            dependencies = []
            if deps_match:
                try:
                    deps_list_str = deps_match.group(1)
                    deps = ast.literal_eval(deps_list_str)
                    if isinstance(deps, list) and all(isinstance(d, str) for d in deps):
                        dependencies = [d.strip() for d in deps if d.strip()]
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"MTC: Could not parse Dependencies list '{deps_match.group(1)}': {e}")
                    # Regex fallback for simple comma-separated names within brackets
                    dependencies = [d.strip("'\" ") for d in re.findall(r"['\"]([^'\"]+)['\"]", deps_match.group(1))]

            file_target = file_target_match.group(1).strip() if file_target_match else None
            if not file_target:  # Default if not specified
                file_target = "main_module.py"  # Or some other logic to assign files

            specs.append(FunctionSpecification(
                function_name=name,
                description=description,
                parameters=parameters,
                return_type=return_type,
                dependencies=dependencies,
                file_target=file_target
            ))
        return specs

    def _prepare_file_assemblies(self):
        """Initializes FileAssemblyInfo objects based on FileTarget in function specs."""
        self._file_assembly_map.clear()
        for spec in self._function_specs_list:
            if spec.file_target:
                if spec.file_target not in self._file_assembly_map:
                    self._file_assembly_map[spec.file_target] = FileAssemblyInfo(
                        filename=spec.file_target,
                        target_function_names=[]
                    )
                self._file_assembly_map[spec.file_target].target_function_names.append(spec.function_name)
            else:
                logger.warning(
                    f"MTC: Function spec '{spec.function_name}' missing FileTarget. It might not be assembled.")
        logger.info(f"MTC: Prepared {len(self._file_assembly_map)} file assemblies.")

    async def _process_next_function_spec(self):
        """Initiates generation for the next function in the list, or moves to review/assembly."""
        if self._current_spec_index_being_processed >= len(self._function_specs_list):
            logger.info(f"MTC ({self._sequence_id}): All function specifications processed. Moving to code assembly.")
            self._current_phase = MicroTaskPhase.CODE_ASSEMBLY
            asyncio.create_task(self._assemble_all_files())
            return

        current_spec = self._function_specs_list[self._current_spec_index_being_processed]

        # Check if all dependencies for the current spec are met (i.e., their code is generated)
        dependencies_met = True
        for dep_name in current_spec.dependencies:
            dep_spec = next((s for s in self._function_specs_list if s.function_name == dep_name), None)
            if not dep_spec or not dep_spec.generated_code:
                dependencies_met = False
                logger.info(
                    f"MTC ({self._sequence_id}): Dependency '{dep_name}' for '{current_spec.function_name}' not yet met. Postponing.")
                # This basic implementation will just wait. A more advanced one could reorder.
                # For now, we assume the initial decomposition provides a somewhat valid order,
                # or we process in batches where dependencies within a batch are not critical.
                # This simplified version might get stuck if order is strictly required and not provided.
                # A better approach is topological sort for specs or batching.
                # For this iteration, if a dependency isn't met, we might error or just try.
                # Let's assume for now that the order from decomposition is generally usable.
                break

                # If not all dependencies met, an advanced system might re-prioritize. Here, we proceed cautiously.
        if not dependencies_met:
            logger.warning(
                f"MTC ({self._sequence_id}): Strict dependencies for {current_spec.function_name} not met. Proceeding, but coding LLM must handle.")

        self._current_phase = MicroTaskPhase.FUNCTION_GENERATION
        current_spec.status = MicroTaskPhase.FUNCTION_GENERATION
        await self._request_function_generation(current_spec)

    async def _request_function_generation(self, spec: FunctionSpecification):
        """Asks the coding LLM to generate code for a single function specification."""
        if not self._sequence_id or not self._backend_coordinator: return

        self._active_llm_request_id = f"mtc_gen_func_{spec.id}"
        spec.request_id = self._active_llm_request_id  # Store request_id in the spec
        spec.generation_attempts += 1

        self._emit_status_update(f"Generating function '{spec.function_name}' (Attempt {spec.generation_attempts})...",
                                 "#c678dd", False)
        self._emit_chat_message_to_ui(
            f"[System: Generating function: `{spec.function_name}`... Attempt {spec.generation_attempts}]")

        # Gather context from already generated dependencies
        dependency_context_code = []
        for dep_name in spec.dependencies:
            dep_func_spec = next(
                (s for s in self._function_specs_list if s.function_name == dep_name and s.generated_code), None)
            if dep_func_spec and dep_func_spec.generated_code:
                dependency_context_code.append(f"# From dependency: {dep_name}\n{dep_func_spec.generated_code}")

        context_str = "\n\n".join(
            dependency_context_code) if dependency_context_code else "No prior generated code context available for dependencies."

        # TODO: Use a well-defined prompt from llm_prompts for function generation
        # This prompt should include the spec details and the dependency context.
        params_str_list = []
        for p in spec.parameters:
            param_desc = f" # {p['description']}" if p.get('description') else ""
            params_str_list.append(f"{p['name']}: {p['type']}{param_desc}")
        params_str_for_prompt = ", ".join(params_str_list)

        generation_prompt = (
            f"Generate the Python code for the following function specification:\n\n"
            f"Function Name: {spec.function_name}\n"
            f"Description: {spec.description}\n"
            f"Parameters: ({params_str_for_prompt})\n"
            f"Return Type: {spec.return_type}\n"
            f"Target File (for context): {spec.file_target or 'Not specified'}\n"
            f"Detailed Requirements: {spec.detailed_requirements if hasattr(spec, 'detailed_requirements') and spec.detailed_requirements else 'Implement as per description.'}\n\n"
            f"Context from dependent functions (already generated or planned):\n{context_str}\n\n"
            f"Adhere to these coding standards:\n"
            f"- Include full type hints for all parameters and the return value.\n"
            f"- Write a comprehensive Google-style docstring (Args, Returns, Raises sections).\n"
            f"- Implement robust error handling using try-except blocks for anticipated issues.\n"
            f"- Add INFO level logging for key operations and ERROR/EXCEPTION for failures.\n"
            f"- Ensure the function is self-contained or only uses provided dependencies or standard libraries.\n"
            f"- Output ONLY the Python function code, starting with 'def' or 'async def'. No extra text, explanations, or markdown fences."
        )

        history = [ChatMessage(role=USER_ROLE, parts=[generation_prompt])]  # type: ignore

        self._backend_coordinator.start_llm_streaming_task(
            request_id=self._active_llm_request_id,
            target_backend_id=self._project_context['coding_backend'],
            history_to_send=history,
            is_modification_response_expected=True,
            options={"temperature": 0.1},  # Very low temp for precise code
            request_metadata={
                "purpose": "microtask_function_generation", "sequence_id": self._sequence_id,
                "function_spec_id": spec.id,  # Pass spec ID to map response back
                "project_id": self._project_context.get('project_id'),
                "session_id": self._project_context.get('session_id')
            }
        )

    def _process_generated_function_code(self, spec_id: Optional[str], generated_code_text: str):
        """Processes the generated code for a function, validates it, and decides next step."""
        if not spec_id:
            logger.error(f"MTC ({self._sequence_id}): Received generated function code without spec_id.");
            return

        spec = next((s for s in self._function_specs_list if s.id == spec_id), None)
        if not spec:
            logger.error(f"MTC ({self._sequence_id}): Could not find function spec for ID '{spec_id}'.");
            return

        logger.info(f"MTC ({self._sequence_id}): Processing generated code for function '{spec.function_name}'.")

        # Use CodeOutputProcessor to extract and clean the code
        extracted_code, quality, notes = self._code_processor.process_llm_response(  # type: ignore
            generated_code_text, f"{spec.function_name}.py"  # Treat as a mini-file for processing
        )

        spec.generated_code = extracted_code
        spec.validation_errors.clear()

        if not extracted_code:
            spec.validation_errors.append("Code extraction failed. LLM did not produce a recognizable code block.")
            logger.warning(
                f"MTC ({self._sequence_id}): Code extraction failed for '{spec.function_name}'. Notes: {notes}")
        else:
            is_valid, syntax_error = self._code_processor._is_valid_python_code(extracted_code)  # type: ignore
            if not is_valid:
                spec.validation_errors.append(f"Syntax error: {syntax_error or 'Unknown syntax issue'}")
            # TODO: Add more sophisticated validation (e.g., checking parameters, return type consistency with spec)

        if not spec.validation_errors:
            spec.status = MicroTaskPhase.COMPLETED  # Mark this function as tentatively complete
            self._functions_generated += 1
            self._completed_functions[spec.function_name] = spec.generated_code  # Store for context

            # Add to file assembly
            if spec.file_target and spec.file_target in self._file_assembly_map:
                self._file_assembly_map[spec.file_target].generated_functions[spec.function_name] = spec.generated_code

            logger.info(
                f"MTC ({self._sequence_id}): Function '{spec.function_name}' generated and passed basic validation (Attempt {spec.generation_attempts}).")
            self._emit_chat_message_to_ui(
                f"[System: Successfully generated function `{spec.function_name}`. ({self._functions_generated}/{self._total_functions_planned})]")

            # Move to next spec
            self._current_spec_index_being_processed += 1
            asyncio.create_task(self._process_next_function_spec())
        else:  # Validation failed
            logger.warning(
                f"MTC ({self._sequence_id}): Validation failed for '{spec.function_name}' (Attempt {spec.generation_attempts}): {spec.validation_errors}")
            if spec.generation_attempts < self.MAX_GENERATION_ATTEMPTS_PER_FUNCTION:
                self._emit_chat_message_to_ui(
                    f"[System: Validation failed for `{spec.function_name}`. Errors: {'; '.join(spec.validation_errors)}. Retrying...]")
                # TODO: Optionally, could involve planning model for review/retry prompt generation here
                # For now, just retry with the same (or slightly modified) generation prompt
                asyncio.create_task(self._request_function_generation(spec))  # Retry
            else:
                logger.error(
                    f"MTC ({self._sequence_id}): Max retries reached for function '{spec.function_name}'. Marking as ERROR.")
                spec.status = MicroTaskPhase.ERROR
                self._emit_chat_message_to_ui(
                    f"[System Error: Failed to generate valid code for function `{spec.function_name}` after {spec.generation_attempts} attempts. Errors: {'; '.join(spec.validation_errors)}]",
                    is_error=True)
                # Decide how to handle overall sequence failure - for now, continue with other functions if possible
                # or halt the entire sequence.
                self._current_spec_index_being_processed += 1  # Move on, but this function is errored
                asyncio.create_task(self._process_next_function_spec())

    async def _assemble_all_files(self):
        """Assembles all generated functions into their target files."""
        logger.info(
            f"MTC ({self._sequence_id}): Starting code assembly phase for {len(self._file_assembly_map)} files.")
        self._emit_chat_message_to_ui("[System: All functions generated. Assembling files...]")
        self._emit_status_update("Assembling code into files...", "#56b6c2", False)

        for filename, assembly_info in self._file_assembly_map.items():
            logger.info(f"MTC ({self._sequence_id}): Assembling file '{filename}'.")
            # TODO: Use a more sophisticated assembly prompt if needed, e.g., asking LLM to arrange functions, add imports, class structures.
            # For now, simple concatenation with basic import and class scaffolding.

            all_code_for_file = []
            # Basic import gathering (can be improved by LLM or AST analysis later)
            # For now, let's assume coding LLM included necessary imports within functions or we handle it globally.

            # Simple concatenation of functions intended for this file
            for func_name in assembly_info.target_function_names:
                if func_name in assembly_info.generated_functions:
                    all_code_for_file.append(assembly_info.generated_functions[func_name])
                else:
                    logger.warning(
                        f"MTC ({self._sequence_id}): Function '{func_name}' targeted for '{filename}' was not successfully generated.")
                    all_code_for_file.append(
                        f"\n# TODO: Implement function: {func_name} (generation failed or pending)\npass\n")

            assembled_code = "\n\n".join(all_code_for_file)

            # Add a basic header
            header = f"# File: {filename}\n# Generated by AvA Micro-Task Coordinator (Sequence: {self._sequence_id})\n"
            header += f"# Original User Query: {self._original_query[:100]}...\n\n"
            # TODO: Add logic to collect and add necessary imports at the top.
            # For now, assume functions included their own imports or they are standard libs.

            final_code_for_file = header + assembled_code
            assembly_info.assembled_code = final_code_for_file

            # Write to disk
            self._write_assembled_file_to_disk(filename, final_code_for_file)
            self._event_bus.modificationFileReadyForDisplay.emit(filename, final_code_for_file)
            assembly_info.is_complete = True

        self._current_phase = MicroTaskPhase.FINAL_VALIDATION  # Or COMPLETED if no further validation
        self._finalize_micro_task_sequence()

    def _write_assembled_file_to_disk(self, relative_filename: str, content: str):
        """Writes the assembled file content to the project directory."""
        try:
            project_root = self._project_context.get('project_dir')
            if not project_root:
                logger.error(
                    f"MTC ({self._sequence_id}): Project directory not set. Cannot write file '{relative_filename}'.")
                return  # Or raise error

            abs_file_path = os.path.abspath(os.path.join(project_root, relative_filename))
            if not abs_file_path.startswith(os.path.abspath(project_root)):
                logger.error(
                    f"MTC ({self._sequence_id}): Security risk! Attempt to write file '{relative_filename}' outside project root. Blocked.")
                return  # Or raise error

            os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
            with open(abs_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"MTC ({self._sequence_id}): Successfully wrote assembled file: {abs_file_path}")
        except Exception as e_write:
            logger.error(f"MTC ({self._sequence_id}): Error writing assembled file '{relative_filename}': {e_write}",
                         exc_info=True)

    def _finalize_micro_task_sequence(self):
        """Finalizes the micro-task sequence and reports results."""
        logger.info(f"MTC ({self._sequence_id}): Finalizing micro-task sequence.")
        self._current_phase = MicroTaskPhase.COMPLETED

        successful_files = [f.filename for f in self._file_assembly_map.values() if f.is_complete and f.assembled_code]
        failed_specs = [s.function_name for s in self._function_specs_list if s.status == MicroTaskPhase.ERROR]

        if not failed_specs and len(successful_files) == len(self._file_assembly_map):
            summary_msg = f"[System: Micro-task generation completed successfully! {len(successful_files)} file(s) created/updated: {', '.join(successful_files)}]"
            self._emit_status_update(f"✅ Micro-tasks complete: {len(successful_files)} files.", "#98c379", False)
        else:
            summary_msg = f"[System: Micro-task generation finished with issues. {len(successful_files)}/{len(self._file_assembly_map)} files assembled. "
            if failed_specs:
                summary_msg += f"Failed functions: {', '.join(failed_specs[:3])}{'...' if len(failed_specs) > 3 else ''}."
            self._emit_status_update(f"⚠️ Micro-tasks finished with issues. Check generated files.", "#e5c07b", False)

        self._emit_chat_message_to_ui(summary_msg, is_error=bool(failed_specs))
        self._log_communication("SEQ_COMPLETE",
                                f"Files assembled: {len(successful_files)}. Failed functions: {len(failed_specs)}.")
        self._reset_sequence_state()

    @Slot(str, str)
    def _handle_llm_error(self, request_id: str, error_message: str):
        if not self._sequence_id or self._active_llm_request_id != request_id:
            return

        logger.error(f"MTC ({self._sequence_id}): LLM error for ReqID '{request_id}': {error_message}")
        self._active_llm_request_id = None  # Clear active request

        current_phase_at_error = self._current_phase
        spec_id_at_error = None

        # Find if the error was for a specific function spec
        for spec in self._function_specs_list:
            if spec.request_id == request_id:
                spec_id_at_error = spec.id
                spec.status = MicroTaskPhase.ERROR
                spec.error_message = error_message
                logger.error(
                    f"MTC ({self._sequence_id}): LLM error during phase {current_phase_at_error.name} for function '{spec.function_name}'.")
                self._emit_chat_message_to_ui(
                    f"[System Error: AI failed while working on function `{spec.function_name}`: {error_message}]",
                    is_error=True)
                # Decide if we retry this spec or halt
                if spec.generation_attempts < self.MAX_GENERATION_ATTEMPTS_PER_FUNCTION:
                    asyncio.create_task(self._request_function_generation(spec))  # Retry
                    return  # Don't reset the whole sequence yet
                else:
                    logger.error(
                        f"MTC ({self._sequence_id}): Max retries for '{spec.function_name}' reached after LLM error.")
                break  # Found the spec, stop iterating

        if current_phase_at_error == MicroTaskPhase.INITIAL_DECOMPOSITION:
            self._emit_chat_message_to_ui(
                f"[System Error: AI failed during initial task decomposition: {error_message}]", is_error=True)

        # If error was critical or max retries reached for a function, reset the sequence
        self._reset_sequence_state(error_occurred=True,
                                   message=f"LLM error in phase {current_phase_at_error.name}: {error_message[:60]}")

    def _reset_sequence_state(self, error_occurred: bool = False, message: Optional[str] = None):
        logger.info(
            f"MTC ({self._sequence_id or 'N/A'}): Resetting sequence state. Error: {error_occurred}. Message: {message}")
        self._current_phase = MicroTaskPhase.IDLE
        self._sequence_id = None
        self._original_query = None
        self._project_context.clear()
        self._function_specs_list.clear()
        self._file_assembly_map.clear()
        self._current_spec_index_being_processed = 0
        self._active_llm_request_id = None
        self._functions_generated = 0
        self._total_functions_planned = 0

        self._event_bus.uiInputBarBusyStateChanged.emit(False)
        if error_occurred:
            self._emit_status_update(f"❌ Micro-task sequence failed: {message or 'Unknown error'}", "#e06c75", False)
        else:  # If reset without explicit error (e.g. user cancellation - not yet implemented)
            self._emit_status_update("Micro-task sequence ended.", "#abb2bf", True, 3000)

    def _log_communication(self, stage: str, message: str):
        if self._llm_comm_logger and self._sequence_id:
            self._llm_comm_logger.log_message(f"MTC:{self._sequence_id}:{stage.upper()}", message)

    def _emit_status_update(self, message: str, color: str, is_temporary: bool, duration_ms: int = 0):
        self._event_bus.uiStatusUpdateGlobal.emit(message, color, is_temporary, duration_ms)

    def _emit_chat_message_to_ui(self, text: str, is_error: bool = False):
        project_id = self._project_context.get('project_id')
        session_id = self._project_context.get('session_id')
        if project_id and session_id:
            role = ERROR_ROLE if is_error else SYSTEM_ROLE
            msg_id = f"mtc_msg_{self._sequence_id or 'unknown'}_{uuid.uuid4().hex[:4]}"
            chat_msg = ChatMessage(id=msg_id, role=role, parts=[text])  # type: ignore
            self._event_bus.newMessageAddedToHistory.emit(project_id, session_id, chat_msg)

    def is_busy(self) -> bool:
        return self._current_phase != MicroTaskPhase.IDLE