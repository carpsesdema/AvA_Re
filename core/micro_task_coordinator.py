# app/core/micro_task_coordinator.py
import asyncio
import logging
import os
import uuid
import re
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import ast

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    from models.chat_message import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from models.message_enums import MessageLoadingState
    from llm.backend_coordinator import BackendCoordinator
    from services.llm_communication_logger import LlmCommunicationLogger
    from core.code_output_processor import CodeOutputProcessor, CodeQualityLevel
    from core.python_code_assembler import PythonCodeAssembler
    from core.dependency_resolver import DependencyResolver
    from utils import constants
    from llm import prompts as llm_prompts
except ImportError as e_mtc:
    logging.getLogger(__name__).critical(f"MicroTaskCoordinator: Critical import error: {e_mtc}", exc_info=True)
    # Fallback types
    EventBus = type("EventBus", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})
    MessageLoadingState = type("MessageLoadingState", (object,), {})
    BackendCoordinator = type("BackendCoordinator", (object,), {})
    LlmCommunicationLogger = type("LlmCommunicationLogger", (object,), {})
    CodeOutputProcessor = type("CodeOutputProcessor", (object,), {})
    CodeQualityLevel = type("CodeQualityLevel", (object,), {})
    PythonCodeAssembler = type("PythonCodeAssembler", (object,), {})
    DependencyResolver = type("DependencyResolver", (object,), {})
    constants = type("constants", (object,), {})
    llm_prompts = type("llm_prompts", (object,), {})
    USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "model", "system", "error"
    raise

logger = logging.getLogger(__name__)


class MicroTaskPhase(Enum):
    IDLE = auto()
    INITIAL_DECOMPOSITION = auto()
    ATOMIC_TASK_GENERATION = auto()
    DEPENDENCY_ANALYSIS = auto()
    INTELLIGENT_ASSEMBLY = auto()
    INTEGRATION_GUIDANCE = auto()
    FINAL_VALIDATION = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class AtomicTask:
    """Represents a single, focused coding task - smaller than functions"""
    # Required fields (no defaults)
    task_type: str  # "function", "method", "class_def", "import", "constant", "property"
    name: str  # Function/method/class name
    description: str

    # Optional fields (with defaults)
    id: str = field(default_factory=lambda: f"atomic_{uuid.uuid4().hex[:8]}")
    parent_context: Optional[str] = None  # Parent class if it's a method
    target_file: str = ""

    # Code generation details
    signature: Optional[str] = None  # Function signature or class definition
    docstring_requirements: str = ""
    dependencies: List[str] = field(default_factory=list)  # Other atomic tasks this depends on
    estimated_complexity: int = 1  # 1-5 scale

    # Generation state
    generated_code: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    generation_attempts: int = 0
    status: MicroTaskPhase = MicroTaskPhase.IDLE
    request_id: Optional[str] = None


@dataclass
class FileAssemblyPlan:
    """Detailed plan for assembling atomic tasks into a complete file"""
    filename: str
    module_purpose: str
    atomic_tasks: List[AtomicTask] = field(default_factory=list)

    # File structure elements
    required_imports: Set[str] = field(default_factory=set)
    constants: List[AtomicTask] = field(default_factory=list)
    classes: Dict[str, List[AtomicTask]] = field(default_factory=dict)  # class_name -> methods
    standalone_functions: List[AtomicTask] = field(default_factory=list)

    # Assembly state
    integration_requirements: List[str] = field(default_factory=list)
    assembly_complete: bool = False
    final_code: Optional[str] = None


class MicroTaskCoordinator(QObject):
    """
    Enhanced coordinator for atomic-level code generation with intelligent assembly
    """

    MAX_GENERATION_ATTEMPTS_PER_TASK = 3

    def __init__(self,
                 backend_coordinator: BackendCoordinator,
                 event_bus: EventBus,
                 llm_comm_logger: Optional[LlmCommunicationLogger],
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(backend_coordinator, BackendCoordinator):
            raise TypeError("MicroTaskCoordinator requires a valid BackendCoordinator.")
        if not isinstance(event_bus, EventBus):
            raise TypeError("MicroTaskCoordinator requires a valid EventBus.")

        self._backend_coordinator = backend_coordinator
        self._event_bus = event_bus
        self._llm_comm_logger = llm_comm_logger
        self._code_processor = CodeOutputProcessor()
        self._assembler = PythonCodeAssembler(backend_coordinator, event_bus)
        self._dependency_resolver = DependencyResolver()

        # Enhanced sequence state
        self._current_phase = MicroTaskPhase.IDLE
        self._sequence_id: Optional[str] = None
        self._original_query: Optional[str] = None
        self._project_context: Dict[str, Any] = {}

        # Atomic task management
        self._atomic_tasks: List[AtomicTask] = []
        self._file_assembly_plans: Dict[str, FileAssemblyPlan] = {}
        self._current_task_index: int = 0
        self._active_llm_request_id: Optional[str] = None

        # Generation tracking
        self._tasks_completed: int = 0
        self._total_tasks_planned: int = 0

        self._connect_event_bus_handlers()
        logger.info("Enhanced MicroTaskCoordinator initialized with intelligent assembly.")

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
        """Starts the enhanced micro-task generation sequence"""
        if self._current_phase != MicroTaskPhase.IDLE:
            logger.warning(f"MTC: Sequence already active. Ignoring new request.")
            self._emit_status_update("Micro-task generation already in progress.", "#e5c07b", True, 3000)
            return False

        self._reset_sequence_state()
        self._sequence_id = f"mtc_seq_{uuid.uuid4().hex[:8]}"
        self._original_query = user_query
        self._project_context = {
            'project_dir': project_dir,
            'project_id': project_id,
            'session_id': session_id,
            'planning_backend': planning_backend,
            'planning_model': planning_model,
            'coding_backend': coding_backend,
            'coding_model': coding_model
        }

        logger.info(f"MTC ({self._sequence_id}): Starting enhanced sequence for: '{user_query[:50]}...'")
        self._emit_chat_message_to_ui(
            f"[System: Starting intelligent micro-task generation for '{user_query[:40]}...']"
            f"\n🧠 **Phase 1**: AI Architect planning atomic tasks..."
        )
        self._event_bus.uiInputBarBusyStateChanged.emit(True)

        self._current_phase = MicroTaskPhase.INITIAL_DECOMPOSITION
        asyncio.create_task(self._request_atomic_task_decomposition())
        return True

    async def _request_atomic_task_decomposition(self):
        """Enhanced decomposition into atomic-level tasks"""
        if not self._original_query or not self._sequence_id: return

        self._active_llm_request_id = f"mtc_decompose_{self._sequence_id}"
        self._emit_status_update(f"🧠 AI Architect breaking down task into atomic components...", "#61afef", False)

        # Enhanced prompt for atomic-level decomposition
        decomposition_prompt = self._build_atomic_decomposition_prompt()
        history = [ChatMessage(role=USER_ROLE, parts=[decomposition_prompt])]

        self._backend_coordinator.start_llm_streaming_task(
            request_id=self._active_llm_request_id,
            target_backend_id=self._project_context['planning_backend'],
            history_to_send=history,
            is_modification_response_expected=True,
            options={"temperature": 0.1},  # Very low for precise decomposition
            request_metadata={
                "purpose": "atomic_task_decomposition",
                "sequence_id": self._sequence_id,
                "project_id": self._project_context.get('project_id'),
                "session_id": self._project_context.get('session_id')
            }
        )

    def _build_atomic_decomposition_prompt(self) -> str:
        """Builds the enhanced prompt for atomic-level task decomposition"""
        return f"""You are an expert Python architect. Break down this request into ATOMIC coding tasks.

**User Request**: {self._original_query}

**ATOMIC TASK BREAKDOWN REQUIRED**:
Each atomic task should be:
- A single function, method, class definition, or code unit
- Implementable in 10-50 lines of code
- Testable in isolation
- Have clear inputs and outputs

**OUTPUT FORMAT** (JSON-like structure):
```
FILES_TO_CREATE: ["file1.py", "file2.py"]

### file1.py
PURPOSE: Brief description of file's role
MODULE_DOCSTRING: What this module does

ATOMIC_TASKS:
[
  {{
    "task_type": "class_def",
    "name": "ClassName", 
    "description": "What this class does",
    "parent_context": null,
    "signature": "class ClassName:",
    "docstring_requirements": "What the class docstring should cover",
    "dependencies": [],
    "estimated_complexity": 2
  }},
  {{
    "task_type": "method",
    "name": "__init__",
    "description": "Initialize the class",
    "parent_context": "ClassName",
    "signature": "def __init__(self, param1: str, param2: int)",
    "docstring_requirements": "Document parameters and purpose",
    "dependencies": [],
    "estimated_complexity": 1
  }},
  {{
    "task_type": "function",
    "name": "helper_function",
    "description": "Utility function for X",
    "parent_context": null,
    "signature": "def helper_function(data: List[str]) -> Dict[str, Any]",
    "docstring_requirements": "Document args, returns, raises",
    "dependencies": ["ClassName"],
    "estimated_complexity": 3
  }}
]
```

Focus on professional Python patterns. Be specific about signatures and dependencies."""

    @Slot(str, ChatMessage, dict)
    def _handle_llm_completion(self, request_id: str, message: ChatMessage, metadata: dict):
        if not self._sequence_id or metadata.get("sequence_id") != self._sequence_id:
            return

        purpose = metadata.get("purpose")
        self._active_llm_request_id = None

        if purpose == "atomic_task_decomposition":
            self._process_atomic_decomposition_response(message.text)
        elif purpose == "atomic_code_generation":
            task_id = metadata.get("atomic_task_id")
            self._process_generated_atomic_code(task_id, message.text)
        elif purpose == "intelligent_assembly":
            filename = metadata.get("filename")
            self._process_assembly_response(filename, message.text)
        elif purpose == "integration_guidance":
            self._process_integration_guidance(message.text)

    def _process_atomic_decomposition_response(self, llm_response_text: str):
        """Process the atomic task decomposition from the planner LLM"""
        try:
            self._atomic_tasks, self._file_assembly_plans = self._parse_atomic_tasks_from_llm(llm_response_text)

            if not self._atomic_tasks:
                raise ValueError("No atomic tasks parsed from LLM response")

            self._total_tasks_planned = len(self._atomic_tasks)
            self._tasks_completed = 0

            logger.info(
                f"MTC ({self._sequence_id}): Decomposed into {self._total_tasks_planned} atomic tasks across {len(self._file_assembly_plans)} files")

            self._emit_chat_message_to_ui(
                f"[System: ✅ **Phase 1 Complete**]\n"
                f"🔬 Decomposed into **{self._total_tasks_planned}** atomic tasks across **{len(self._file_assembly_plans)}** files\n"
                f"⚡ **Phase 2**: Generating atomic code components..."
            )

            # Resolve dependencies
            self._dependency_resolver.analyze_and_order_tasks(self._atomic_tasks)

            self._current_phase = MicroTaskPhase.ATOMIC_TASK_GENERATION
            self._current_task_index = 0
            asyncio.create_task(self._generate_next_atomic_task())

        except Exception as e:
            logger.error(f"MTC ({self._sequence_id}): Failed to process decomposition: {e}", exc_info=True)
            self._emit_chat_message_to_ui(f"[System Error: Could not understand task breakdown: {e}]", is_error=True)
            self._reset_sequence_state(error_occurred=True)

    def _parse_atomic_tasks_from_llm(self, response_text: str) -> Tuple[List[AtomicTask], Dict[str, FileAssemblyPlan]]:
        """Parse the structured atomic task response"""
        atomic_tasks = []
        file_plans = {}

        # Extract files to create
        files_match = re.search(r"FILES_TO_CREATE:\s*(\[.*?])", response_text, re.DOTALL)
        if not files_match:
            raise ValueError("No FILES_TO_CREATE section found")

        try:
            files_list = ast.literal_eval(files_match.group(1))
        except:
            raise ValueError("Invalid FILES_TO_CREATE format")

        # Process each file section
        file_sections = re.split(r'\n###\s+', '\n' + response_text)[1:]

        for section in file_sections:
            lines = section.split('\n')
            if not lines:
                continue

            filename = lines[0].strip()
            if filename not in files_list:
                continue

            # Extract file metadata
            purpose = self._extract_section_value(section, "PURPOSE")
            module_docstring = self._extract_section_value(section, "MODULE_DOCSTRING")

            # Extract atomic tasks JSON
            tasks_match = re.search(r"ATOMIC_TASKS:\s*(\[.*?])", section, re.DOTALL)
            if not tasks_match:
                continue

            try:
                tasks_data = ast.literal_eval(tasks_match.group(1))
                file_atomic_tasks = []

                for task_data in tasks_data:
                    atomic_task = AtomicTask(
                        task_type=task_data.get("task_type", "function"),
                        name=task_data.get("name", "unknown"),
                        description=task_data.get("description", ""),
                        parent_context=task_data.get("parent_context"),
                        target_file=filename,
                        signature=task_data.get("signature", ""),
                        docstring_requirements=task_data.get("docstring_requirements", ""),
                        dependencies=task_data.get("dependencies", []),
                        estimated_complexity=task_data.get("estimated_complexity", 1)
                    )
                    atomic_tasks.append(atomic_task)
                    file_atomic_tasks.append(atomic_task)

                # Create file assembly plan
                file_plans[filename] = FileAssemblyPlan(
                    filename=filename,
                    module_purpose=purpose or f"Implementation for {filename}",
                    atomic_tasks=file_atomic_tasks
                )

            except Exception as e:
                logger.warning(f"Failed to parse atomic tasks for {filename}: {e}")
                continue

        return atomic_tasks, file_plans

    def _extract_section_value(self, section: str, key: str) -> Optional[str]:
        """Extract a value from a section"""
        pattern = rf"{key}:\s*(.+)"
        match = re.search(pattern, section, re.IGNORECASE)
        return match.group(1).strip() if match else None

    async def _generate_next_atomic_task(self):
        """Generate code for the next atomic task"""
        if self._current_task_index >= len(self._atomic_tasks):
            logger.info(f"MTC ({self._sequence_id}): All atomic tasks generated. Starting intelligent assembly.")
            self._current_phase = MicroTaskPhase.INTELLIGENT_ASSEMBLY
            asyncio.create_task(self._start_intelligent_assembly())
            return

        current_task = self._atomic_tasks[self._current_task_index]

        # Check dependencies
        if not self._dependency_resolver.are_dependencies_met(current_task, self._atomic_tasks):
            logger.info(
                f"MTC ({self._sequence_id}): Dependencies not met for {current_task.name}, finding next available task.")
            self._current_task_index += 1
            await self._generate_next_atomic_task()
            return

        await self._generate_atomic_task_code(current_task)

    async def _generate_atomic_task_code(self, task: AtomicTask):
        """Generate code for a single atomic task"""
        task.request_id = f"atomic_{self._sequence_id}_{task.id}"
        task.generation_attempts += 1
        task.status = MicroTaskPhase.ATOMIC_TASK_GENERATION

        self._emit_status_update(
            f"⚡ Generating {task.task_type}: {task.name} ({self._tasks_completed + 1}/{self._total_tasks_planned})",
            "#c678dd", False
        )

        # Build context from dependencies
        dependency_context = self._build_dependency_context(task)

        # Create focused prompt for this atomic task
        generation_prompt = self._build_atomic_generation_prompt(task, dependency_context)
        history = [ChatMessage(role=USER_ROLE, parts=[generation_prompt])]

        self._backend_coordinator.start_llm_streaming_task(
            request_id=task.request_id,
            target_backend_id=self._project_context['coding_backend'],
            history_to_send=history,
            is_modification_response_expected=True,
            options={"temperature": 0.05},  # Very low for precise code
            request_metadata={
                "purpose": "atomic_code_generation",
                "sequence_id": self._sequence_id,
                "atomic_task_id": task.id,
                "project_id": self._project_context.get('project_id'),
                "session_id": self._project_context.get('session_id')
            }
        )

    def _build_dependency_context(self, task: AtomicTask) -> str:
        """Build context from completed dependency tasks"""
        context_parts = []
        for dep_name in task.dependencies:
            dep_task = next((t for t in self._atomic_tasks
                             if t.name == dep_name and t.generated_code), None)
            if dep_task:
                context_parts.append(f"# {dep_name} (already implemented):\n{dep_task.generated_code}")

        return "\n\n".join(context_parts) if context_parts else "# No dependencies available yet"

    def _build_atomic_generation_prompt(self, task: AtomicTask, dependency_context: str) -> str:
        """Build a focused prompt for generating a single atomic task"""
        return f"""Generate ONLY the Python code for this specific atomic task:

**Task Type**: {task.task_type}
**Name**: {task.name}
**Description**: {task.description}
**Signature**: {task.signature}
**Parent Context**: {task.parent_context or 'Module level'}
**Target File**: {task.target_file}

**Docstring Requirements**: {task.docstring_requirements}

**Available Dependencies**:
{dependency_context}

**Quality Requirements**:
- Include proper type hints
- Write comprehensive docstring as specified
- Add appropriate error handling
- Include logging where relevant
- Follow PEP 8 standards

**Output**: ONLY the code for this specific {task.task_type}. No explanations, no markdown blocks, just the Python code."""

    def _process_generated_atomic_code(self, task_id: Optional[str], generated_code_text: str):
        """Process the generated code for an atomic task"""
        if not task_id:
            logger.error(f"MTC ({self._sequence_id}): No task_id provided for generated code")
            return

        task = next((t for t in self._atomic_tasks if t.id == task_id), None)
        if not task:
            logger.error(f"MTC ({self._sequence_id}): Task {task_id} not found")
            return

        # Extract and validate the code
        extracted_code, quality, notes = self._code_processor.process_llm_response(
            generated_code_text, f"{task.name}.py"
        )

        task.generated_code = extracted_code
        task.validation_errors.clear()

        if extracted_code:
            is_valid, syntax_error = self._code_processor._is_valid_python_code(extracted_code)
            if is_valid:
                task.status = MicroTaskPhase.COMPLETED
                self._tasks_completed += 1

                logger.info(
                    f"MTC ({self._sequence_id}): ✅ Generated {task.task_type} '{task.name}' (Quality: {quality.name if quality else 'N/A'})")

                # Move to next task
                self._current_task_index += 1
                asyncio.create_task(self._generate_next_atomic_task())
                return
            else:
                task.validation_errors.append(f"Syntax error: {syntax_error}")
        else:
            task.validation_errors.append("Failed to extract code from LLM response")

        # Handle retry logic
        if task.generation_attempts < self.MAX_GENERATION_ATTEMPTS_PER_TASK:
            logger.warning(f"MTC ({self._sequence_id}): Retrying {task.name} (attempt {task.generation_attempts + 1})")
            asyncio.create_task(self._generate_atomic_task_code(task))
        else:
            logger.error(
                f"MTC ({self._sequence_id}): Failed to generate {task.name} after {task.generation_attempts} attempts")
            task.status = MicroTaskPhase.ERROR
            self._current_task_index += 1
            asyncio.create_task(self._generate_next_atomic_task())

    async def _start_intelligent_assembly(self):
        """Start the intelligent assembly phase"""
        self._emit_chat_message_to_ui(
            f"[System: ✅ **Phase 2 Complete**]\n"
            f"🔧 **Phase 3**: Intelligent Assembly - AI Architect organizing code into production files..."
        )

        for filename, plan in self._file_assembly_plans.items():
            await self._assemble_file_intelligently(plan)

        self._finalize_sequence()

    async def _assemble_file_intelligently(self, plan: FileAssemblyPlan):
        """Use the intelligent assembler to create a production-ready file"""
        self._emit_status_update(f"🔧 Assembling {plan.filename}...", "#56b6c2", False)

        # Gather completed atomic tasks for this file
        completed_tasks = [task for task in plan.atomic_tasks
                           if task.generated_code and task.status == MicroTaskPhase.COMPLETED]

        if not completed_tasks:
            logger.warning(f"MTC ({self._sequence_id}): No completed tasks for {plan.filename}")
            return

        try:
            # Use the intelligent assembler
            assembled_code = await self._assembler.assemble_production_file(
                filename=plan.filename,
                module_purpose=plan.module_purpose,
                atomic_tasks=completed_tasks,
                sequence_id=self._sequence_id
            )

            if assembled_code:
                plan.final_code = assembled_code
                plan.assembly_complete = True

                # Write to disk and notify UI
                self._write_assembled_file_to_disk(plan.filename, assembled_code)
                self._event_bus.modificationFileReadyForDisplay.emit(plan.filename, assembled_code)

                logger.info(f"MTC ({self._sequence_id}): ✅ Successfully assembled {plan.filename}")
                self._emit_chat_message_to_ui(f"✅ **{plan.filename}** assembled and ready for review")
            else:
                logger.error(f"MTC ({self._sequence_id}): Assembly failed for {plan.filename}")

        except Exception as e:
            logger.error(f"MTC ({self._sequence_id}): Error assembling {plan.filename}: {e}", exc_info=True)

    def _write_assembled_file_to_disk(self, relative_filename: str, content: str):
        """Write the assembled file to the project directory"""
        try:
            project_root = self._project_context.get('project_dir')
            if not project_root:
                logger.error(f"MTC ({self._sequence_id}): No project directory set")
                return

            abs_file_path = os.path.abspath(os.path.join(project_root, relative_filename))

            # Security check
            if not abs_file_path.startswith(os.path.abspath(project_root)):
                logger.error(f"MTC ({self._sequence_id}): Security risk - file outside project root")
                return

            os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)

            with open(abs_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"MTC ({self._sequence_id}): Wrote {abs_file_path}")

        except Exception as e:
            logger.error(f"MTC ({self._sequence_id}): Error writing {relative_filename}: {e}", exc_info=True)

    def _finalize_sequence(self):
        """Finalize the micro-task sequence"""
        successful_files = [plan.filename for plan in self._file_assembly_plans.values()
                            if plan.assembly_complete]
        failed_tasks = [task.name for task in self._atomic_tasks if task.status == MicroTaskPhase.ERROR]

        if len(successful_files) == len(self._file_assembly_plans) and not failed_tasks:
            summary = f"[System: ✅ **Micro-Task Generation Complete!**]\n🎉 Successfully generated **{len(successful_files)}** production-ready files:\n"
            for filename in successful_files:
                summary += f"  • **{filename}**\n"
            summary += "\n📝 Files are ready for review in the Code Viewer!"

            self._emit_status_update(f"✅ {len(successful_files)} files generated successfully!", "#98c379", False)
        else:
            summary = f"[System: ⚠️ **Micro-Task Generation Completed with Issues**]\n"
            summary += f"✅ Generated: {len(successful_files)}/{len(self._file_assembly_plans)} files\n"
            if failed_tasks:
                summary += f"❌ Failed tasks: {', '.join(failed_tasks[:3])}{'...' if len(failed_tasks) > 3 else ''}"

            self._emit_status_update(f"⚠️ {len(successful_files)}/{len(self._file_assembly_plans)} files completed",
                                     "#e5c07b", False)

        self._emit_chat_message_to_ui(summary, is_error=bool(failed_tasks))
        self._log_communication("SEQ_COMPLETE", f"Files: {len(successful_files)}, Failed tasks: {len(failed_tasks)}")
        self._reset_sequence_state()

    @Slot(str, str)
    def _handle_llm_error(self, request_id: str, error_message: str):
        """Handle LLM errors during the sequence"""
        if not self._sequence_id:
            return

        # Find the task that failed
        failed_task = None
        for task in self._atomic_tasks:
            if task.request_id == request_id:
                failed_task = task
                break

        if failed_task:
            failed_task.status = MicroTaskPhase.ERROR
            failed_task.validation_errors.append(f"LLM Error: {error_message}")
            logger.error(f"MTC ({self._sequence_id}): LLM error for {failed_task.name}: {error_message}")

            if failed_task.generation_attempts < self.MAX_GENERATION_ATTEMPTS_PER_TASK:
                asyncio.create_task(self._generate_atomic_task_code(failed_task))
            else:
                self._current_task_index += 1
                asyncio.create_task(self._generate_next_atomic_task())
        else:
            logger.error(f"MTC ({self._sequence_id}): LLM error for unknown request {request_id}: {error_message}")
            self._reset_sequence_state(error_occurred=True, message=f"LLM error: {error_message}")

    def _reset_sequence_state(self, error_occurred: bool = False, message: Optional[str] = None):
        """Reset the coordinator state"""
        logger.info(f"MTC ({self._sequence_id or 'N/A'}): Resetting state. Error: {error_occurred}")

        self._current_phase = MicroTaskPhase.IDLE
        self._sequence_id = None
        self._original_query = None
        self._project_context.clear()
        self._atomic_tasks.clear()
        self._file_assembly_plans.clear()
        self._current_task_index = 0
        self._active_llm_request_id = None
        self._tasks_completed = 0
        self._total_tasks_planned = 0

        self._event_bus.uiInputBarBusyStateChanged.emit(False)

        if error_occurred:
            self._emit_status_update(f"❌ Micro-task sequence failed: {message or 'Unknown error'}", "#e06c75", False)
        else:
            self._emit_status_update("Ready for new micro-task generation", "#abb2bf", True, 3000)

    def _log_communication(self, stage: str, message: str):
        """Log communication for debugging"""
        if self._llm_comm_logger and self._sequence_id:
            self._llm_comm_logger.log_message(f"MTC:{self._sequence_id}:{stage}", message)

    def _emit_status_update(self, message: str, color: str, is_temporary: bool, duration_ms: int = 0):
        """Emit status update to UI"""
        self._event_bus.uiStatusUpdateGlobal.emit(message, color, is_temporary, duration_ms)

    def _emit_chat_message_to_ui(self, text: str, is_error: bool = False):
        """Emit chat message to UI"""
        project_id = self._project_context.get('project_id')
        session_id = self._project_context.get('session_id')
        if project_id and session_id:
            role = ERROR_ROLE if is_error else SYSTEM_ROLE
            msg_id = f"mtc_msg_{self._sequence_id or 'unknown'}_{uuid.uuid4().hex[:4]}"
            chat_msg = ChatMessage(id=msg_id, role=role, parts=[text])
            self._event_bus.newMessageAddedToHistory.emit(project_id, session_id, chat_msg)

    def is_busy(self) -> bool:
        """Check if coordinator is busy"""
        return self._current_phase != MicroTaskPhase.IDLE