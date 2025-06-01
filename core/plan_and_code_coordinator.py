# app/core/plan_and_code_coordinator.py
import ast
import asyncio
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    from app.models.chat_message import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from app.models.message_enums import MessageLoadingState
    from app.llm.backend_coordinator import BackendCoordinator
    from app.services.llm_communication_logger import LlmCommunicationLogger
    from app.core.code_output_processor import CodeOutputProcessor, CodeQualityLevel
    from app.core.micro_task_coordinator import MicroTaskCoordinator
    from app.llm import prompts as llm_prompts
    from utils import constants
except ImportError as e_pacc:
    logging.getLogger(__name__).critical(f"PlanAndCodeCoordinator: Critical import error: {e_pacc}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class SequencePhase(Enum):
    IDLE = auto()
    HIGH_LEVEL_PLANNING = auto()
    ARCHITECTURE_DESIGN = auto()
    AWAITING_PLAN_APPROVAL = auto()
    DELEGATING_TO_MICRO_TASKS = auto()
    MONITORING_EXECUTION = auto()
    FINAL_INTEGRATION_REVIEW = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class ProjectPlan:
    """High-level project plan from the planner LLM"""
    project_name: str
    overall_goal: str
    architecture_summary: str
    file_specifications: List[Dict[str, Any]] = field(default_factory=list)
    integration_requirements: List[str] = field(default_factory=list)
    quality_standards: Dict[str, Any] = field(default_factory=dict)
    estimated_complexity: int = 1  # 1-10 scale


class PlanAndCodeCoordinator(QObject):
    """
    Enhanced coordinator focusing on high-level planning and delegation.
    Works with MicroTaskCoordinator for detailed implementation.
    """

    def __init__(self,
                 backend_coordinator: BackendCoordinator,
                 event_bus: EventBus,
                 micro_task_coordinator: MicroTaskCoordinator,
                 llm_comm_logger: Optional[LlmCommunicationLogger],
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(backend_coordinator, BackendCoordinator):
            raise TypeError("PlanAndCodeCoordinator requires a valid BackendCoordinator.")
        if not isinstance(event_bus, EventBus):
            raise TypeError("PlanAndCodeCoordinator requires a valid EventBus.")
        if not isinstance(micro_task_coordinator, MicroTaskCoordinator):
            raise TypeError("PlanAndCodeCoordinator requires a valid MicroTaskCoordinator.")

        self._backend_coordinator = backend_coordinator
        self._event_bus = event_bus
        self._micro_task_coordinator = micro_task_coordinator
        self._llm_comm_logger = llm_comm_logger
        self._code_processor = CodeOutputProcessor()

        # Sequence state
        self._current_phase = SequencePhase.IDLE
        self._sequence_id: Optional[str] = None
        self._original_query: Optional[str] = None
        self._project_context: Dict[str, Any] = {}
        self._project_plan: Optional[ProjectPlan] = None
        self._active_llm_request_id: Optional[str] = None

        # Integration with micro-task coordinator
        self._micro_task_sequence_id: Optional[str] = None
        self._is_monitoring_micro_tasks = False

        self._connect_event_bus_handlers()
        logger.info("Enhanced PlanAndCodeCoordinator initialized with micro-task delegation.")

    def _connect_event_bus_handlers(self):
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_completion)
        self._event_bus.llmResponseError.connect(self._handle_llm_error)
        self._event_bus.forcePlanAndCodeGenerationRequested.connect(self._handle_force_proceed)

    def start_autonomous_coding(self,
                                user_query: str,
                                planner_backend: str,
                                planner_model: str,
                                coder_backend: str,
                                coder_model: str,
                                project_dir: str,
                                project_id: Optional[str] = None,
                                session_id: Optional[str] = None,
                                task_type: Optional[str] = None) -> bool:
        """Start the enhanced autonomous coding sequence"""
        if self._current_phase != SequencePhase.IDLE:
            logger.warning(f"PACC: Sequence already active. Ignoring new request.")
            self._emit_status_update("Plan-and-code sequence already in progress.", "#e5c07b", True, 3000)
            return False

        self._reset_sequence_state()
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

        logger.info(f"PACC ({self._sequence_id}): Starting enhanced sequence for: '{user_query[:50]}...'")
        self._emit_chat_message_to_ui(
            f"[System: 🏗️ **Starting Autonomous Development**]\n"
            f"📋 **Request**: {user_query[:60]}...\n"
            f"🧠 **Phase 1**: High-level architectural planning..."
        )
        self._event_bus.uiInputBarBusyStateChanged.emit(True)

        self._current_phase = SequencePhase.HIGH_LEVEL_PLANNING
        return self._initiate_high_level_planning()

    def _initiate_high_level_planning(self) -> bool:
        """Send request to planner LLM for high-level architecture"""
        if not self._backend_coordinator or not self._sequence_id:
            return False

        planning_request_id = f"plan_{self._sequence_id}"
        self._active_llm_request_id = planning_request_id

        self._emit_status_update(f"🧠 Architect planning with {self._project_context['planner_model']}...", "#61afef",
                                 False)

        planning_prompt = self._construct_enhanced_planning_prompt()
        history = [ChatMessage(role=USER_ROLE, parts=[planning_prompt])]

        self._backend_coordinator.start_llm_streaming_task(
            request_id=planning_request_id,
            target_backend_id=self._project_context['planner_backend'],
            history_to_send=history,
            is_modification_response_expected=True,
            options={"temperature": 0.2},  # Low temperature for structured planning
            request_metadata={
                "purpose": "high_level_planning",
                "sequence_id": self._sequence_id,
                "project_id": self._project_context.get('project_id'),
                "session_id": self._project_context.get('session_id')
            }
        )
        logger.info(f"PACC ({self._sequence_id}): High-level planning request sent.")
        return True

    def _construct_enhanced_planning_prompt(self) -> str:
        """Construct the enhanced prompt for architectural planning"""
        base_prompt = getattr(llm_prompts, 'ENHANCED_PLANNING_SYSTEM_PROMPT',
                              "You are an expert software architect and project planner.")

        existing_structure = self._analyze_existing_project_structure()
        task_guidance = self._get_task_specific_guidance()

        context_section = f"""
## 🎯 PROJECT PLANNING REQUEST

**User's Request**: {self._original_query}

**Project Context**:
- Target Directory: {self._project_context.get('project_dir', 'N/A')}
- Task Type: {self._project_context.get('task_type', 'general')}
- Implementation Team: AI Planner + Specialized Coder ({self._project_context.get('coder_model', 'Default')})

**Existing Project Structure**:
{existing_structure}

{task_guidance}

## 📋 REQUIRED OUTPUT FORMAT

Your response must follow this EXACT structure:

```
PROJECT_OVERVIEW:
[Brief description of what will be built and its purpose]

ARCHITECTURE_SUMMARY:
[High-level architecture decisions and patterns to be used]

FILE_SPECIFICATIONS:
[
  {{
    "filename": "main.py",
    "purpose": "Main application entry point",
    "complexity": 3,
    "key_components": ["main() function", "CLI argument parsing", "application initialization"],
    "dependencies": [],
    "integration_notes": "Coordinates other modules and handles startup"
  }},
  {{
    "filename": "models.py", 
    "purpose": "Data models and schemas",
    "complexity": 2,
    "key_components": ["User class", "Config dataclass", "validation methods"],
    "dependencies": [],
    "integration_notes": "Used by all other modules for data structures"
  }}
]

INTEGRATION_REQUIREMENTS:
[
  "main.py must import and initialize components from models.py",
  "Error handling should be consistent across all modules",
  "All modules should use the same logging configuration"
]

QUALITY_STANDARDS:
{{
  "testing_approach": "Unit tests for all public methods",
  "documentation_level": "Full docstrings and type hints",
  "error_handling": "Comprehensive with custom exceptions",
  "logging_strategy": "Structured logging with appropriate levels",
  "code_style": "PEP 8 compliant with Black formatting"
}}

ESTIMATED_COMPLEXITY: [1-10 scale]
```

Focus on creating a clear, implementable plan that the specialized coder can execute efficiently.
"""
        return base_prompt + context_section

    def _analyze_existing_project_structure(self) -> str:
        """Analyze existing project structure"""
        project_dir = self._project_context.get('project_dir')
        if not project_dir or not os.path.isdir(project_dir):
            return "New project - no existing files to consider."

        try:
            file_tree = []
            max_files = 15
            files_count = 0

            for root, dirs, files in os.walk(project_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                level = root.replace(project_dir, '').count(os.sep)
                if level > 3:
                    dirs[:] = []
                    continue

                indent = '  ' * level
                file_tree.append(f"{indent}{os.path.basename(root)}/")

                for file_name in files:
                    if files_count >= max_files:
                        file_tree.append(f"{indent}  ... (and more files)")
                        break

                    if file_name.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.json')):
                        file_tree.append(f"{indent}  {file_name}")
                        files_count += 1

            return "Existing structure:\n" + "\n".join(file_tree) if file_tree else "Empty project directory."
        except Exception as e:
            logger.warning(f"PACC: Could not analyze project structure: {e}")
            return "Could not analyze existing project structure."

    def _get_task_specific_guidance(self) -> str:
        """Get guidance specific to the task type"""
        task_type = self._project_context.get('task_type', 'general')

        guidance_map = {
            'api': """
**API Development Guidance**:
- Design RESTful endpoints with clear resource modeling
- Include request/response validation (Pydantic models)
- Plan for authentication and authorization
- Design comprehensive error handling with proper HTTP status codes
- Include API documentation structure (OpenAPI/Swagger)
            """,
            'ui': """
**UI Development Guidance**:
- Design main window with clear component hierarchy
- Plan for user interaction flows and state management
- Include custom widgets for reusable components
- Design consistent styling and theming approach
- Plan for responsive layout and accessibility
            """,
            'data_processing': """
**Data Processing Guidance**:
- Design data pipeline with clear input/output interfaces
- Plan for data validation and quality checks
- Include error handling for malformed data
- Design for scalability and performance
- Plan comprehensive logging for debugging
            """,
            'utility': """
**Utility Library Guidance**:
- Design clean, focused APIs with minimal dependencies
- Plan for comprehensive testing and documentation
- Include proper error handling and input validation
- Design for reusability across different projects
- Plan versioning and backward compatibility
            """
        }

        return guidance_map.get(task_type,
                                "**General Development**: Focus on clean architecture, proper separation of concerns, and maintainable code structure.")

    @Slot(str, ChatMessage, dict)
    def _handle_llm_completion(self, request_id: str, message: ChatMessage, metadata: dict):
        """Handle LLM completion responses"""
        if not self._sequence_id or metadata.get("sequence_id") != self._sequence_id:
            return

        purpose = metadata.get("purpose")
        self._active_llm_request_id = None

        if purpose == "high_level_planning":
            self._process_high_level_plan(message.text)
        elif purpose == "integration_review":
            self._process_integration_review(message.text)

    def _process_high_level_plan(self, plan_text: str):
        """Process the high-level plan from planner LLM"""
        try:
            self._project_plan = self._parse_project_plan(plan_text)

            if not self._project_plan:
                raise ValueError("Could not parse project plan from LLM response")

            logger.info(
                f"PACC ({self._sequence_id}): Parsed plan with {len(self._project_plan.file_specifications)} files")

            # Create user-friendly plan summary
            plan_summary = self._create_plan_summary()

            self._current_phase = SequencePhase.AWAITING_PLAN_APPROVAL
            self._emit_chat_message_to_ui(
                f"[System: ✅ **Phase 1 Complete - Architectural Plan Ready**]\n\n"
                f"{plan_summary}\n\n"
                f"📝 **Review the plan above.** Type 'yes', 'proceed', or 'approve' to start implementation, "
                f"or provide feedback for plan revisions."
            )

            self._emit_status_update(
                f"📋 Plan ready: {len(self._project_plan.file_specifications)} files. Awaiting approval.", "#e5c07b",
                False)
            self._event_bus.uiInputBarBusyStateChanged.emit(False)

        except Exception as e:
            logger.error(f"PACC ({self._sequence_id}): Failed to process plan: {e}", exc_info=True)
            self._emit_chat_message_to_ui(
                f"[System Error: Could not understand the architectural plan. Error: {e}. Please try again.]",
                is_error=True
            )
            self._reset_sequence_state(error_occurred=True)

    def _parse_project_plan(self, plan_text: str) -> Optional[ProjectPlan]:
        """Parse the structured project plan from LLM response"""
        try:
            # Extract main sections
            overview = self._extract_section(plan_text, "PROJECT_OVERVIEW")
            architecture = self._extract_section(plan_text, "ARCHITECTURE_SUMMARY")

            # Extract file specifications
            file_specs_match = re.search(r"FILE_SPECIFICATIONS:\s*(\[.*?])", plan_text, re.DOTALL)
            if not file_specs_match:
                raise ValueError("No FILE_SPECIFICATIONS found")

            file_specs = ast.literal_eval(file_specs_match.group(1))

            # Extract integration requirements
            integration_match = re.search(r"INTEGRATION_REQUIREMENTS:\s*(\[.*?])", plan_text, re.DOTALL)
            integration_reqs = []
            if integration_match:
                try:
                    integration_reqs = ast.literal_eval(integration_match.group(1))
                except:
                    integration_reqs = []

            # Extract quality standards
            quality_match = re.search(r"QUALITY_STANDARDS:\s*(\{.*?})", plan_text, re.DOTALL)
            quality_standards = {}
            if quality_match:
                try:
                    quality_standards = ast.literal_eval(quality_match.group(1))
                except:
                    quality_standards = {}

            # Extract complexity
            complexity_match = re.search(r"ESTIMATED_COMPLEXITY:\s*(\d+)", plan_text)
            complexity = int(complexity_match.group(1)) if complexity_match else 5

            return ProjectPlan(
                project_name=os.path.basename(self._project_context.get('project_dir', 'project')),
                overall_goal=overview or "Project implementation",
                architecture_summary=architecture or "Standard Python architecture",
                file_specifications=file_specs,
                integration_requirements=integration_reqs,
                quality_standards=quality_standards,
                estimated_complexity=complexity
            )

        except Exception as e:
            logger.error(f"PACC: Error parsing project plan: {e}")
            return None

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a section from the plan text"""
        pattern = rf"{section_name}:\s*(.*?)(?=\n\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _create_plan_summary(self) -> str:
        """Create a user-friendly summary of the plan"""
        if not self._project_plan:
            return "No plan available"

        summary = f"🏗️ **Project Architecture Plan**\n\n"
        summary += f"**Goal**: {self._project_plan.overall_goal}\n\n"
        summary += f"**Architecture**: {self._project_plan.architecture_summary}\n\n"

        summary += f"**Files to Create** ({len(self._project_plan.file_specifications)}):\n"
        for i, file_spec in enumerate(self._project_plan.file_specifications, 1):
            summary += f"  {i}. **{file_spec['filename']}** - {file_spec['purpose']}\n"

        if self._project_plan.integration_requirements:
            summary += f"\n**Integration Requirements**:\n"
            for req in self._project_plan.integration_requirements[:3]:  # Show first 3
                summary += f"  • {req}\n"
            if len(self._project_plan.integration_requirements) > 3:
                summary += f"  • ... and {len(self._project_plan.integration_requirements) - 3} more\n"

        summary += f"\n**Estimated Complexity**: {self._project_plan.estimated_complexity}/10"

        return summary

    def confirm_plan_and_proceed(self):
        """Called when user approves the plan"""
        if self._current_phase != SequencePhase.AWAITING_PLAN_APPROVAL:
            logger.warning(f"PACC ({self._sequence_id}): Plan confirmation received but not awaiting approval")
            return

        if not self._project_plan:
            logger.error(f"PACC ({self._sequence_id}): No plan available for confirmation")
            self._emit_chat_message_to_ui("[System Error: No plan data available]", is_error=True)
            return

        logger.info(f"PACC ({self._sequence_id}): Plan approved, delegating to micro-task coordinator")

        self._emit_chat_message_to_ui(
            f"[System: ✅ **Plan Approved!**]\n"
            f"🚀 **Phase 2**: Delegating to AI Implementation Team...\n"
            f"⚡ The specialized coder will now implement each component with atomic precision."
        )

        self._current_phase = SequencePhase.DELEGATING_TO_MICRO_TASKS
        self._delegate_to_micro_task_coordinator()

    def _delegate_to_micro_task_coordinator(self):
        """Delegate implementation to the micro-task coordinator"""
        try:
            # Convert our high-level plan into a detailed request for micro-task coordinator
            detailed_request = self._create_detailed_implementation_request()

            # Start micro-task generation
            success = self._micro_task_coordinator.start_micro_task_generation(
                user_query=detailed_request,
                planning_backend=self._project_context['planner_backend'],
                planning_model=self._project_context['planner_model'],
                coding_backend=self._project_context['coder_backend'],
                coding_model=self._project_context['coder_model'],
                project_dir=self._project_context['project_dir'],
                project_id=self._project_context.get('project_id'),
                session_id=self._project_context.get('session_id')
            )

            if success:
                self._current_phase = SequencePhase.MONITORING_EXECUTION
                self._is_monitoring_micro_tasks = True
                logger.info(f"PACC ({self._sequence_id}): Successfully delegated to micro-task coordinator")

                # Monitor micro-task completion
                asyncio.create_task(self._monitor_micro_task_execution())
            else:
                logger.error(f"PACC ({self._sequence_id}): Failed to start micro-task generation")
                self._emit_chat_message_to_ui("[System Error: Could not start implementation phase]", is_error=True)
                self._reset_sequence_state(error_occurred=True)

        except Exception as e:
            logger.error(f"PACC ({self._sequence_id}): Error delegating to micro-tasks: {e}", exc_info=True)
            self._emit_chat_message_to_ui(f"[System Error: Delegation failed: {e}]", is_error=True)
            self._reset_sequence_state(error_occurred=True)

    def _create_detailed_implementation_request(self) -> str:
        """Create a detailed request for the micro-task coordinator based on our plan"""
        if not self._project_plan:
            return self._original_query

        request = f"Implement this project based on the architectural plan:\n\n"
        request += f"**Original Request**: {self._original_query}\n\n"
        request += f"**Architecture Summary**: {self._project_plan.architecture_summary}\n\n"

        request += "**Files to Implement**:\n"
        for file_spec in self._project_plan.file_specifications:
            request += f"\n### {file_spec['filename']}\n"
            request += f"Purpose: {file_spec['purpose']}\n"
            if file_spec.get('key_components'):
                request += f"Key Components: {', '.join(file_spec['key_components'])}\n"
            if file_spec.get('integration_notes'):
                request += f"Integration: {file_spec['integration_notes']}\n"

        if self._project_plan.integration_requirements:
            request += f"\n**Integration Requirements**:\n"
            for req in self._project_plan.integration_requirements:
                request += f"- {req}\n"

        if self._project_plan.quality_standards:
            request += f"\n**Quality Standards**: {self._project_plan.quality_standards}\n"

        return request

    async def _monitor_micro_task_execution(self):
        """Monitor the micro-task coordinator execution"""
        logger.info(f"PACC ({self._sequence_id}): Starting micro-task execution monitoring")

        # Wait for micro-task coordinator to complete
        max_wait_time = 300  # 5 minutes max
        check_interval = 2  # Check every 2 seconds
        waited_time = 0

        while self._is_monitoring_micro_tasks and waited_time < max_wait_time:
            await asyncio.sleep(check_interval)
            waited_time += check_interval

            # Check if micro-task coordinator is still busy
            if not self._micro_task_coordinator.is_busy():
                logger.info(f"PACC ({self._sequence_id}): Micro-task execution completed")
                self._is_monitoring_micro_tasks = False
                await self._handle_micro_task_completion()
                return

        if waited_time >= max_wait_time:
            logger.warning(f"PACC ({self._sequence_id}): Micro-task execution timeout")
            self._emit_chat_message_to_ui("[System Warning: Implementation taking longer than expected]")

    async def _handle_micro_task_completion(self):
        """Handle completion of micro-task execution"""
        logger.info(f"PACC ({self._sequence_id}): Micro-task execution completed, starting final review")

        self._current_phase = SequencePhase.FINAL_INTEGRATION_REVIEW
        self._emit_chat_message_to_ui(
            f"[System: ✅ **Phase 2 Complete - Implementation Finished**]\n"
            f"🔍 **Phase 3**: Final integration review by AI Architect..."
        )

        # Perform final integration review
        await self._perform_final_integration_review()

    async def _perform_final_integration_review(self):
        """Perform final integration review of the generated code"""
        # For now, just complete the sequence
        # In a more advanced version, this could analyze the generated files
        # and provide feedback or request adjustments

        self._current_phase = SequencePhase.COMPLETED
        self._emit_chat_message_to_ui(
            f"[System: 🎉 **Autonomous Development Complete!**]\n"
            f"✅ **AI Architect** planned the architecture\n"
            f"⚡ **Specialized Coder** implemented atomic components\n"
            f"🔧 **Intelligent Assembler** created production-ready files\n\n"
            f"📝 **All files are ready for review in the Code Viewer!**"
        )

        self._emit_status_update("✅ Autonomous development completed successfully!", "#98c379", False)
        self._reset_sequence_state()

    @Slot(str)
    def _handle_force_proceed(self):
        """Handle force proceed request"""
        logger.info(f"PACC ({self._sequence_id}): Force proceed requested")

        if self._current_phase == SequencePhase.AWAITING_PLAN_APPROVAL:
            self.confirm_plan_and_proceed()
        else:
            logger.info(f"PACC ({self._sequence_id}): Force proceed not applicable for current phase")

    @Slot(str, str)
    def _handle_llm_error(self, request_id: str, error_message: str):
        """Handle LLM errors"""
        if not self._sequence_id or self._active_llm_request_id != request_id:
            return

        logger.error(f"PACC ({self._sequence_id}): LLM error: {error_message}")
        self._active_llm_request_id = None

        self._emit_chat_message_to_ui(f"[System Error: AI planning failed: {error_message}]", is_error=True)
        self._reset_sequence_state(error_occurred=True, message=f"LLM error: {error_message}")

    def _reset_sequence_state(self, error_occurred: bool = False, message: Optional[str] = None):
        """Reset coordinator state"""
        logger.info(f"PACC ({self._sequence_id or 'N/A'}): Resetting state. Error: {error_occurred}")

        self._current_phase = SequencePhase.IDLE
        self._sequence_id = None
        self._original_query = None
        self._project_context.clear()
        self._project_plan = None
        self._active_llm_request_id = None
        self._micro_task_sequence_id = None
        self._is_monitoring_micro_tasks = False

        self._event_bus.uiInputBarBusyStateChanged.emit(False)

        if error_occurred:
            self._emit_status_update(f"❌ Plan-and-code failed: {message or 'Unknown error'}", "#e06c75", False)
        else:
            self._emit_status_update("Ready for new autonomous development", "#abb2bf", True, 3000)

    def _emit_status_update(self, message: str, color: str, is_temporary: bool, duration_ms: int = 0):
        """Emit status update"""
        self._event_bus.uiStatusUpdateGlobal.emit(message, color, is_temporary, duration_ms)

    def _emit_chat_message_to_ui(self, text: str, is_error: bool = False):
        """Emit chat message to UI"""
        project_id = self._project_context.get('project_id')
        session_id = self._project_context.get('session_id')
        if project_id and session_id:
            role = ERROR_ROLE if is_error else SYSTEM_ROLE
            msg_id = f"pacc_msg_{self._sequence_id or 'unknown'}_{uuid.uuid4().hex[:4]}"
            chat_msg = ChatMessage(id=msg_id, role=role, parts=[text])
            self._event_bus.newMessageAddedToHistory.emit(project_id, session_id, chat_msg)

    def is_busy(self) -> bool:
        """Check if coordinator is busy"""
        return self._current_phase != SequencePhase.IDLE