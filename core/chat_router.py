# app/core/chat_router.py
import logging
import asyncio
from typing import TYPE_CHECKING, Optional, Dict, Any, List

from PySide6.QtCore import QObject

try:
    from core.user_input_handler import ProcessedInput, UserInputIntent
    from llm import prompts as llm_prompts
    from models.chat_message import ChatMessage, USER_ROLE, ERROR_ROLE
except ImportError as e_cr:
    logging.getLogger(__name__).critical(f"ChatRouter: Critical import error: {e_cr}", exc_info=True)
    ProcessedInput = type("ProcessedInput", (object,), {})
    UserInputIntent = type("UserInputIntent", (object,), {})
    llm_prompts = type("llm_prompts", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})
    USER_ROLE = ERROR_ROLE = "user"
    raise

if TYPE_CHECKING:
    from core.conversation_orchestrator import ConversationOrchestrator
    from core.plan_and_code_coordinator import PlanAndCodeCoordinator
    from core.micro_task_coordinator import MicroTaskCoordinator
    from core.llm_request_handler import LlmRequestHandler

logger = logging.getLogger(__name__)


class ChatRouter(QObject):
    """
    Enhanced router that intelligently routes requests to the appropriate coordination system.
    Now works with the enhanced Plan-and-Code + Micro-Task architecture.
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        # Core coordinators
        self._conversation_orchestrator: Optional['ConversationOrchestrator'] = None
        self._plan_and_code_coordinator: Optional['PlanAndCodeCoordinator'] = None
        self._micro_task_coordinator: Optional['MicroTaskCoordinator'] = None
        self._llm_request_handler: Optional['LlmRequestHandler'] = None

        logger.info("Enhanced ChatRouter initialized.")

    def set_handlers(self,
                     conversation_orchestrator: Optional['ConversationOrchestrator'] = None,
                     plan_and_code_coordinator: Optional['PlanAndCodeCoordinator'] = None,
                     micro_task_coordinator: Optional['MicroTaskCoordinator'] = None,
                     llm_request_handler: Optional['LlmRequestHandler'] = None):
        """Set the enhanced coordination handlers"""
        self._conversation_orchestrator = conversation_orchestrator
        self._plan_and_code_coordinator = plan_and_code_coordinator
        self._micro_task_coordinator = micro_task_coordinator
        self._llm_request_handler = llm_request_handler
        logger.info("Enhanced ChatRouter dependencies set.")

    async def route_request(self,
                            processed_input: ProcessedInput,
                            chat_history_for_llm: List['ChatMessage'],
                            image_data: List[Dict[str, Any]],
                            # Context from ChatManager
                            project_id: Optional[str],
                            session_id: Optional[str],
                            # Primary Chat LLM Config
                            chat_backend_id: str,
                            chat_model_name: str,
                            chat_temperature: float,
                            chat_system_prompt: Optional[str],
                            # Specialized Coder LLM Config
                            specialized_backend_id: str,
                            specialized_model_name: str,
                            specialized_temperature: float,
                            specialized_system_prompt: Optional[str],  # FIXED: Added missing parameter
                            # Project Context
                            current_project_directory: str,
                            project_name_for_context: Optional[str] = None,
                            project_description_for_context: Optional[str] = None):
        """
        Enhanced routing with intelligent decision making
        """
        intent = processed_input.intent
        query_for_handler = processed_input.processed_query
        original_full_query = processed_input.original_query
        data_from_input_handler = processed_input.data

        logger.info(f"ChatRouter: Routing enhanced intent '{intent.name}' for query: '{query_for_handler[:50]}...'")

        # Enhanced routing logic
        if intent == UserInputIntent.CONVERSATIONAL_PLANNING:
            await self._route_to_conversational_planning(
                original_full_query, project_id, session_id
            )

        elif intent == UserInputIntent.PLAN_THEN_CODE_REQUEST:
            await self._route_to_autonomous_development(
                query_for_handler, chat_backend_id, chat_model_name,
                specialized_backend_id, specialized_model_name,
                current_project_directory, project_id, session_id,
                data_from_input_handler.get("task_type", "general")
            )

        elif intent == UserInputIntent.MICRO_TASK_REQUEST:
            await self._route_to_micro_task_generation(
                query_for_handler, chat_backend_id, chat_model_name,
                specialized_backend_id, specialized_model_name,
                current_project_directory, project_id, session_id
            )

        elif intent == UserInputIntent.FILE_CREATION_REQUEST:
            await self._route_to_smart_file_creation(
                original_full_query, data_from_input_handler.get("filename", "new_file.py"),
                specialized_backend_id, specialized_temperature,
                project_id, session_id
            )

        elif intent == UserInputIntent.PROJECT_ITERATION_REQUEST:
            await self._route_to_project_iteration(
                original_full_query, chat_history_for_llm, chat_backend_id,
                chat_temperature, project_id, session_id,
                project_name_for_context, project_description_for_context
            )

        elif intent == UserInputIntent.NORMAL_CHAT:
            await self._route_to_normal_chat(
                chat_history_for_llm, chat_backend_id, chat_temperature,
                project_id, session_id
            )

        else:
            logger.error(f"ChatRouter: Unknown intent '{intent.name}'")
            await self._handle_unknown_intent(intent, project_id, session_id)

    async def _route_to_conversational_planning(self,
                                                user_input: str,
                                                project_id: Optional[str],
                                                session_id: Optional[str]):
        """Route complex requests to conversational orchestrator"""
        if self._conversation_orchestrator:
            logger.debug("Routing to ConversationOrchestrator for complex planning")
            await self._conversation_orchestrator.start_conversation(
                user_input=user_input,
                project_id=project_id,
                session_id=session_id
            )
        else:
            await self._handle_missing_handler("Conversational Planning", project_id, session_id)

    async def _route_to_autonomous_development(self,
                                               user_query: str,
                                               planner_backend: str,
                                               planner_model: str,
                                               coder_backend: str,
                                               coder_model: str,
                                               project_dir: str,
                                               project_id: Optional[str],
                                               session_id: Optional[str],
                                               task_type: str):
        """Route to the enhanced Plan-and-Code coordinator"""
        if self._plan_and_code_coordinator:
            logger.debug("Routing to enhanced PlanAndCodeCoordinator")

            # Determine task complexity to decide routing
            complexity_score = self._assess_task_complexity(user_query)

            if complexity_score >= 7:  # High complexity - use full autonomous development
                success = self._plan_and_code_coordinator.start_autonomous_coding(
                    user_query=user_query,
                    planner_backend=planner_backend,
                    planner_model=planner_model,
                    coder_backend=coder_backend,
                    coder_model=coder_model,
                    project_dir=project_dir,
                    project_id=project_id,
                    session_id=session_id,
                    task_type=task_type
                )

                if not success:
                    logger.warning("Plan-and-Code coordinator could not start, falling back to micro-tasks")
                    await self._route_to_micro_task_generation(
                        user_query, planner_backend, planner_model,
                        coder_backend, coder_model, project_dir,
                        project_id, session_id
                    )
            else:
                # Medium complexity - route directly to micro-tasks
                logger.info("Task complexity suggests micro-task approach")
                await self._route_to_micro_task_generation(
                    user_query, planner_backend, planner_model,
                    coder_backend, coder_model, project_dir,
                    project_id, session_id
                )
        else:
            await self._handle_missing_handler("Plan-and-Code", project_id, session_id)

    async def _route_to_micro_task_generation(self,
                                              user_query: str,
                                              planning_backend: str,
                                              planning_model: str,
                                              coding_backend: str,
                                              coding_model: str,
                                              project_dir: str,
                                              project_id: Optional[str],
                                              session_id: Optional[str]):
        """Route to enhanced micro-task coordinator"""
        if self._micro_task_coordinator:
            logger.debug("Routing to enhanced MicroTaskCoordinator")

            success = self._micro_task_coordinator.start_micro_task_generation(
                user_query=user_query,
                planning_backend=planning_backend,
                planning_model=planning_model,
                coding_backend=coding_backend,
                coding_model=coding_model,
                project_dir=project_dir,
                project_id=project_id,
                session_id=session_id
            )

            if not success:
                logger.error("Micro-task coordinator could not start")
                await self._handle_missing_handler("Micro-Task Generation", project_id, session_id, is_critical=True)
        else:
            await self._handle_missing_handler("Micro-Task Generation", project_id, session_id, is_critical=True)

    async def _route_to_smart_file_creation(self,
                                            user_query: str,
                                            filename: str,
                                            backend_id: str,
                                            temperature: float,
                                            project_id: Optional[str],
                                            session_id: Optional[str]):
        """Enhanced file creation with intelligence"""
        if self._llm_request_handler:
            logger.debug("Routing to enhanced file creation")

            # Build intelligent file creation prompt using existing prompt system
            smart_prompt = self._build_smart_file_creation_prompt(user_query, filename)

            self._llm_request_handler.submit_simple_file_creation_request(
                prompt_text=smart_prompt,
                filename=filename,
                backend_id=backend_id,
                options={"temperature": temperature},
                project_id=project_id,
                session_id=session_id
            )
        else:
            await self._handle_missing_handler("Smart File Creation", project_id, session_id)

    async def _route_to_project_iteration(self,
                                          user_query: str,
                                          chat_history: List['ChatMessage'],
                                          backend_id: str,
                                          temperature: float,
                                          project_id: Optional[str],
                                          session_id: Optional[str],
                                          project_name: Optional[str],
                                          project_description: Optional[str]):
        """Route project iteration requests"""
        if self._llm_request_handler:
            logger.debug("Routing to project iteration")

            # Build context-aware iteration prompt
            iteration_prompt = self._build_project_iteration_prompt(
                user_query, project_name, project_description
            )

            self._llm_request_handler.submit_project_iteration_request(
                iteration_prompt_text=iteration_prompt,
                history_for_context=chat_history,
                backend_id=backend_id,
                options={"temperature": temperature + 0.1},  # Slightly more creative
                project_id=project_id,
                session_id=session_id
            )
        else:
            await self._handle_missing_handler("Project Iteration", project_id, session_id)

    async def _route_to_normal_chat(self,
                                    chat_history: List['ChatMessage'],
                                    backend_id: str,
                                    temperature: float,
                                    project_id: Optional[str],
                                    session_id: Optional[str]):
        """Route normal chat requests"""
        if self._llm_request_handler:
            logger.debug("Routing to normal chat")
            self._llm_request_handler.submit_normal_chat_request(
                history_to_send=chat_history,
                backend_id=backend_id,
                options={"temperature": temperature},
                project_id=project_id,
                session_id=session_id
            )
        else:
            await self._handle_missing_handler("Normal Chat", project_id, session_id, is_critical=True)

    def _assess_task_complexity(self, user_query: str) -> int:
        """Assess task complexity on a 1-10 scale to determine optimal routing"""
        query_lower = user_query.lower()
        complexity_score = 1

        # Multi-file indicators
        if any(word in query_lower for word in ['system', 'application', 'project', 'architecture']):
            complexity_score += 3

        # Multiple component indicators
        if any(word in query_lower for word in ['api and', 'backend and', 'frontend and', 'database and']):
            complexity_score += 2

        # Advanced feature indicators
        if any(word in query_lower for word in ['authentication', 'authorization', 'microservice', 'scalable']):
            complexity_score += 2

        # Length-based complexity
        word_count = len(user_query.split())
        if word_count > 20:
            complexity_score += 1
        if word_count > 50:
            complexity_score += 1

        return min(complexity_score, 10)

    def _build_smart_file_creation_prompt(self, user_query: str, filename: str) -> str:
        """Build intelligent file creation prompt using existing prompt system"""
        # Determine file type from filename
        file_ext = filename.lower().split('.')[-1] if '.' in filename else 'py'

        # Select appropriate base prompt
        base_prompt = llm_prompts.GENERAL_CODING_PROMPT

        if any(word in filename.lower() for word in ['api', 'server', 'endpoint']):
            base_prompt = llm_prompts.API_DEVELOPMENT_PROMPT
        elif any(word in filename.lower() for word in ['data', 'process', 'etl', 'transform']):
            base_prompt = llm_prompts.DATA_PROCESSING_PROMPT
        elif any(word in filename.lower() for word in ['ui', 'widget', 'dialog', 'window']):
            base_prompt = llm_prompts.UI_DEVELOPMENT_PROMPT
        elif any(word in filename.lower() for word in ['util', 'helper', 'tool']):
            base_prompt = llm_prompts.UTILITY_DEVELOPMENT_PROMPT

        # Enhance with specific file context
        enhanced_prompt = f"""Create a complete Python file named '{filename}'.

**User Request**: {user_query}

**File-Specific Requirements**:
- The file must be production-ready and immediately executable
- Include all necessary imports and dependencies
- Follow the coding standards specified below
- Implement comprehensive error handling and logging

{base_prompt}

**Critical**: Generate ONLY the complete Python code for {filename}. No explanations outside the code block."""

        return enhanced_prompt

    def _build_project_iteration_prompt(self,
                                        user_query: str,
                                        project_name: Optional[str],
                                        project_description: Optional[str]) -> str:
        """Build context-aware project iteration prompt"""
        project_context = ""
        if project_name:
            project_context += f"**Project Name**: {project_name}\n"
        if project_description:
            project_context += f"**Project Description**: {project_description}\n"

        iteration_prompt = f"""## PROJECT ITERATION/MODIFICATION REQUEST

{project_context}

**User's Iteration Request**: {user_query}

**Task**: Analyze the request and provide the necessary code changes, additions, or modifications.

**Instructions**:
1. If modifying existing files: Provide the COMPLETE updated code for the entire file
2. If creating new files: Provide complete code with proper integration
3. If providing analysis: Be specific about implementation approach
4. Follow the enhanced coding standards from the system prompt
5. Ensure changes integrate properly with existing project structure

**Output Format**:
For each file affected:
```
### FILENAME: path/to/file.py
```python
# Complete file contents here
```
```

Use the enhanced coding system prompt standards for all generated code."""

        return iteration_prompt

    async def _handle_missing_handler(self,
                                      handler_name: str,
                                      project_id: Optional[str],
                                      session_id: Optional[str],
                                      is_critical: bool = False):
        """Handle cases where required handlers are not available"""
        logger.error(f"{handler_name} handler not available in ChatRouter")

        if project_id and session_id:
            error_msg = ChatMessage(
                role=ERROR_ROLE,
                parts=[f"[System Error: {handler_name} handler not available]"]
            )
            # Would need access to event bus to emit this
            # self._event_bus.newMessageAddedToHistory.emit(project_id, session_id, error_msg)

        if is_critical:
            logger.critical(f"Critical handler {handler_name} missing - core functionality unavailable")

    async def _handle_unknown_intent(self,
                                     intent: Any,
                                     project_id: Optional[str],
                                     session_id: Optional[str]):
        """Handle unknown intents"""
        logger.error(f"ChatRouter: Unknown intent '{intent.name}' - cannot route request")

        if project_id and session_id:
            error_msg = ChatMessage(
                role=ERROR_ROLE,
                parts=[f"[System Error: Unknown request type '{intent.name}']"]
            )
            # Would need access to event bus to emit this
            # self._event_bus.newMessageAddedToHistory.emit(project_id, session_id, error_msg)