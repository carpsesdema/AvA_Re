# app/core/chat_router.py
import logging
import asyncio  # For async calls to handlers
from typing import TYPE_CHECKING, Optional, Dict, Any, List

from PySide6.QtCore import QObject

try:
    # Corrected path for UserInputHandler and its enum
    from core.user_input_handler import ProcessedInput, UserInputIntent
    # If specific prompts are constructed here, import from llm.prompts
    from llm import prompts as llm_prompts
    from models.chat_message import ChatMessage, USER_ROLE  # For constructing prompts if needed
except ImportError as e_cr:
    logging.getLogger(__name__).critical(f"ChatRouter: Critical import error: {e_cr}", exc_info=True)
    # Fallback types
    ProcessedInput = type("ProcessedInput", (object,), {})  # type: ignore
    UserInputIntent = type("UserInputIntent", (object,), {})  # type: ignore
    llm_prompts = type("llm_prompts", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    USER_ROLE = "user"  # type: ignore
    raise

if TYPE_CHECKING:
    # These are forward references for type hinting to avoid circular imports at runtime
    from core.conversation_orchestrator import ConversationOrchestrator
    from core.plan_and_code_coordinator import PlanAndCodeCoordinator
    from core.micro_task_coordinator import MicroTaskCoordinator
    from core.llm_request_handler import LlmRequestHandler
    # from core.project_iteration_handler import ProjectIterationHandler # If we create this

logger = logging.getLogger(__name__)


class ChatRouter(QObject):
    """
    Routes processed user input to the appropriate handler or coordinator
    based on the detected intent. This class itself doesn't perform LLM calls
    but delegates to specialized components.
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        # Dependencies will be injected by the ChatManager
        self._conversation_orchestrator: Optional['ConversationOrchestrator'] = None
        self._plan_and_code_coordinator: Optional['PlanAndCodeCoordinator'] = None
        self._micro_task_coordinator: Optional['MicroTaskCoordinator'] = None
        self._llm_request_handler: Optional['LlmRequestHandler'] = None
        # self._project_iteration_handler: Optional['ProjectIterationHandler'] = None # Example for future

        logger.info("ChatRouter initialized.")

    def set_handlers(self,
                     conversation_orchestrator: Optional['ConversationOrchestrator'] = None,
                     plan_and_code_coordinator: Optional['PlanAndCodeCoordinator'] = None,
                     micro_task_coordinator: Optional['MicroTaskCoordinator'] = None,
                     llm_request_handler: Optional['LlmRequestHandler'] = None
                     # project_iteration_handler: Optional['ProjectIterationHandler'] = None
                     ):
        """Injects the necessary handlers/coordinators that this router will delegate to."""
        self._conversation_orchestrator = conversation_orchestrator
        self._plan_and_code_coordinator = plan_and_code_coordinator
        self._micro_task_coordinator = micro_task_coordinator
        self._llm_request_handler = llm_request_handler
        # self._project_iteration_handler = project_iteration_handler
        logger.info("ChatRouter dependencies (handlers/coordinators) set.")

    async def route_request(self,
                            processed_input: ProcessedInput,
                            chat_history_for_llm: List['ChatMessage'],  # Full history for context
                            image_data: List[Dict[str, Any]],
                            # --- Context from ChatManager/BackendConfigManager ---
                            project_id: Optional[str],
                            session_id: Optional[str],
                            # Primary Chat LLM Config
                            chat_backend_id: str,
                            chat_model_name: str,
                            chat_temperature: float,
                            chat_system_prompt: Optional[str],  # Current system prompt for chat
                            # Specialized Coder LLM Config
                            specialized_backend_id: str,
                            specialized_model_name: str,
                            specialized_temperature: float,
                            # Other Context
                            current_project_directory: str,
                            project_name_for_context: Optional[str] = None,  # For iteration prompt
                            project_description_for_context: Optional[str] = None  # For iteration prompt
                            ):
        """
        Routes the user's request to the appropriate handler based on intent.
        """
        intent = processed_input.intent
        query_for_handler = processed_input.processed_query  # Query after any prefix stripping
        original_full_query = processed_input.original_query  # Full original query from user
        data_from_input_handler = processed_input.data

        logger.info(f"ChatRouter: Routing intent '{intent.name}' for query: '{query_for_handler[:50]}...'")

        if intent == UserInputIntent.CONVERSATIONAL_PLANNING:  # type: ignore
            if self._conversation_orchestrator:
                logger.debug("Routing to ConversationOrchestrator.")
                await self._conversation_orchestrator.start_conversation(
                    user_input=original_full_query,  # CO typically starts with the broader, original query
                    project_id=project_id,
                    session_id=session_id
                )
            else:
                await self._handle_missing_handler("Conversational Planning", chat_history_for_llm, image_data,
                                                   project_id, session_id, chat_backend_id, chat_model_name,
                                                   chat_temperature)

        elif intent == UserInputIntent.MICRO_TASK_REQUEST:  # type: ignore
            if self._micro_task_coordinator:
                logger.debug("Routing to MicroTaskCoordinator.")
                self._micro_task_coordinator.start_micro_task_generation(
                    user_query=query_for_handler,
                    planning_backend=chat_backend_id,  # Use chat LLM for planning phase
                    planning_model=chat_model_name,
                    coding_backend=specialized_backend_id,
                    coding_model=specialized_model_name,
                    project_dir=current_project_directory,
                    project_id=project_id,
                    session_id=session_id
                )
            else:
                await self._handle_missing_handler("Micro-Task Generation", chat_history_for_llm, image_data,
                                                   project_id, session_id, chat_backend_id, chat_model_name,
                                                   chat_temperature)

        elif intent == UserInputIntent.PLAN_THEN_CODE_REQUEST:  # type: ignore
            if self._plan_and_code_coordinator:
                logger.debug("Routing to PlanAndCodeCoordinator.")
                self._plan_and_code_coordinator.start_autonomous_coding(
                    user_query=query_for_handler,
                    planner_backend=chat_backend_id,
                    planner_model=chat_model_name,
                    coder_backend=specialized_backend_id,
                    coder_model=specialized_model_name,
                    project_dir=current_project_directory,
                    project_id=project_id,
                    session_id=session_id,
                    task_type=data_from_input_handler.get("task_type", "general")
                )
            else:
                await self._handle_missing_handler("Plan-then-Code", chat_history_for_llm, image_data, project_id,
                                                   session_id, chat_backend_id, chat_model_name, chat_temperature)

        elif intent == UserInputIntent.FILE_CREATION_REQUEST:  # type: ignore
            if self._llm_request_handler:
                logger.debug("Routing simple file creation to LlmRequestHandler.")
                filename = data_from_input_handler.get("filename", "new_file.py")
                # Construct the specialized prompt for file creation
                # This is where a PromptBuilderService would be ideal.
                prompt_for_file = self._build_file_creation_prompt(original_full_query, filename)

                self._llm_request_handler.submit_simple_file_creation_request(
                    prompt_text=prompt_for_file,
                    filename=filename,
                    backend_id=specialized_backend_id,  # Use specialized for code
                    options={"temperature": specialized_temperature},
                    project_id=project_id,  # type: ignore
                    session_id=session_id  # type: ignore
                )
            else:
                await self._handle_missing_handler("File Creation", chat_history_for_llm, image_data, project_id,
                                                   session_id, chat_backend_id, chat_model_name, chat_temperature)

        elif intent == UserInputIntent.PROJECT_ITERATION_REQUEST:  # type: ignore
            if self._llm_request_handler:  # Using LlmRequestHandler for iteration for now
                logger.debug("Routing project iteration request to LlmRequestHandler.")

                project_context_str = f"Project: {project_name_for_context or 'Current Project'}"
                if project_description_for_context: project_context_str += f"\nDescription: {project_description_for_context}"

                iteration_prompt = self._build_project_iteration_prompt(original_full_query, project_context_str)

                # For iteration, the history should be used, and the iteration_prompt becomes the new user message
                # or a system message augmenting the user's request.
                # Let's prepend it as a system message before the user's actual iteration query.
                # The `chat_history_for_llm` already contains the user's latest query.
                # A better approach might be to have LlmRequestHandler.submit_project_iteration_request
                # that takes the iteration_prompt and the history separately.

                # For now, create a new history list starting with the iteration prompt
                # This might lose some conversational nuance if chat_history_for_llm isn't used effectively.
                # This part needs careful thought on how iteration prompts integrate with ongoing history.

                # Option: Treat the iteration_prompt as the primary instruction.
                history_for_iteration_request = [ChatMessage(role=USER_ROLE, parts=[iteration_prompt])]  # type: ignore
                # If image_data is relevant to iteration, it should be part of the iteration_prompt construction.

                self._llm_request_handler.submit_project_iteration_request(
                    iteration_prompt_text=iteration_prompt,  # Pass the fully formed prompt
                    history_for_context=chat_history_for_llm,  # Pass existing history for broader context
                    backend_id=chat_backend_id,  # Iteration is more analytical
                    options={"temperature": chat_temperature + 0.1 if chat_temperature <= 1.9 else 2.0},
                    # Slightly more creative
                    project_id=project_id,  # type: ignore
                    session_id=session_id  # type: ignore
                )
            else:
                await self._handle_missing_handler("Project Iteration", chat_history_for_llm, image_data, project_id,
                                                   session_id, chat_backend_id, chat_model_name, chat_temperature)

        elif intent == UserInputIntent.NORMAL_CHAT:  # type: ignore
            if self._llm_request_handler:
                logger.debug("Routing normal chat to LlmRequestHandler.")
                # The chat_history_for_llm already includes the latest user message.
                self._llm_request_handler.submit_normal_chat_request(
                    history_to_send=chat_history_for_llm,
                    backend_id=chat_backend_id,
                    options={"temperature": chat_temperature},
                    project_id=project_id,  # type: ignore
                    session_id=session_id  # type: ignore
                )
            else:
                await self._handle_missing_handler("Normal Chat", chat_history_for_llm, image_data, project_id,
                                                   session_id, chat_backend_id, chat_model_name, chat_temperature,
                                                   is_critical_fallback=True)
        else:
            logger.error(f"ChatRouter: Unknown intent '{intent.name}'. Cannot route request.")
            if project_id and session_id and self._event_bus:
                from models.chat_message import ChatMessage, ERROR_ROLE  # type: ignore
                err_msg = ChatMessage(role=ERROR_ROLE, parts=[
                    f"[System Error: Could not understand request type '{intent.name}'.]"])  # type: ignore
                self._event_bus.newMessageAddedToHistory.emit(project_id, session_id, err_msg)

    async def _handle_missing_handler(self, handler_name: str,
                                      chat_history_for_llm: List['ChatMessage'],
                                      image_data: List[Dict[str, Any]],
                                      project_id: Optional[str], session_id: Optional[str],
                                      chat_backend_id: str, chat_model_name: str, chat_temperature: float,
                                      is_critical_fallback: bool = False):
        logger.error(f"{handler_name} intent detected but the handler is not set in ChatRouter.")
        if is_critical_fallback and project_id and session_id and hasattr(self, '_event_bus'):
            from models.chat_message import ChatMessage, ERROR_ROLE  # type: ignore
            err_msg = ChatMessage(role=ERROR_ROLE,
                                  parts=[f"[System Error: {handler_name} handler not available.]"])  # type: ignore
            self._event_bus.newMessageAddedToHistory.emit(project_id, session_id, err_msg)  # type: ignore
        elif self._llm_request_handler:  # Fallback to normal chat if possible
            logger.warning(f"Falling back to normal chat for missing {handler_name} handler.")
            self._llm_request_handler.submit_normal_chat_request(
                history_to_send=chat_history_for_llm,  # This history includes the original user query
                backend_id=chat_backend_id,
                options={"temperature": chat_temperature},
                project_id=project_id,  # type: ignore
                session_id=session_id  # type: ignore
            )
        # else: no handler at all, error already logged.

    # --- Prompt Construction Helpers (Ideally move to a PromptBuilderService) ---
    def _build_file_creation_prompt(self, original_query: str, filename: str) -> str:
        # This is a simplified adaptation. A dedicated PromptBuilder would be better.
        # It needs access to the various specific coding prompts from llm_prompts.
        # For now, using a generic approach.
        task_type = self._infer_task_type_from_filename(filename)  # Basic inference

        specific_prompt_template = getattr(llm_prompts, 'GENERAL_CODING_PROMPT', "")  # type: ignore
        if task_type == 'api':
            specific_prompt_template = getattr(llm_prompts, 'API_DEVELOPMENT_PROMPT',
                                               specific_prompt_template)  # type: ignore
        elif task_type == 'ui':
            specific_prompt_template = getattr(llm_prompts, 'UI_DEVELOPMENT_PROMPT',
                                               specific_prompt_template)  # type: ignore
        # Add more task types...

        return (f"Generate the complete Python code for a new file named '{filename}'.\n"
                f"User's original request for context: \"{original_query}\"\n\n"
                f"Follow these specific coding guidelines if applicable:\n{specific_prompt_template}\n\n"
                f"Ensure the output is ONLY the code itself, enclosed in a single Python code block (```python ... ```). "
                f"The code must be complete, executable, and adhere to high quality standards including error handling, "
                f"type hints, docstrings, and comments where necessary.")

    def _build_project_iteration_prompt(self, original_query: str, project_context_str: Optional[str]) -> str:
        base_prompt = getattr(llm_prompts, 'ENHANCED_CODING_SYSTEM_PROMPT',
                              "You are an expert software architect...")  # type: ignore
        iteration_header = (
            "## PROJECT ITERATION/MODIFICATION TASK\n"
            "Based on the user's request and the provided project context, generate the necessary code changes or additions.\n"
        )
        user_request_section = f"**User's Request**: {original_query}\n\n"
        project_context_section = f"**Existing Project Context Summary**:\n{project_context_str or 'No specific project context provided.'}\n\n"
        rag_placeholder = "**Relevant Code Snippets from Project (if RAG is used, this will be populated by the system)**:\n[RAG_CONTEXT_PLACEHOLDER]\n\n"
        instructions = (
            "**Your Task Details**:\n"
            "1.  Carefully analyze the user's request and the existing project context.\n"
            "2.  If modifying existing files: Clearly state the filename. Provide the *complete, updated code* for that file in a fenced Python code block. Do NOT output only diffs or snippets unless explicitly asked.\n"
            "3.  If creating new files: Provide the complete code for each new file in its own fenced Python code block, clearly stating the filename.\n"
            "4.  If providing analysis or explanation, do so concisely before any code blocks.\n"
            "5.  Adhere to high code quality standards (PEP 8, type hints, docstrings, error handling, comments).\n"
            "6.  If RAG context is present above, prioritize it for understanding existing code.\n\n"
            "**Output Format for Code**:\n"
            "  `### FILENAME: path/to/your/file.py`\n"
            "  ` ```python\n  # ... your complete code for this file ...\n  ``` `\n"
        )
        return f"{base_prompt}\n\n{iteration_header}{user_request_section}{project_context_section}{rag_placeholder}{instructions}"

    def _infer_task_type_from_filename(self, filename: str) -> str:
        # Duplicated from PlanAndCodeCoordinator - ideally, this is a shared utility.
        fn_lower = filename.lower()
        if "api" in fn_lower or "server" in fn_lower or "route" in fn_lower: return "api"
        if "test" in fn_lower: return "test"
        if "util" in fn_lower or "helper" in fn_lower: return "utility"
        if "config" in fn_lower: return "config"
        if "model" in fn_lower or "schema" in fn_lower: return "data"
        if "ui" in fn_lower or "view" in fn_lower or "widget" in fn_lower: return "ui"
        return "general"
        return "general"