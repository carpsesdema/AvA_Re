# app/core/conversation_orchestrator.py
import logging
import re
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime

from PySide6.QtCore import QObject, Slot

# Assuming UserInputHandler is now in app.core
try:
    from core.user_input_handler import UserInputHandler
except ImportError:
    # Fallback if UserInputHandler isn't found immediately (e.g. during restructuring)
    UserInputHandler = type("UserInputHandler", (object,), {"_extract_filename_from_query": lambda self, query: None})
    logging.getLogger(__name__).warning("ConversationOrchestrator: UserInputHandler not found, using fallback.")

try:
    from core.event_bus import EventBus  # This should be app.core.event_bus if EventBus is in app/core
    from models.chat_message import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE
    from llm.backend_coordinator import BackendCoordinator
    from utils import constants
    # Prompts would be imported from app.llm.prompts if specific ones were used here
    # from app.llm import prompts as llm_prompts
except ImportError as e:
    logging.getLogger(__name__).critical(f"ConversationOrchestrator: Critical import error: {e}", exc_info=True)
    # Fallback types
    EventBus = type("EventBus", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    USER_ROLE, MODEL_ROLE, SYSTEM_ROLE = "user", "model", "system"
    BackendCoordinator = type("BackendCoordinator", (object,), {})
    constants = type("constants", (object,), {"DEFAULT_CHAT_BACKEND_ID": "gemini_chat_default",
                                              "DEFAULT_GEMINI_CHAT_MODEL": "gemini-1.5-pro-latest"})  # type: ignore
    raise

if TYPE_CHECKING:
    # This import is fine for type checking even if ChatManager is not fully defined yet
    from core.chat_manager import ChatManager  # Assuming ChatManager will be in app.core

logger = logging.getLogger(__name__)


class ConversationPhase(Enum):
    """Defines the current phase of an orchestrated conversation."""
    IDLE = auto()
    UNDERSTANDING_INTENT = auto()
    GATHERING_REQUIREMENTS = auto()
    CONFIRMING_UNDERSTANDING = auto()
    BUILDING_FINAL_PROMPT = auto()
    ROUTING_TO_HANDLER = auto()
    COMPLETED = auto()
    ERROR = auto()


class ConversationIntent(Enum):
    """Broad categories of what the user might want to achieve through conversation."""
    UNKNOWN = auto()
    GENERAL_QUERY = auto()
    SINGLE_FILE_GENERATION = auto()
    MULTI_FILE_PROJECT_CREATION = auto()
    PROJECT_ITERATION_OR_MODIFICATION = auto()
    COMPLEX_TASK_DECOMPOSITION = auto()


@dataclass
class OrchestratedConversation:
    """Stores the state and context of an ongoing orchestrated conversation."""
    conversation_id: str
    original_user_request: str
    current_project_id: Optional[str] = None
    current_session_id: Optional[str] = None

    phase: ConversationPhase = ConversationPhase.IDLE
    detected_intent: ConversationIntent = ConversationIntent.UNKNOWN
    confidence_in_intent: float = 0.0

    gathered_requirements: List[str] = field(default_factory=list)
    clarification_history: List[Dict[str, str]] = field(default_factory=list)
    llm_insights: List[str] = field(default_factory=list)

    target_handler_suggestion: Optional[str] = None
    final_actionable_prompt: Optional[str] = None
    intermediate_llm_request_id: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.now)
    last_updated_at: datetime = field(default_factory=datetime.now)

    def update_timestamp(self):
        self.last_updated_at = datetime.now()


class ConversationOrchestrator(QObject):
    """
    Manages multi-turn conversations to refine user requests into actionable prompts.
    """

    def __init__(self, event_bus: EventBus, parent: Optional[QObject] = None):
        super().__init__(parent)
        if not isinstance(event_bus, EventBus):  # type: ignore
            raise TypeError("ConversationOrchestrator requires a valid EventBus instance.")

        self._event_bus = event_bus
        self._active_conversations: Dict[str, OrchestratedConversation] = {}
        self._chat_manager: Optional['ChatManager'] = None
        self._backend_coordinator: Optional[BackendCoordinator] = None
        self._user_input_handler = UserInputHandler()  # Instantiate UserInputHandler

        self._connect_event_bus_handlers()
        logger.info("ConversationOrchestrator initialized.")

    def set_dependencies(self, chat_manager: 'ChatManager', backend_coordinator: BackendCoordinator):
        """Sets dependencies required by the orchestrator."""
        from core.chat_manager import ChatManager  # Local import for type check
        if not isinstance(chat_manager, ChatManager):
            raise TypeError("Invalid ChatManager instance provided to ConversationOrchestrator.")
        if not isinstance(backend_coordinator, BackendCoordinator):  # type: ignore
            raise TypeError("Invalid BackendCoordinator instance provided to ConversationOrchestrator.")

        self._chat_manager = chat_manager
        self._backend_coordinator = backend_coordinator
        logger.info("ConversationOrchestrator dependencies (ChatManager, BackendCoordinator) set.")

    def _connect_event_bus_handlers(self):
        self._event_bus.llmResponseCompleted.connect(self._handle_orchestrator_llm_response)
        self._event_bus.llmResponseError.connect(self._handle_orchestrator_llm_error)

    def can_handle_input(self, user_input: str) -> bool:
        """Determines if the input is suitable for conversational orchestration."""
        input_lower = user_input.lower().strip()
        if not input_lower: return False

        planning_keywords = [
            "project", "application", "system", "tool", "feature", "design",
            "architecture", "build", "create", "develop", "implement", "make me a",
            "i need a", "i want to", "help me with", "how do i start", "what if",
            "complex", "large scale", "multi-part"
        ]
        simple_query_keywords = [
            "what is", "who is", "explain", "define", "how to", "example of", "convert",
            "hello", "hi", "thanks", "ok"
        ]

        if len(input_lower.split()) < 4 and any(sqk in input_lower for sqk in simple_query_keywords):
            return False

        if any(pk in input_lower for pk in planning_keywords):
            if "file called" in input_lower or "script named" in input_lower:
                # Check if UserInputHandler can extract a filename
                if self._user_input_handler._extract_filename_from_query(user_input):
                    return False  # Specific file creation, not for orchestration
            return True

        if len(input_lower) > 120 and not any(sqk in input_lower for sqk in simple_query_keywords):
            return True
        return False

    async def start_conversation(self, user_input: str, project_id: Optional[str], session_id: Optional[str]) -> str:
        """Initiates a new orchestrated conversation."""
        if not self._chat_manager or not self._backend_coordinator:
            logger.error("Cannot start conversation: Core dependencies not set.")
            if self._chat_manager and hasattr(self._chat_manager, '_handle_normal_chat_async'):
                # Fallback to normal chat if possible, but this indicates a setup issue
                await self._chat_manager._handle_normal_chat_async(user_input, [])  # type: ignore
            raise RuntimeError("ConversationOrchestrator dependencies not set.")

        convo_id = f"convo_{uuid.uuid4().hex[:8]}"
        conversation = OrchestratedConversation(
            conversation_id=convo_id,
            original_user_request=user_input,
            current_project_id=project_id,
            current_session_id=session_id,
            phase=ConversationPhase.UNDERSTANDING_INTENT
        )
        self._active_conversations[convo_id] = conversation
        logger.info(f"Started new orchestrated conversation (ID: {convo_id}) for: '{user_input[:70]}...'")
        self._post_system_message_to_ui(conversation,
                                        "[System: Thinking about how to approach this complex request...]")
        await self._request_llm_for_understanding(conversation)
        return convo_id

    async def process_user_response(self, user_response: str, conversation_id: str) -> bool:
        """Processes a user's response during an ongoing conversation."""
        conversation = self._active_conversations.get(conversation_id)
        if not conversation:
            logger.warning(f"Received response for unknown/completed conversation_id: {conversation_id}")
            return False

        conversation.update_timestamp()
        last_q = conversation.clarification_history[-1]["q"] if conversation.clarification_history else "Initial phase"
        conversation.clarification_history.append({"q": f"(Response to: {last_q[:50]}...)", "a": user_response})
        logger.info(f"Processing user response for convo {conversation_id}: '{user_response[:70]}...'")

        if conversation.phase == ConversationPhase.GATHERING_REQUIREMENTS or \
                conversation.phase == ConversationPhase.CONFIRMING_UNDERSTANDING:  # User might provide feedback to confirmation
            await self._request_llm_for_analysis_and_next_step(conversation)
        else:
            logger.warning(
                f"User response received in unexpected phase: {conversation.phase.name} for convo {conversation_id}")
            await self._request_llm_for_analysis_and_next_step(conversation)  # Attempt to recover
        return conversation.phase == ConversationPhase.COMPLETED

    def _get_llm_config_for_orchestration(self) -> Tuple[str, str, float]:
        """Gets LLM configuration for orchestration tasks."""
        if self._chat_manager:
            backend_id = self._chat_manager.get_current_active_chat_backend_id()
            model_name = self._chat_manager.get_model_for_backend(backend_id)
            temp = self._chat_manager.get_current_chat_temperature()
            if model_name:
                return backend_id, model_name, temp if temp is not None else 0.5  # Default temp for planning
        logger.warning("CO: ChatManager LLM config not fully available. Using defaults for orchestration.")
        return constants.DEFAULT_CHAT_BACKEND_ID, constants.DEFAULT_GEMINI_CHAT_MODEL, 0.5  # type: ignore

    async def _request_llm_for_understanding(self, conversation: OrchestratedConversation):
        """Asks LLM to analyze initial request and formulate clarifying questions."""
        if not self._backend_coordinator: return
        prompt = (
            f"You are an expert software planning assistant. A user has made the following request:\n"
            f"\"{conversation.original_user_request}\"\n\n"
            f"Your goal is to fully understand their needs. Ask 2-3 critical clarifying questions to help define "
            f"the scope, key features, constraints, and desired output. "
            f"Phrase your questions clearly and number them. Do not add any conversational filler."
        )
        history = [ChatMessage(role=USER_ROLE, parts=[prompt])]  # type: ignore
        req_id = f"convo_understand_{conversation.conversation_id}"
        conversation.intermediate_llm_request_id = req_id
        conversation.phase = ConversationPhase.GATHERING_REQUIREMENTS  # Expecting questions, so will gather after
        conversation.update_timestamp()
        backend_id, model_name, temp = self._get_llm_config_for_orchestration()
        self._backend_coordinator.start_llm_streaming_task(
            request_id=req_id, target_backend_id=backend_id, history_to_send=history,
            is_modification_response_expected=False, options={"temperature": temp},
            request_metadata={"purpose": "orchestration_understanding", "conversation_id": conversation.conversation_id,
                              "project_id": conversation.current_project_id,
                              "session_id": conversation.current_session_id}
        )

    async def _request_llm_for_analysis_and_next_step(self, conversation: OrchestratedConversation):
        """Asks LLM to analyze gathered info and decide next step."""
        if not self._backend_coordinator: return
        history_summary = "\n".join([f"Q: {qa['q']}\nA: {qa['a']}" for qa in conversation.clarification_history])
        prompt = (
            f"Original Request: \"{conversation.original_user_request}\"\n"
            f"Conversation History:\n{history_summary}\n\n"
            f"Analyze the current understanding. Determine the next step:\n"
            f"1. If more critical details are needed: `NEXT_STEP: ASK_QUESTIONS\\n[New numbered questions]`\n"
            f"2. If understanding is good, summarize it for user confirmation: `NEXT_STEP: CONFIRM_UNDERSTANDING\\n[Your summary of the project/task.]`\n"
            f"3. If understanding is sufficient to create a detailed execution prompt for another AI: `NEXT_STEP: BUILD_PROMPT\\n[Key requirements for the final prompt. Suggest target handler: PLAN_AND_CODE, FILE_CREATION, ITERATION, MICRO_TASK, or CHAT.]`"
        )
        history = [ChatMessage(role=USER_ROLE, parts=[prompt])]  # type: ignore
        req_id = f"convo_analyze_{conversation.conversation_id}"
        conversation.intermediate_llm_request_id = req_id
        conversation.update_timestamp()
        backend_id, model_name, temp = self._get_llm_config_for_orchestration()
        self._backend_coordinator.start_llm_streaming_task(
            request_id=req_id, target_backend_id=backend_id, history_to_send=history,
            is_modification_response_expected=False, options={"temperature": temp},
            request_metadata={"purpose": "orchestration_analysis_next_step",
                              "conversation_id": conversation.conversation_id,
                              "project_id": conversation.current_project_id,
                              "session_id": conversation.current_session_id}
        )

    async def _request_llm_to_build_final_prompt(self, conversation: OrchestratedConversation):
        """Asks LLM to synthesize all info into a final, actionable prompt."""
        if not self._backend_coordinator: return
        history_summary = "\n".join([f"Q: {qa['q']}\nA: {qa['a']}" for qa in conversation.clarification_history])
        prompt = (
            f"Original Request: \"{conversation.original_user_request}\"\n"
            f"Conversation Summary:\n{history_summary}\n"
            f"Identified Key Insights/Requirements by Planner: {conversation.llm_insights}\n"
            f"Suggested Target Handler for Execution: {conversation.target_handler_suggestion}\n\n"
            f"Based on ALL the above, construct a comprehensive, actionable prompt. This prompt will be given to another AI specialized in the target handler's task. "
            f"Ensure it contains all necessary details, context, and constraints for successful execution. "
            f"Output ONLY the final prompt itself, without any extra conversation or explanation."
        )
        history = [ChatMessage(role=USER_ROLE, parts=[prompt])]  # type: ignore
        req_id = f"convo_build_final_prompt_{conversation.conversation_id}"
        conversation.intermediate_llm_request_id = req_id
        conversation.phase = ConversationPhase.BUILDING_FINAL_PROMPT
        conversation.update_timestamp()
        backend_id, model_name, temp = self._get_llm_config_for_orchestration()
        self._backend_coordinator.start_llm_streaming_task(
            request_id=req_id, target_backend_id=backend_id, history_to_send=history,
            is_modification_response_expected=False, options={"temperature": max(0.1, temp - 0.2)},  # Lower temp
            request_metadata={"purpose": "orchestration_build_final_prompt",
                              "conversation_id": conversation.conversation_id,
                              "project_id": conversation.current_project_id,
                              "session_id": conversation.current_session_id}
        )

    @Slot(str, ChatMessage, dict)  # type: ignore
    def _handle_orchestrator_llm_response(self, request_id: str, message: ChatMessage, metadata: dict):  # type: ignore
        purpose = metadata.get("purpose")
        conversation_id = metadata.get("conversation_id")
        if not purpose or not purpose.startswith("orchestration_") or not conversation_id: return
        conversation = self._active_conversations.get(conversation_id)
        if not conversation or conversation.intermediate_llm_request_id != request_id: return

        logger.info(f"CO: Handling LLM response for '{purpose}', convo '{conversation_id}'")
        conversation.intermediate_llm_request_id = None
        llm_response_text = message.text.strip()  # type: ignore

        if purpose == "orchestration_understanding":
            conversation.phase = ConversationPhase.GATHERING_REQUIREMENTS
            cleaned_questions = self._clean_llm_questions(llm_response_text)
            conversation.clarification_history.append({"q": cleaned_questions, "a": "<User response pending>"})
            self._post_system_message_to_ui(conversation,
                                            f"[System: To help me understand better, please answer these questions:]\n{cleaned_questions}")
        elif purpose == "orchestration_analysis_next_step":
            self._parse_analysis_and_proceed(conversation, llm_response_text)
        elif purpose == "orchestration_build_final_prompt":
            conversation.final_actionable_prompt = llm_response_text  # Assume LLM followed instructions
            conversation.phase = ConversationPhase.ROUTING_TO_HANDLER
            self._route_to_target_handler(conversation)
            # Don't pop yet, let the routed handler complete or error, then ChatManager can close it.
        conversation.update_timestamp()

    def _clean_llm_questions(self, raw_questions: str) -> str:
        """Cleans up LLM-generated questions, removing conversational filler."""
        lines = raw_questions.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Remove common conversational prefixes
            line = re.sub(r"^(Okay, |Sure, |Great, |Here are some questions:|Certainly, |I have a few questions:)\s*",
                          "", line, flags=re.IGNORECASE)
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                cleaned_lines.append(line)
            elif line and not cleaned_lines:  # If first line isn't a list item but has content
                cleaned_lines.append(f"1. {line}")  # Assume it's the first question
        return "\n".join(cleaned_lines) if cleaned_lines else raw_questions  # Return original if no cleaning done

    def _parse_analysis_and_proceed(self, conversation: OrchestratedConversation, llm_response: str):
        """Parses LLM's analysis of next step and acts accordingly."""
        response_upper = llm_response.upper()
        if response_upper.startswith("NEXT_STEP: ASK_QUESTIONS"):
            questions = llm_response[len("NEXT_STEP: ASK_QUESTIONS"):].strip()
            cleaned_questions = self._clean_llm_questions(questions)
            conversation.phase = ConversationPhase.GATHERING_REQUIREMENTS
            conversation.clarification_history.append({"q": cleaned_questions, "a": "<User response pending>"})
            self._post_system_message_to_ui(conversation, f"[System: I need a bit more info:]\n{cleaned_questions}")
        elif response_upper.startswith("NEXT_STEP: CONFIRM_UNDERSTANDING"):
            summary = llm_response[len("NEXT_STEP: CONFIRM_UNDERSTANDING"):].strip()
            conversation.phase = ConversationPhase.CONFIRMING_UNDERSTANDING
            conversation.llm_insights.append(f"LLM Understanding: {summary}")
            self._post_system_message_to_ui(conversation,
                                            f"[System: Here's my understanding:]\n{summary}\n\nIs this correct, or are there changes? (Type 'yes' or provide corrections)")
        elif response_upper.startswith("NEXT_STEP: BUILD_PROMPT"):
            details = llm_response[len("NEXT_STEP: BUILD_PROMPT"):].strip()
            handler_match = re.search(r"Target Handler:\s*(\w+)", details, re.IGNORECASE)
            if handler_match:
                conversation.target_handler_suggestion = handler_match.group(1).upper()
            else:
                conversation.target_handler_suggestion = "PLAN_AND_CODE"  # Default if not specified
            conversation.llm_insights.append(f"LLM Requirements for Prompt: {details}")
            asyncio.create_task(self._request_llm_to_build_final_prompt(conversation))
        else:
            logger.warning(
                f"CO ({conversation.conversation_id}): LLM analysis response unclear. Defaulting to more questions. Response: {llm_response[:100]}")
            cleaned_fallback_q = self._clean_llm_questions(llm_response)  # Try to clean what we got
            conversation.phase = ConversationPhase.GATHERING_REQUIREMENTS
            conversation.clarification_history.append({"q": cleaned_fallback_q, "a": "<User response pending>"})
            self._post_system_message_to_ui(conversation,
                                            f"[System: I'm still working through that. Could you clarify based on this?]\n{cleaned_fallback_q}")

    @Slot(str, str)
    def _handle_orchestrator_llm_error(self, request_id: str, error_message: str):
        """Handles LLM errors for requests made by this orchestrator."""
        convo_to_update: Optional[OrchestratedConversation] = None
        for convo in self._active_conversations.values():
            if convo.intermediate_llm_request_id == request_id:
                convo_to_update = convo;
                break
        if convo_to_update:
            logger.error(
                f"CO: LLM error for convo {convo_to_update.conversation_id} (req: {request_id}): {error_message}")
            convo_to_update.phase = ConversationPhase.ERROR
            convo_to_update.intermediate_llm_request_id = None
            self._post_system_message_to_ui(convo_to_update,
                                            f"[System Error: My apologies, I hit a snag: {error_message}. Please try rephrasing or starting over.]")
        else:
            logger.debug(f"CO: LLM error for unmanaged request_id '{request_id}'.")

    def _post_system_message_to_ui(self, conversation: OrchestratedConversation, text: str):
        """Helper to send a system message to the Chat UI."""
        if self._event_bus and conversation.current_project_id and conversation.current_session_id:
            sys_msg = ChatMessage(role=SYSTEM_ROLE, parts=[text],
                                  metadata={"conversation_id": conversation.conversation_id,
                                            "is_orchestration_msg": True})  # type: ignore
            self._event_bus.newMessageAddedToHistory.emit(conversation.current_project_id,
                                                          conversation.current_session_id, sys_msg)
        else:
            logger.warning(f"CO: Cannot post sys msg for convo {conversation.conversation_id}: Bus or P/S ID missing.")

    def _route_to_target_handler(self, conversation: OrchestratedConversation):
        """Routes the finalized prompt to the appropriate handler in ChatManager."""
        if not self._chat_manager:
            logger.error(f"CO: Cannot route for {conversation.conversation_id}: ChatManager missing.");
            return
        final_prompt = conversation.final_actionable_prompt
        handler_type = conversation.target_handler_suggestion
        if not final_prompt:
            logger.error(f"CO: Cannot route for {conversation.conversation_id}: Final prompt missing.");
            return

        self._post_system_message_to_ui(conversation,
                                        f"[System: Okay, I have a plan! Handing off for '{handler_type or 'processing'}'...]")
        logger.info(
            f"CO: Routing {conversation.conversation_id} to handler '{handler_type}' with prompt: '{final_prompt[:100]}...'")

        cm = self._chat_manager  # Alias for brevity
        original_request = conversation.original_user_request

        # Ensure ChatManager methods are called correctly
        if handler_type == "PLAN_AND_CODE" and hasattr(cm, '_handle_plan_then_code_request'):
            cm._handle_plan_then_code_request(final_prompt)
        elif handler_type == "FILE_CREATION" and hasattr(cm, '_handle_file_creation_request'):
            filename = self._user_input_handler._extract_filename_from_query(original_request)
            cm._handle_file_creation_request(final_prompt, filename)
        elif handler_type == "ITERATION" and hasattr(cm, '_handle_project_iteration_request'):
            asyncio.create_task(cm._handle_project_iteration_request(final_prompt, []))
        elif handler_type == "MICRO_TASK" and hasattr(cm, '_handle_micro_task_request'):
            asyncio.create_task(cm._handle_micro_task_request(final_prompt, []))
        elif handler_type == "CHAT" or not handler_type:
            asyncio.create_task(cm._handle_normal_chat_async(final_prompt, []))
        else:
            logger.warning(f"CO: Unknown target handler '{handler_type}'. Defaulting to normal chat.")
            asyncio.create_task(cm._handle_normal_chat_async(final_prompt, []))

        # Mark conversation as completed after routing
        conversation.phase = ConversationPhase.COMPLETED
        # Consider whether to remove from _active_conversations here or let ChatManager signal completion.
        # For now, keep it until ChatManager confirms the task started by the handler is done.
        # This allows for potential follow-up or error handling related to the routed task.
        # ChatManager would need to emit an event like "OrchestratedTaskCompleted" with conversation_id.
        # For simplicity in this iteration, we can assume routing is the end of CO's direct involvement.
        # self._active_conversations.pop(conversation.conversation_id, None)

    def get_conversation_status(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Returns the status of an active conversation."""
        convo = self._active_conversations.get(conversation_id)
        if convo:
            return {
                "id": convo.conversation_id,
                "phase": convo.phase.name,
                "intent": convo.detected_intent.name,
                "last_updated": convo.last_updated_at.isoformat()
            }
        return None

    def cleanup_completed_conversation(self, conversation_id: str):
        """Removes a completed or errored conversation from active tracking."""
        if conversation_id in self._active_conversations:
            logger.info(f"CO: Cleaning up conversation {conversation_id}.")
            del self._active_conversations[conversation_id]
        else:
            logger.debug(f"CO: Attempted to cleanup non-existent or already cleaned conversation {conversation_id}.")