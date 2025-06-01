# app/core/user_input_handler.py
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class UserInputIntent(Enum):
    NORMAL_CHAT = auto()
    FILE_CREATION_REQUEST = auto()
    PLAN_THEN_CODE_REQUEST = auto()
    PROJECT_ITERATION_REQUEST = auto()
    CONVERSATIONAL_PLANNING = auto()
    MICRO_TASK_REQUEST = auto()


@dataclass
class ProcessedInput:
    intent: UserInputIntent
    original_query: str
    processed_query: str  # Query after removing any prefixes
    confidence: float
    data: Dict[str, Any]  # Store extracted entities like filename


class UserInputHandler:
    """
    Enhanced user input handler with intent detection including micro-task and prefix support.
    Analyzes user input to determine the most probable intent and extracts relevant data.
    """

    def __init__(self):
        self.logger = logger.getChild('UserInputHandler')

        # --- Intent Detection Patterns ---
        # Order matters: more specific or higher-priority patterns should come first.

        # 1. Explicit Generation Method Prefixes (Highest Priority)
        # These allow users to force a specific generation method.
        self._explicit_method_prefixes = {
            "MICRO:": UserInputIntent.MICRO_TASK_REQUEST,
            "MICRO_TASK:": UserInputIntent.MICRO_TASK_REQUEST,
            "TRADITIONAL:": UserInputIntent.PLAN_THEN_CODE_REQUEST,  # Traditional multi-file
            "PLAN_AND_CODE:": UserInputIntent.PLAN_THEN_CODE_REQUEST,
            "PLAN:": UserInputIntent.CONVERSATIONAL_PLANNING,  # If user just wants to plan
            "CREATE_FILE:": UserInputIntent.FILE_CREATION_REQUEST,
            "ITERATE:": UserInputIntent.PROJECT_ITERATION_REQUEST,
        }

        # 2. Micro-task Generation Patterns (High Priority for auto-detection)
        # Keywords suggesting a desire for incremental, reviewed, or complex generation.
        self._micro_task_keywords = [
            "micro task", "micro-task", "microgeneration", "micro-generation",
            "function by function", "step by step generation", "generate incrementally",
            "planning oversight", "model review", "adaptive generation", "continuous planning",
            "fast generation", "quick generation", "rapid generation",  # Often implies local/iterative
            "local generation", "iterative development", "iterative coding",
            "large project", "complex system", "complex application", "complex codebase",
            "multi-file structure", "multi-module project", "multi-component system",
            "high quality generation", "reviewed code", "oversight model"
        ]

        # 3. File Creation Patterns (Specific single file requests)
        # Regex patterns to detect explicit requests to create a single named file.
        self._file_creation_patterns = [
            re.compile(
                r"\bcreate (?:a )?(?:file|script) (?:called|named|titled)\s+['\"]?([a-zA-Z0-9_\-./\s]+\.\w+)['\"]?",
                re.IGNORECASE),
            re.compile(r"\bmake (?:a )?(?:file|script)\s+['\"]?([a-zA-Z0-9_\-./\s]+\.\w+)['\"]?", re.IGNORECASE),
            re.compile(r"\bwrite (?:a )?(?:file|script) (?:called|named)\s+['\"]?([a-zA-Z0-9_\-./\s]+\.\w+)['\"]?",
                       re.IGNORECASE),
            re.compile(r"\bgenerate file\s+['\"]?([a-zA-Z0-9_\-./\s]+\.\w+)['\"]?", re.IGNORECASE),
            re.compile(r"\bsave (?:this|code|output) as\s+['\"]?([a-zA-Z0-9_\-./\s]+\.\w+)['\"]?", re.IGNORECASE),
            # Simpler pattern if filename is mentioned early
            re.compile(r"^['\"]?([a-zA-Z0-9_\-./\s]+\.\w+)['\"]?\s*(?:creation|generation|script)", re.IGNORECASE),
        ]

        # 4. Plan-then-Code Patterns (Traditional multi-file project generation)
        # Keywords suggesting a desire for a full project plan before coding.
        self._plan_code_keywords = [
            "plan and code", "plan then code", "plan then implement", "plan and build",
            "autonomous coding", "autonomous development", "autonomous implementation",
            "multi-file project", "multi file generation", "multi file creation",
            "create a complete project", "build a complete application", "generate a full system",
            "implement entire project", "full project plan", "project architecture"
        ]

        # 5. Project Iteration Patterns
        # Keywords suggesting modification or improvement of existing code/project.
        self._iteration_keywords = [
            "improve", "enhancement", "refactor", "optimize", "update this", "modify that",
            "add feature", "extend project", "fix this code", "debug this part",
            "iterate on", "review and change", "make this better"
        ]

        # 6. Conversational Planning Patterns
        # Phrases indicating the user wants to discuss or plan interactively.
        self._conversational_keywords = [
            "let's plan", "can we discuss", "talk about the design", "help me architect",
            "what's the best approach", "how should I structure", "your thoughts on this",
            "recommend a plan", "brainstorm ideas for", "outline the steps"
        ]

        # --- Complexity Analysis Keywords (used by _calculate_complexity_score) ---
        self._complex_project_type_keywords = [
            "web application", "api server", "desktop application", "gui app",
            "data pipeline", "microservice", "full stack", "system architecture",
            "multi-component", "multi-module", "enterprise solution", "production system"
        ]
        self._complex_feature_keywords = [
            "database integration", "user authentication", "user authorization", "caching mechanism",
            "advanced logging", "robust error handling", "unit testing framework", "api documentation",
            "deployment script", "monitoring setup", "real-time updates", "websocket communication",
            "async programming", "concurrent tasks", "parallel processing", "distributed system",
            "machine learning model", "data analysis pipeline", "secure transactions"
        ]
        self._multi_file_indicator_keywords = [
            "multiple files", "separate modules", "organized structure", "modular design",
            "clean architecture", "well-structured project", "component-based", "service-oriented"
        ]
        self._high_quality_indicator_keywords = [
            "production ready", "enterprise grade", "high quality code", "robust implementation",
            "maintainable solution", "scalable architecture", "professional standard", "industry best practices"
        ]
        logger.info("UserInputHandler initialized with enhanced intent detection.")

    def process_input(self, user_input: str, image_data: Optional[List[Dict[str, Any]]] = None) -> ProcessedInput:
        """Processes raw user input to determine intent and extract relevant data."""
        if image_data is None:
            image_data = []

        original_query = user_input.strip()
        processed_query = original_query  # Will be modified if prefix is stripped
        intent = UserInputIntent.NORMAL_CHAT  # Default
        confidence = 0.5  # Default confidence
        data: Dict[str, Any] = {}

        # 1. Check for explicit method prefixes
        for prefix, associated_intent in self._explicit_method_prefixes.items():
            if original_query.upper().startswith(prefix):
                intent = associated_intent
                processed_query = original_query[len(prefix):].strip()
                confidence = 0.99  # Very high confidence for explicit prefix
                data['explicit_method'] = prefix.strip(':').lower()
                logger.debug(
                    f"Detected explicit method prefix '{prefix}', intent: {intent.name}, query: '{processed_query[:50]}...'")
                # If file creation prefix, try to extract filename
                if intent == UserInputIntent.FILE_CREATION_REQUEST:
                    filename = self._extract_filename_from_query(processed_query)
                    if filename: data['filename'] = filename
                return ProcessedInput(intent, original_query, processed_query, confidence, data)

        # If no explicit prefix, proceed with auto-detection
        input_lower = processed_query.lower()  # Use potentially stripped query for keyword checks

        # 2. Micro-task (high priority for complex auto-detection)
        if any(keyword in input_lower for keyword in self._micro_task_keywords):
            intent = UserInputIntent.MICRO_TASK_REQUEST
            confidence = 0.8  # Good confidence if keywords match
            data['trigger_keywords'] = [kw for kw in self._micro_task_keywords if kw in input_lower]
            logger.debug(f"Detected micro-task keywords: {data['trigger_keywords']}")

        # Further check complexity for micro-task, potentially upgrading from plan-then-code
        complexity_score = self._calculate_complexity_score(processed_query)
        data['auto_complexity_score'] = complexity_score
        if complexity_score >= 4.5 and intent != UserInputIntent.MICRO_TASK_REQUEST:  # Threshold for auto-upgrading
            if intent == UserInputIntent.NORMAL_CHAT or intent == UserInputIntent.PLAN_THEN_CODE_REQUEST:
                logger.info(
                    f"High complexity ({complexity_score}) suggests micro-task. Overriding previous intent {intent.name}.")
                intent = UserInputIntent.MICRO_TASK_REQUEST
                confidence = 0.75  # Confidence based on complexity
        elif intent == UserInputIntent.MICRO_TASK_REQUEST:  # If already micro-task, boost confidence by complexity
            confidence = min(0.95, confidence + (complexity_score / 20.0))

        # 3. File Creation (if not already a higher priority intent)
        if intent == UserInputIntent.NORMAL_CHAT:  # Only if not already classified
            filename = self._extract_filename_from_query(processed_query)
            if filename:
                intent = UserInputIntent.FILE_CREATION_REQUEST
                confidence = 0.90  # Filename presence is a strong indicator
                data['filename'] = filename
                logger.debug(f"Detected file creation for '{filename}'")

        # 4. Plan-then-Code (if not already micro-task or file creation)
        if intent == UserInputIntent.NORMAL_CHAT:
            if any(keyword in input_lower for keyword in self._plan_code_keywords):
                intent = UserInputIntent.PLAN_THEN_CODE_REQUEST
                confidence = 0.80
                data['trigger_keywords'] = [kw for kw in self._plan_code_keywords if kw in input_lower]
                logger.debug(f"Detected plan-then-code keywords: {data['trigger_keywords']}")

        # 5. Project Iteration (check regardless of previous, but might be overridden by complexity to micro-task)
        if any(keyword in input_lower for keyword in self._iteration_keywords):
            if intent == UserInputIntent.NORMAL_CHAT or intent == UserInputIntent.PLAN_THEN_CODE_REQUEST:
                # If it looks like iteration and wasn't complex enough for micro-task yet
                intent = UserInputIntent.PROJECT_ITERATION_REQUEST
                confidence = 0.78
                data['trigger_keywords'] = [kw for kw in self._iteration_keywords if kw in input_lower]
                logger.debug(f"Detected iteration keywords: {data['trigger_keywords']}")

        # 6. Conversational Planning
        if intent == UserInputIntent.NORMAL_CHAT:  # If still normal chat, check for planning phrases
            if any(keyword in input_lower for keyword in self._conversational_keywords):
                intent = UserInputIntent.CONVERSATIONAL_PLANNING
                confidence = 0.70
                data['trigger_keywords'] = [kw for kw in self._conversational_keywords if kw in input_lower]
                logger.debug(f"Detected conversational planning keywords: {data['trigger_keywords']}")

        # If images are present and intent is still normal chat, it's likely a multimodal query
        if image_data and intent == UserInputIntent.NORMAL_CHAT:
            data['has_images'] = True
            confidence = max(confidence, 0.65)  # Slightly boost confidence if images are present for a chat
            logger.debug("Image data present with normal chat intent.")

        final_result = ProcessedInput(intent, original_query, processed_query, confidence, data)
        self.logger.info(
            f"Processed input: Intent={final_result.intent.name}, Confidence={final_result.confidence:.2f}, Data={final_result.data}")
        return final_result

    def _extract_filename_from_query(self, query: str) -> Optional[str]:
        """Extracts a filename if specified in the query using file creation patterns."""
        for pattern in self._file_creation_patterns:
            match = pattern.search(query)
            if match:
                # Group 1 usually captures the filename in these patterns
                filename = match.group(1).strip()
                # Basic validation: ensure it looks like a filename (e.g., has an extension)
                if '.' in filename and len(filename) > 3:
                    # Remove surrounding quotes if any were captured by a lenient regex part
                    filename = filename.strip("'\"")
                    return filename
        return None

    def _calculate_complexity_score(self, user_input: str) -> float:
        """Calculates a heuristic complexity score for the user input."""
        input_lower = user_input.lower()
        score = 0.0

        # Length contributes to complexity
        word_count = len(user_input.split())
        if word_count > 50:
            score += 2.0
        elif word_count > 25:
            score += 1.0
        elif word_count > 10:
            score += 0.5

        # Keywords indicating complex project types
        score += sum(1.5 for kw in self._complex_project_type_keywords if kw in input_lower)
        # Keywords indicating complex features
        score += sum(1.0 for kw in self._complex_feature_keywords if kw in input_lower)
        # Keywords indicating multi-file structure
        score += sum(1.0 for kw in self._multi_file_indicator_keywords if kw in input_lower)
        # Keywords indicating high quality requirements (which implies more complex generation)
        score += sum(0.5 for kw in self._high_quality_indicator_keywords if kw in input_lower)

        return score

    def get_intent_description(self, intent: UserInputIntent) -> str:
        """Returns a human-readable description for an intent."""
        descriptions = {
            UserInputIntent.NORMAL_CHAT: "General conversation or query.",
            UserInputIntent.FILE_CREATION_REQUEST: "Request to create a single specific file.",
            UserInputIntent.PLAN_THEN_CODE_REQUEST: "Request for traditional multi-file project generation with upfront planning.",
            UserInputIntent.PROJECT_ITERATION_REQUEST: "Request to modify or improve an existing project/code.",
            UserInputIntent.CONVERSATIONAL_PLANNING: "Desire to discuss and plan a project interactively.",
            UserInputIntent.MICRO_TASK_REQUEST: "Request for complex or high-quality code generation using micro-tasks and planning oversight."
        }
        return descriptions.get(intent, "Unknown user intent.")