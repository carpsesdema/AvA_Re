# app/llm/backend_interface.py
from abc import ABC, abstractmethod
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

try:
    # Assuming ChatMessage will be in app.models.chat_message
    from app.models.chat_message import ChatMessage
except ImportError:
    # Fallback for type hinting if ChatMessage is not available at this point
    # This might happen if files are being created out of order or in a flat structure temporarily
    ChatMessage = "ChatMessage"  # type: ignore


class BackendInterface(ABC):
    """
    Abstract Base Class defining the interface for all LLM backend adapters.
    Each specific LLM implementation (e.g., Gemini, OpenAI, Ollama) should
    inherit from this class and implement its abstract methods.
    """

    @abstractmethod
    def configure(self,
                  api_key: Optional[str],
                  model_name: str,
                  system_prompt: Optional[str] = None) -> bool:
        """
        Configures the backend adapter with necessary credentials and settings.

        Args:
            api_key: The API key for the LLM service, if applicable.
            model_name: The specific model to be used (e.g., "gemini-1.5-pro-latest").
            system_prompt: An optional system-level prompt or instruction for the model.

        Returns:
            True if configuration was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def get_response_stream(self,
                                  history: List[ChatMessage],  # type: ignore
                                  options: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """
        Gets a response from the LLM as an asynchronous stream of text chunks.

        Args:
            history: A list of ChatMessage objects representing the conversation history.
            options: A dictionary of additional options for the LLM request (e.g., temperature).

        Yields:
            str: Chunks of the LLM's response as they become available.

        Raises:
            RuntimeError: If the adapter is not configured or other critical errors occur.
            ValueError: If input parameters are invalid (e.g., empty history with no system prompt).
        """
        # This is an abstract method, the `if False: yield ''` is a common pattern
        # to make linters and type checkers happy that an AsyncGenerator is returned.
        if False:  # type: ignore
            yield ''  # type: ignore
        pass

    @abstractmethod
    def get_last_error(self) -> Optional[str]:
        """
        Returns the last error message encountered by the adapter, if any.

        Returns:
            A string containing the error message, or None if no error occurred.
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """
        Checks if the backend adapter has been successfully configured.

        Returns:
            True if configured, False otherwise.
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Retrieves a list of available model names for this backend.
        This might involve an API call or return a predefined list.

        Returns:
            A list of strings, where each string is an available model name.
        """
        pass

    @abstractmethod
    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        """
        Retrieves the token usage from the last LLM interaction, if available.

        Returns:
            A tuple (prompt_tokens, completion_tokens), or None if not available/supported.
        """
        pass