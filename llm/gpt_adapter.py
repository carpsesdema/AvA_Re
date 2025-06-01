# app/llm/gpt_adapter.py
import asyncio
import logging
import os
import base64  # For image data if using vision models
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

try:
    # Assuming BackendInterface is in the same package
    from .backend_interface import BackendInterface
    # Models are now in app.models
    from app.models.chat_message import ChatMessage, MODEL_ROLE, USER_ROLE, SYSTEM_ROLE
except ImportError as e_imp:
    logging.getLogger(__name__).critical(f"GPTAdapter: Critical import error: {e_imp}", exc_info=True)
    # Fallback types for type hinting
    BackendInterface = type("BackendInterface", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MODEL_ROLE, USER_ROLE, SYSTEM_ROLE = "model", "user", "system"
    raise

# Attempt to import OpenAI library
try:
    import openai
    from openai import APIError, AuthenticationError, RateLimitError, NotFoundError, APIConnectionError, \
        APITimeoutError  # type: ignore

    # Newer versions might use different exception paths, e.g., openai.error.AuthenticationError
    # Adjust if necessary based on your openai library version.
    OPENAI_API_LIBRARY_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    # Define dummy exception types for type hinting and preventing runtime errors
    APIError = type("APIError", (Exception,), {})  # type: ignore
    AuthenticationError = type("AuthenticationError", (APIError,), {})  # type: ignore
    RateLimitError = type("RateLimitError", (APIError,), {})  # type: ignore
    NotFoundError = type("NotFoundError", (APIError,), {})  # type: ignore
    APIConnectionError = type("APIConnectionError", (APIError,), {})  # type: ignore
    APITimeoutError = type("APITimeoutError", (APIError,), {})  # type: ignore
    OPENAI_API_LIBRARY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "GPTAdapter: 'openai' library not found. Please install it: pip install openai"
    )

logger = logging.getLogger(__name__)

# Sentinel object for stream iteration helper
_SENTINEL_GPT = object()


def _blocking_next_or_sentinel_gpt(iterator: Any) -> Any:
    """Helper to get next item from a synchronous iterator or return sentinel if exhausted."""
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL_GPT
    except Exception as e_next:  # Catch other errors during iteration
        # This could be an error within the OpenAI library's stream handling
        logger.error(f"Error in OpenAI stream iterator's next() call: {type(e_next).__name__} - {e_next}",
                     exc_info=True)
        # Re-raise as a RuntimeError to be caught by the main streaming logic
        raise RuntimeError(f"OpenAI stream iterator error: {e_next}") from e_next


class GPTAdapter(BackendInterface):
    """
    Adapter for interacting with OpenAI's GPT models via the openai SDK.
    """

    def __init__(self):
        self._client: Optional[openai.OpenAI] = None  # type: ignore # Stores the OpenAI client instance
        self._model_name: Optional[str] = None
        self._system_prompt: Optional[str] = None
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None
        logger.info("GPTAdapter initialized.")

    def configure(self,
                  api_key: Optional[str],  # API key can be passed or loaded from env
                  model_name: str,
                  system_prompt: Optional[str] = None) -> bool:
        logger.info(f"GPTAdapter: Configuring. Model: {model_name}. System Prompt: {'Yes' if system_prompt else 'No'}")
        # Reset state
        self._client = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not OPENAI_API_LIBRARY_AVAILABLE:
            self._last_error = "OpenAI API library ('openai') not installed."
            logger.error(self._last_error)
            return False

        effective_api_key = api_key
        if not effective_api_key or not effective_api_key.strip():
            effective_api_key = os.getenv("OPENAI_API_KEY")

        if not effective_api_key or not effective_api_key.strip():
            self._last_error = "OpenAI API key not provided and not found in OPENAI_API_KEY environment variable."
            logger.error(self._last_error)
            return False

        if not model_name:
            self._last_error = "Model name is required for GPT configuration."
            logger.error(self._last_error)
            return False

        try:
            # Initialize the OpenAI client
            # The new OpenAI SDK (v1.0.0+) uses `openai.OpenAI()`
            self._client = openai.OpenAI(api_key=effective_api_key)  # type: ignore
            self._model_name = model_name
            self._system_prompt = system_prompt.strip() if isinstance(system_prompt,
                                                                      str) and system_prompt.strip() else None
            self._is_configured = True
            logger.info(f"GPTAdapter configured successfully for model '{self._model_name}'.")
            return True
        except AuthenticationError as auth_err:  # type: ignore
            self._last_error = f"OpenAI Authentication Error: {auth_err}. Check your API key."
            logger.error(self._last_error, exc_info=True)
        except APIConnectionError as conn_err:  # type: ignore
            self._last_error = f"OpenAI API Connection Error: {conn_err}."
            logger.error(self._last_error, exc_info=True)
        except Exception as e:  # Catch-all for other unexpected errors
            self._last_error = f"Unexpected error configuring OpenAI for model '{model_name}': {type(e).__name__} - {e}"
            logger.error(self._last_error, exc_info=True)

        self._is_configured = False
        return False

    def is_configured(self) -> bool:
        return self._is_configured and self._client is not None

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self,
                                  history: List[ChatMessage],  # type: ignore
                                  options: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        logger.info(
            f"GPTAdapter: Generating stream. Model: {self._model_name}, History items: {len(history)}, Options: {options}")
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._client:
            self._last_error = "GPTAdapter is not configured or client missing."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)

        messages_for_api = self._format_history_for_api(history)
        if not messages_for_api:  # OpenAI API requires at least one message if no system prompt (which is added in _format_history)
            # If system prompt is also None after formatting, then it's an issue.
            # _format_history_for_api adds system prompt if self._system_prompt exists.
            self._last_error = "Cannot send request: No valid messages to send to OpenAI API."
            logger.error(self._last_error)
            raise ValueError(self._last_error)

        api_params: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages_for_api,
            "stream": True
        }
        if options:
            if "temperature" in options and isinstance(options["temperature"], (float, int)):
                api_params["temperature"] = float(options["temperature"])
            if "max_tokens" in options and isinstance(options["max_tokens"], int) and options["max_tokens"] > 0:
                api_params["max_tokens"] = options["max_tokens"]
            # Add other OpenAI specific options like top_p, presence_penalty, frequency_penalty
            if "top_p" in options and isinstance(options["top_p"], (float, int)):
                api_params["top_p"] = float(options["top_p"])

        logger.debug(
            f"Sending request to OpenAI with params (excluding messages): { {k: v for k, v in api_params.items() if k != 'messages'} }")

        sync_iterator = None
        try:
            # The SDK call is blocking, so run it in a thread
            def _initiate_openai_stream_call_in_thread():
                if not self._client or not hasattr(self._client, 'chat') or not hasattr(self._client.chat,
                                                                                        'completions'):
                    raise RuntimeError("OpenAI client or chat completions endpoint not available.")
                return self._client.chat.completions.create(**api_params)

            sync_iterator = await asyncio.to_thread(_initiate_openai_stream_call_in_thread)

            chunk_count = 0
            # Iterate over the synchronous iterator, fetching next item in a thread
            while True:
                chunk_obj = await asyncio.to_thread(_blocking_next_or_sentinel_gpt, sync_iterator)
                if chunk_obj is _SENTINEL_GPT:
                    break  # End of stream

                chunk_count += 1
                # logger.debug(f"OpenAI raw chunk #{chunk_count} for {self._model_name}: Type {type(chunk_obj)}")

                if not hasattr(chunk_obj, 'choices') or not chunk_obj.choices:
                    # Sometimes a final chunk might not have choices but other metadata (like usage in older versions)
                    # logger.debug("OpenAI chunk has no choices, skipping delta processing.")
                    # Check for usage stats on such chunks if applicable (newer SDKs put it on the final response object)
                    if hasattr(chunk_obj, 'usage') and chunk_obj.usage:
                        self._last_prompt_tokens = chunk_obj.usage.prompt_tokens
                        self._last_completion_tokens = chunk_obj.usage.completion_tokens
                    continue

                delta = chunk_obj.choices[0].delta
                finish_reason = chunk_obj.choices[0].finish_reason

                if delta and delta.content:
                    yield delta.content

                # Usage stats in streaming are often on the *last* chunk or on the response object itself
                # For OpenAI, `usage` is typically on the final non-streaming response or sometimes on the last chunk of a stream.
                # The current openai SDK (v1+) might provide 'usage' on the Stream object after iteration or on the final chunk.
                # Let's assume for now it might appear on the last chunk if finish_reason is present.
                # A more reliable way is to get it from the response object if the SDK provides a method post-streaming.
                # For now, this is a common pattern for some SDK versions.
                if hasattr(chunk_obj, 'usage') and chunk_obj.usage:  # Check if usage is on this chunk
                    self._last_prompt_tokens = chunk_obj.usage.prompt_tokens
                    self._last_completion_tokens = chunk_obj.usage.completion_tokens

                if finish_reason:
                    logger.info(f"OpenAI stream finished. Reason: {finish_reason}. Total Chunks: {chunk_count}")
                    # If the SDK provides total usage on the iterator/response object *after* streaming:
                    if hasattr(sync_iterator, 'usage') and sync_iterator.usage:  # Hypothetical, check SDK docs
                        self._last_prompt_tokens = sync_iterator.usage.prompt_tokens
                        self._last_completion_tokens = sync_iterator.usage.completion_tokens
                    elif hasattr(chunk_obj, 'usage') and chunk_obj.usage:  # If usage is on the *final* chunk
                        self._last_prompt_tokens = chunk_obj.usage.prompt_tokens
                        self._last_completion_tokens = chunk_obj.usage.completion_tokens
                    break

                if chunk_count % 5 == 0: await asyncio.sleep(0)  # Cooperative yield

            if self._last_prompt_tokens is not None or self._last_completion_tokens is not None:
                logger.info(
                    f"OpenAI Token Usage ({self._model_name}): Prompt={self._last_prompt_tokens or 'N/A'}, Completion={self._last_completion_tokens or 'N/A'}")

        except AuthenticationError as e:  # type: ignore
            self._last_error = f"OpenAI API Authentication Error: {e}"
            raise RuntimeError(self._last_error) from e
        except RateLimitError as e:  # type: ignore
            self._last_error = f"OpenAI API Rate Limit Error: {e}"
            raise RuntimeError(self._last_error) from e
        except APIConnectionError as e:  # type: ignore
            self._last_error = f"OpenAI API Connection Error: {e}"
            raise RuntimeError(self._last_error) from e
        except APITimeoutError as e:  # type: ignore
            self._last_error = f"OpenAI API Timeout Error: {e}"
            raise RuntimeError(self._last_error) from e
        except NotFoundError as e:  # type: ignore
            self._last_error = f"OpenAI API Not Found Error (model '{self._model_name}' invalid or endpoint issue?): {e}"
            raise RuntimeError(self._last_error) from e
        except APIError as e:  # type: ignore
            self._last_error = f"OpenAI API Error: {type(e).__name__} - {e}"
            raise RuntimeError(self._last_error) from e
        except RuntimeError as e_rt:  # Catch re-raised errors from _blocking_next_or_sentinel_gpt
            if not self._last_error: self._last_error = f"Runtime error during OpenAI stream: {e_rt}"
            logger.error(self._last_error, exc_info=True)
            raise  # Re-raise critical runtime errors
        except Exception as e_general:
            if not self._last_error: self._last_error = f"Unexpected error in OpenAI stream ({self._model_name}): {type(e_general).__name__} - {e_general}"
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from e_general

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:  # type: ignore
        """Formats chat history for the OpenAI API, including system prompt and multimodal content."""
        openai_messages: List[Dict[str, Any]] = []

        # Add system prompt first if it exists
        if self._system_prompt:
            openai_messages.append({"role": "system", "content": self._system_prompt})

        for msg in history:  # type: ignore
            role_for_api: Optional[str] = None
            if msg.role == USER_ROLE:
                role_for_api = "user"
            elif msg.role == MODEL_ROLE:
                role_for_api = "assistant"
            elif msg.role == SYSTEM_ROLE:
                # If a global system prompt is already added, decide how to handle additional system messages.
                # Option 1: Skip them. Option 2: Prepend/Append (might confuse model). Option 3: Treat as user/assistant (less ideal).
                # For now, if global system prompt exists, we skip others. If not, the first one encountered is used.
                if not self._system_prompt and not any(m['role'] == 'system' for m in openai_messages):
                    role_for_api = "system"
                else:
                    logger.debug(
                        f"GPTAdapter: Skipping SYSTEM_ROLE message from history as a global system_prompt is already set or one was already added: '{msg.text[:50]}...'")
                    continue
            else:  # Skip unknown roles
                logger.debug(f"GPTAdapter: Skipping message with unhandled role: {msg.role}")
                continue

            # OpenAI API expects 'content' to be a string for simple text,
            # or a list of parts for multimodal messages (text and images).
            message_content_parts_for_api: List[Dict[str, Any]] = []

            text_content = msg.text  # .text property gets all text parts concatenated
            if text_content and text_content.strip():
                message_content_parts_for_api.append({"type": "text", "text": text_content})

            # Image handling for vision-capable models
            # Check if model is vision-capable (e.g., "gpt-4-vision-preview", "gpt-4-turbo", "gpt-4o")
            if msg.has_images and self._model_name and \
                    (
                            "vision" in self._model_name.lower() or "gpt-4-turbo" in self._model_name.lower() or "gpt-4o" in self._model_name.lower()):
                for img_part_dict in msg.image_parts:  # image_parts is List[Dict[str, Any]]
                    if isinstance(img_part_dict, dict) and img_part_dict.get("type") == "image" and \
                            img_part_dict.get("mime_type") and img_part_dict.get("data"):  # data is base64
                        # OpenAI expects image_url format for base64 images
                        image_url_data = f"data:{img_part_dict['mime_type']};base64,{img_part_dict['data']}"
                        message_content_parts_for_api.append(
                            {"type": "image_url", "image_url": {"url": image_url_data}})

            final_content_value_for_api: Any
            if not message_content_parts_for_api:
                # If a system message has no content parts (e.g. only a role was provided, which is unusual)
                # or a user/assistant message is empty (which API might reject).
                # For system role, empty content is okay. For user/assistant, it might be an issue.
                if role_for_api in ["user", "assistant"]:
                    logger.debug(f"GPTAdapter: Skipping empty user/assistant message for role {role_for_api}.")
                    continue  # Skip empty user/assistant messages
                else:  # system role
                    final_content_value_for_api = ""  # Empty content for system is fine
            elif len(message_content_parts_for_api) == 1 and message_content_parts_for_api[0]["type"] == "text":
                # If only text, content is just the string
                final_content_value_for_api = message_content_parts_for_api[0]["text"]
            else:
                # If multiple parts (text and images, or multiple text parts - though ChatMessage.text consolidates text),
                # content is the list of these parts.
                final_content_value_for_api = message_content_parts_for_api

            openai_messages.append({"role": role_for_api, "content": final_content_value_for_api})

        return openai_messages

    def get_available_models(self) -> List[str]:
        """Retrieves available GPT models from OpenAI, with fallbacks."""
        self._last_error = None
        if not OPENAI_API_LIBRARY_AVAILABLE:
            self._last_error = "OpenAI API library ('openai') not installed."
            return []
        if not self.is_configured() or not self._client:
            # Configuration (API key) is essential for listing models.
            self._last_error = "GPTAdapter not configured (API key likely missing). Cannot fetch models."
            logger.warning(self._last_error)
            return []  # Return empty list or a predefined minimal list

        fetched_models: List[str] = []
        try:
            logger.debug("Fetching available models from OpenAI...")
            # The new SDK uses client.models.list()
            model_list_response = self._client.models.list()

            # Filter for chat-completion models, and try to exclude embeddings, edits, etc.
            # This filtering can be heuristic as OpenAI doesn't always clearly flag model purpose in the ID.
            chat_model_indicators = ("gpt-4", "gpt-3.5-turbo")  # Key prefixes/indicators
            # Keywords for models typically NOT used for chat completions
            non_chat_keywords = ("embedding", "instruct", "davinci", "curie", "babbage", "ada",
                                 "text-", "code-", "edit-", "audio", "image", "-dalle", "whisper")

            for model_obj in model_list_response.data:
                model_id_lower = model_obj.id.lower()
                is_potential_chat_model = any(indicator in model_id_lower for indicator in chat_model_indicators)

                if is_potential_chat_model:
                    # Further refine: avoid models that are explicitly non-chat despite having gpt-3.5/4 in name
                    is_non_chat_type = False
                    # Check if it contains a non-chat keyword *unless* it's a known chat model variant that might include such words (e.g. gpt-4-turbo often handles text)
                    if any(kw in model_id_lower for kw in non_chat_keywords):
                        # If it's a "text-" model but not clearly a chat variant (like turbo), assume non-chat
                        if model_id_lower.startswith("text-") and not (
                                "turbo" in model_id_lower or "gpt-4" in model_id_lower):
                            is_non_chat_type = True
                        # If it's an "instruct" model but not a clear chat variant
                        elif "instruct" in model_id_lower and not (
                                "turbo" in model_id_lower or "gpt-4" in model_id_lower):
                            is_non_chat_type = True
                        # Add other specific exclusions if needed.
                        # This logic is heuristic. For example, "gpt-4-vision-preview" *is* a chat model.
                        # "gpt-4-turbo" is also a chat model. "gpt-4o" is a chat model.
                        # The key is to ensure the model supports the Chat Completions API structure.
                        # The initial `chat_model_indicators` is usually quite good.
                        # The `non_chat_keywords` helps filter out clear non-chat ones.
                        # A model like "text-davinci-003" is not for chat.
                        # "gpt-3.5-turbo-instruct" is for completions, not chat.

                    if not is_non_chat_type:
                        fetched_models.append(model_obj.id)
            logger.info(f"Fetched {len(fetched_models)} potential chat models from OpenAI.")

        except AuthenticationError as e_auth:  # type: ignore
            self._last_error = f"OpenAI API Authentication Error while listing models: {e_auth}"
            logger.error(self._last_error)
        except APIError as e_api:  # type: ignore
            self._last_error = f"OpenAI API Error while listing models: {e_api}"
            logger.error(self._last_error)
        except Exception as e:  # Catch-all
            self._last_error = f"Unexpected error fetching OpenAI models: {type(e).__name__} - {e}"
            logger.error(self._last_error, exc_info=True)

        if self._last_error and not fetched_models:  # If an error occurred and we got no models
            # Provide a common default list as a fallback
            default_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            logger.warning(f"Using default model list due to error: {default_models}")
            return default_models

        # Sort the fetched models: gpt-4o, gpt-4-turbo, other gpt-4, gpt-3.5-turbo-16k, other gpt-3.5
        def sort_key_openai(model_id: str) -> Tuple[int, str]:
            model_id_lower = model_id.lower()
            if "gpt-4o" in model_id_lower: return 0, model_id_lower
            if "gpt-4-turbo" in model_id_lower: return 1, model_id_lower
            if "gpt-4" in model_id_lower: return 2, model_id_lower  # Other gpt-4 variants
            if "gpt-3.5-turbo-16k" in model_id_lower: return 3, model_id_lower
            if "gpt-3.5-turbo" in model_id_lower: return 4, model_id_lower
            return 5, model_id_lower  # Others

        final_model_list = sorted(list(set(fetched_models)), key=sort_key_openai)

        # Ensure the currently configured model is in the list and at the top
        if self._model_name and self._is_configured:
            if self._model_name not in final_model_list:
                final_model_list.insert(0, self._model_name)
            # Re-sort to place the current model at the top if it's already present
            final_model_list.sort(key=lambda x: (x != self._model_name, *sort_key_openai(x)))

        logger.debug(f"Returning OpenAI models: {final_model_list}")
        return final_model_list

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return self._last_prompt_tokens, self._last_completion_tokens
        return None