# app/llm/gemini_adapter.py
import asyncio
import logging
import os
import sys  # Not directly used but often good for pathing or version checks
import base64  # For image data handling
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

try:
    # Assuming BackendInterface is in the same package
    from .backend_interface import BackendInterface
    # Models are now in app.models
    from models.chat_message import ChatMessage, MODEL_ROLE, USER_ROLE
except ImportError as e_imp:
    logging.getLogger(__name__).critical(f"GeminiAdapter: Critical import error: {e_imp}", exc_info=True)
    # Fallback types for type hinting if imports fail
    BackendInterface = type("BackendInterface", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MODEL_ROLE, USER_ROLE = "model", "user"
    raise  # Re-raise to ensure the issue is visible

# Attempt to import Google Generative AI library
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory, GenerationConfig
    from google.generativeai.types.generation_types import GenerateContentResponse, BlockedPromptException
    # Import specific API error types for more granular error handling
    from google.api_core.exceptions import GoogleAPIError, ClientError, PermissionDenied, ResourceExhausted, \
        InvalidArgument

    API_LIBRARY_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    # Define dummy types if library is not available, for type hinting and preventing runtime errors on attribute access
    HarmCategory = type("HarmCategory", (object,), {  # type: ignore
        'HARM_CATEGORY_HARASSMENT': None, 'HARM_CATEGORY_HATE_SPEECH': None,
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': None, 'HARM_CATEGORY_DANGEROUS_CONTENT': None
    })
    HarmBlockThreshold = type("HarmBlockThreshold", (object,), {'BLOCK_NONE': None})  # type: ignore
    GenerationConfig = type("GenerationConfig", (object,), {})  # type: ignore
    GenerateContentResponse = type("GenerateContentResponse", (object,), {})  # type: ignore
    BlockedPromptException = type("BlockedPromptException", (Exception,), {})  # type: ignore
    # Dummy API error types
    GoogleAPIError = type("GoogleAPIError", (Exception,), {})  # type: ignore
    ClientError = type("ClientError", (GoogleAPIError,), {})  # type: ignore
    PermissionDenied = type("PermissionDenied", (ClientError,), {})  # type: ignore
    ResourceExhausted = type("ResourceExhausted", (ClientError,), {})  # type: ignore
    InvalidArgument = type("InvalidArgument", (ClientError,), {})  # type: ignore
    API_LIBRARY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "GeminiAdapter: google-generativeai library not found. Please install it: pip install google-generativeai"
    )

logger = logging.getLogger(__name__)


class GeminiAdapter(BackendInterface):
    """
    Adapter for interacting with Google Gemini models via the google-generativeai SDK.
    """

    def __init__(self):
        self._model: Optional[genai.GenerativeModel] = None  # type: ignore
        self._model_name: Optional[str] = None
        self._system_prompt: Optional[str] = None  # Gemini now supports system_instruction
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None
        logger.info("GeminiAdapter initialized.")

    def configure(self,
                  api_key: Optional[str],  # API key can be passed directly or loaded from env
                  model_name: str,
                  system_prompt: Optional[str] = None) -> bool:
        logger.info(
            f"GeminiAdapter: Configuring. Model: {model_name}. System Prompt: {'Yes' if system_prompt else 'No'}")
        self._model = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Gemini API library (google-generativeai) not installed."
            logger.error(self._last_error)
            return False

        # Prioritize passed api_key, then environment variables
        effective_api_key = api_key
        if not effective_api_key or not effective_api_key.strip():
            effective_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not effective_api_key:
            self._last_error = "Gemini API key not provided and not found in GOOGLE_API_KEY/GEMINI_API_KEY environment variables."
            logger.error(self._last_error)
            return False

        if not model_name:
            self._last_error = "Model name is required for Gemini configuration."
            logger.error(self._last_error)
            return False

        try:
            genai.configure(api_key=effective_api_key)  # type: ignore

            # Define safety settings - typically set to BLOCK_NONE for more permissive generation in dev tools
            # Adjust these based on application requirements and content policies.
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,  # type: ignore
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,  # type: ignore
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,  # type: ignore
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,  # type: ignore
            }

            # Handle system prompt (system_instruction in Gemini API)
            effective_system_instruction = system_prompt.strip() if system_prompt and system_prompt.strip() else None

            self._model = genai.GenerativeModel(  # type: ignore
                model_name=model_name,
                safety_settings=safety_settings,
                system_instruction=effective_system_instruction  # Pass system prompt here
            )
            self._model_name = model_name
            self._system_prompt = effective_system_instruction  # Store what was used
            self._is_configured = True
            logger.info(f"GeminiAdapter configured successfully for model '{self._model_name}'.")
            return True
        except ValueError as ve:  # Often related to API key format or other config issues
            self._last_error = f"Gemini Configuration Error (ValueError): {ve}."
            logger.error(self._last_error, exc_info=True)
        except InvalidArgument as iae:  # Specific API error for bad arguments
            self._last_error = f"Gemini API Error (Invalid Argument): {iae}. This might be due to an unsupported model name or invalid API key format."
            logger.error(self._last_error, exc_info=True)
        except PermissionDenied as pde:  # API key issues or model access
            self._last_error = f"Gemini API Error (Permission Denied): {pde}. Check API key and model access permissions."
            logger.error(self._last_error, exc_info=True)
        except Exception as e:  # Catch-all for other unexpected errors during configuration
            self._last_error = f"Unexpected error configuring Gemini model '{model_name}': {type(e).__name__} - {e}"
            logger.error(self._last_error, exc_info=True)

        self._is_configured = False
        return False

    def is_configured(self) -> bool:
        return self._is_configured and self._model is not None

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self,
                                  history: List[ChatMessage],  # type: ignore
                                  options: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        logger.info(
            f"GeminiAdapter: Generating stream. Model: {self._model_name}, History items: {len(history)}, Options: {options}")
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._model:
            self._last_error = "Adapter not configured or Gemini model object missing."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)  # Critical error, adapter must be configured

        gemini_history_api_format = self._format_history_for_api(history)

        # If there's a system prompt, we can send a request even if user history is empty.
        # If no system prompt, and no user history, then it's an invalid request.
        if not gemini_history_api_format and not self._system_prompt:
            self._last_error = "Cannot send request: No valid messages in history and no system instruction is set."
            logger.error(self._last_error)
            raise ValueError(self._last_error)

        logger.debug(
            f"Sending {len(gemini_history_api_format)} formatted messages to model '{self._model_name}'. System instruction: {'Yes' if self._system_prompt else 'No'}")

        generation_config_dict = {}
        if options:
            if "temperature" in options and isinstance(options["temperature"], (float, int)):
                generation_config_dict["temperature"] = float(options["temperature"])
                logger.info(f"Applying temperature from options: {generation_config_dict['temperature']}")
            # Add other options like top_p, top_k, max_output_tokens if needed
            if "max_tokens" in options and isinstance(options["max_tokens"], int):
                generation_config_dict["max_output_tokens"] = options["max_tokens"]
                logger.info(f"Applying max_output_tokens: {generation_config_dict['max_output_tokens']}")

        effective_generation_config = GenerationConfig(
            **generation_config_dict) if generation_config_dict else None  # type: ignore

        # Use asyncio.to_thread for the blocking SDK call
        try:
            # The generate_content call itself is blocking, so run it in a thread
            # The `stream=True` makes the *response object* an iterable, not the call itself async.
            response_iterator = await asyncio.to_thread(
                self._model.generate_content,  # type: ignore
                contents=gemini_history_api_format,
                stream=True,
                generation_config=effective_generation_config
            )

            chunk_count = 0
            # Iterate over the response (which is an iterable of chunks when stream=True)
            # This iteration also happens in the thread if not careful,
            # so we need to ensure we yield from the async generator correctly.
            # The `async for` construct is tricky with `asyncio.to_thread` on iterables.
            # A common pattern is to wrap the iterator's next() call.

            if not hasattr(response_iterator, '__iter__'):  # Should be GenerateContentResponse which is iterable
                logger.error(
                    f"Gemini response object is not iterable as expected for model {self._model_name}. Type: {type(response_iterator)}")
                # Handle potential non-streaming error response (e.g., if prompt was immediately blocked)
                if hasattr(response_iterator, 'prompt_feedback') and response_iterator.prompt_feedback and \
                        hasattr(response_iterator.prompt_feedback,
                                'block_reason') and response_iterator.prompt_feedback.block_reason:  # type: ignore
                    err_msg = f"Content blocked by API: {response_iterator.prompt_feedback.block_reason}."  # type: ignore
                    self._last_error = err_msg
                    yield f"[SYSTEM ERROR: {err_msg}]"
                elif hasattr(response_iterator, 'text'):  # If it's a non-streaming successful response
                    yield response_iterator.text  # type: ignore
                else:  # Unknown error state
                    self._last_error = "Gemini API did not return an iterable stream or directly readable response."
                    yield f"[SYSTEM ERROR: {self._last_error}]"
                # Attempt to get usage metadata even from non-iterable response
                if response_iterator and hasattr(response_iterator,
                                                 'usage_metadata') and response_iterator.usage_metadata:  # type: ignore
                    self._process_usage_metadata(response_iterator.usage_metadata)  # type: ignore
                return

            # Iterate over chunks (this part will run within the thread if not handled carefully)
            # To make it truly async yielding, we'd typically fetch each chunk in the thread.
            for chunk in response_iterator:  # type: ignore
                chunk_count += 1
                # logger.debug(f"Gemini raw chunk #{chunk_count} for {self._model_name}: Type {type(chunk)}")

                # Check for immediate blocking reasons in the chunk itself (though less common than overall prompt_feedback)
                if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback and \
                        hasattr(chunk.prompt_feedback, 'block_reason') and chunk.prompt_feedback.block_reason:
                    err_msg = f"Content blocked by API (chunk feedback): {chunk.prompt_feedback.block_reason}."
                    self._last_error = err_msg
                    logger.error(self._last_error)
                    yield f"[SYSTEM ERROR: {err_msg}]"
                    return  # Stop streaming if a chunk indicates a block

                # Process candidates and parts from the chunk
                text_parts_from_chunk = []
                if hasattr(chunk, 'parts') and chunk.parts:  # Simpler structure in newer versions
                    for part_item in chunk.parts:
                        if hasattr(part_item, 'text') and part_item.text is not None:
                            text_parts_from_chunk.append(part_item.text)
                elif hasattr(chunk,
                             'text') and chunk.text is not None:  # Direct text in chunk (older versions or simple cases)
                    text_parts_from_chunk.append(chunk.text)
                elif hasattr(chunk, 'candidates') and chunk.candidates:  # More complex structure with candidates
                    for candidate in chunk.candidates:
                        # Check candidate-level finish reason or safety ratings
                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason and \
                                candidate.finish_reason.name not in ["STOP", "MAX_TOKENS", "UNSPECIFIED", "NULL",
                                                                     "FINISH_REASON_UNSPECIFIED"]:
                            err_msg = f"Generation stopped by API (candidate finish reason): {candidate.finish_reason.name}."
                            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                ratings_str = "; ".join(
                                    [f"{sr.category.name}: {sr.probability.name}" for sr in candidate.safety_ratings])
                                err_msg += f" Safety Details: {ratings_str}"
                            self._last_error = err_msg
                            logger.error(self._last_error)
                            yield f"[SYSTEM ERROR: {err_msg}]"
                            # Attempt to get usage metadata from the main response_iterator before returning
                            if hasattr(response_iterator,
                                       'usage_metadata') and response_iterator.usage_metadata:  # type: ignore
                                self._process_usage_metadata(response_iterator.usage_metadata)  # type: ignore
                            return

                        if hasattr(candidate, 'content') and candidate.content and \
                                hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for part_item_cand in candidate.content.parts:
                                if hasattr(part_item_cand, 'text') and part_item_cand.text is not None:
                                    text_parts_from_chunk.append(part_item_cand.text)

                if text_parts_from_chunk:
                    yield "".join(text_parts_from_chunk)

                # Cooperative yield for GUI responsiveness if many small chunks
                if chunk_count % 5 == 0:  # Adjust frequency as needed
                    await asyncio.sleep(0)

            logger.info(f"Gemini stream for {self._model_name} iteration completed. Total Chunks: {chunk_count}.")

            # After loop, get usage metadata from the main response_iterator object
            # This is available on the `GenerateContentResponse` object itself, not individual chunks.
            if hasattr(response_iterator, 'usage_metadata') and response_iterator.usage_metadata:  # type: ignore
                self._process_usage_metadata(response_iterator.usage_metadata)  # type: ignore
            else:
                logger.warning(
                    f"Gemini usage_metadata not found on the main stream response object for {self._model_name}.")

        except BlockedPromptException as bpe:  # If prompt is blocked before streaming starts
            self._last_error = f"Gemini API Error: Prompt blocked by content safety filter. Details: {bpe.args}"
            logger.error(self._last_error, exc_info=True)
            yield f"[SYSTEM ERROR: {self._last_error}]"
            return  # Stop generation
        except InvalidArgument as iae:
            self._last_error = f"Gemini API Error (Invalid Argument): {iae}."
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from iae  # Re-raise as critical error
        except PermissionDenied as pde:
            self._last_error = f"Gemini API Error (Permission Denied): {pde}."
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from pde  # Re-raise
        except ResourceExhausted as ree:
            self._last_error = f"Gemini API Error (Resource Exhausted/Quota): {ree}."
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from ree  # Re-raise
        except GoogleAPIError as gae:  # Catch other Google API errors
            self._last_error = f"Gemini API Error: {type(gae).__name__} - {gae}."
            logger.error(self._last_error, exc_info=True)
            # Decide whether to re-raise or yield error message
            if isinstance(gae, ClientError):  # More specific client-side errors
                raise RuntimeError(self._last_error) from gae
            yield f"[SYSTEM ERROR: {self._last_error}]"  # For other API errors
            return
        except Exception as e_general:
            # Catch-all for other unexpected errors during the streaming process
            if not self._last_error:  # Only set if not already set by a more specific handler
                self._last_error = f"Error during Gemini stream processing ({self._model_name}): {type(e_general).__name__} - {e_general}"
            logger.error(self._last_error, exc_info=True)
            if not isinstance(e_general, RuntimeError):  # Avoid double-wrapping RuntimeErrors
                yield f"[SYSTEM ERROR: {self._last_error}]"  # Yield error to stream
            else:  # Re-raise if it's already a RuntimeError we want to propagate
                raise
            return

    def _process_usage_metadata(self, usage_metadata_obj):
        """Helper to process usage_metadata from Gemini response."""
        if not usage_metadata_obj:
            logger.warning(f"No usage_metadata object provided for {self._model_name}.")
            return

        self._last_prompt_tokens = getattr(usage_metadata_obj, 'prompt_token_count', None)
        # For completion tokens, Gemini often uses 'candidates_token_count' for the sum of tokens in all candidates' responses
        # or 'total_token_count' which includes prompt + completion.
        self._last_completion_tokens = getattr(usage_metadata_obj, 'candidates_token_count', None)

        if self._last_completion_tokens is None:  # Fallback if candidates_token_count is not present
            if hasattr(usage_metadata_obj, 'total_token_count') and self._last_prompt_tokens is not None:
                self._last_completion_tokens = usage_metadata_obj.total_token_count - self._last_prompt_tokens

        if self._last_prompt_tokens is not None or self._last_completion_tokens is not None:
            logger.info(
                f"Gemini Token Usage ({self._model_name}): "
                f"Prompt={self._last_prompt_tokens if self._last_prompt_tokens is not None else 'N/A'}, "
                f"Completion={self._last_completion_tokens if self._last_completion_tokens is not None else 'N/A'}"
            )
        else:
            logger.warning(
                f"Gemini token usage data incomplete for {self._model_name}. Raw Usage Obj: {usage_metadata_obj}")

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:  # type: ignore
        """
        Formats the chat history into the structure required by the Gemini API.
        Handles multimodal inputs (text and images).
        """
        gemini_history: List[Dict[str, Any]] = []
        for msg in history:
            # Gemini uses 'user' and 'model' roles.
            role = 'user' if msg.role == USER_ROLE else ('model' if msg.role == MODEL_ROLE else None)
            if not role:
                logger.debug(f"GeminiAdapter: Skipping message with unhandled role: {msg.role}")
                continue

            api_parts: List[Any] = []  # Parts can be text strings or dicts for inline_data (images)

            # Add text part if present
            text_content = msg.text  # msg.text is a property that concatenates text parts
            if text_content and text_content.strip():
                api_parts.append({'text': text_content})

            # Add image parts if present and model supports vision
            # Assuming model name indicates vision capability (e.g., includes "vision" or "pro-vision")
            # Gemini Pro Vision models generally accept interleaved text and images.
            if msg.has_images and self._model_name and (
                    "vision" in self._model_name.lower() or "pro" in self._model_name.lower() or "flash" in self._model_name.lower()):  # Adjusted check for vision models
                for img_part_data_dict in msg.image_parts:  # msg.image_parts returns list of dicts
                    if isinstance(img_part_data_dict, dict) and \
                            img_part_data_dict.get("type") == "image" and \
                            img_part_data_dict.get("mime_type") and \
                            img_part_data_dict.get("data"):  # Base64 encoded data
                        try:
                            # Gemini SDK expects Blob for inline_data for images
                            if API_LIBRARY_AVAILABLE and hasattr(genai, 'types') and hasattr(genai.types,
                                                                                             'Blob'):  # type: ignore
                                img_blob = genai.types.Blob(  # type: ignore
                                    mime_type=img_part_data_dict["mime_type"],
                                    data=base64.b64decode(img_part_data_dict["data"])  # Decode base64 string to bytes
                                )
                                api_parts.append({'inline_data': img_blob})
                            else:
                                logger.warning("Gemini types.Blob not available. Cannot format image for Gemini API.")
                        except Exception as e_img_format:
                            logger.warning(f"Could not format image for Gemini API: {e_img_format}", exc_info=True)

            if api_parts:
                gemini_history.append({"role": role, "parts": api_parts})
            else:
                logger.debug(f"GeminiAdapter: Skipping message for role {role} due to no valid parts (text/image).")

        return gemini_history

    def get_available_models(self) -> List[str]:
        """
        Retrieves available Gemini models, with timeout protection and default fallbacks.
        """
        self._last_error = None
        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Gemini API library (google-generativeai) not installed."
            return []

        fetched_models_ids: List[str] = []
        try:
            # It's good practice to check if configure() has been called,
            # as API key might be needed for listing models.
            # However, genai.list_models() might work without prior genai.configure() if key is in env.
            if not self._is_configured:
                logger.warning(
                    "GeminiAdapter.get_available_models called when adapter might not be fully configured (API key status unknown). Attempting fetch anyway.")

            # Use threading for timeout protection for the genai.list_models() call
            import threading
            import queue

            result_queue: queue.Queue[List[str]] = queue.Queue()
            exception_queue: queue.Queue[Exception] = queue.Queue()

            def fetch_models_in_thread():
                try:
                    models_list = []
                    # Iterate through models and filter for those supporting 'generateContent'
                    # and typically containing "gemini" in their name.
                    for model_info in genai.list_models():  # type: ignore
                        if (hasattr(model_info, 'supported_generation_methods') and
                                'generateContent' in model_info.supported_generation_methods and
                                hasattr(model_info, 'name') and "gemini" in model_info.name.lower()):
                            models_list.append(model_info.name)
                    result_queue.put(models_list)
                except Exception as e_thread:
                    exception_queue.put(e_thread)

            fetch_thread = threading.Thread(target=fetch_models_in_thread)
            fetch_thread.daemon = True  # Allow main program to exit even if thread is running
            fetch_thread.start()
            fetch_thread.join(timeout=7.0)  # 7-second timeout for listing models

            if fetch_thread.is_alive():
                logger.warning("Gemini model fetch timed out after 7 seconds.")
                # Thread is still running, can't easily kill it, but we won't wait longer.
            elif not exception_queue.empty():
                raise exception_queue.get()  # Re-raise exception from thread
            elif not result_queue.empty():
                fetched_models_ids = result_queue.get()
                logger.info(f"Successfully fetched {len(fetched_models_ids)} Gemini models.")
            else:  # Thread finished but queues are empty (should not happen)
                logger.warning("Gemini model fetch thread finished but no result or exception.")

        except PermissionDenied as pde:  # Catch specific API errors
            self._last_error = f"Gemini API Permission Denied while fetching models: {pde}. Check API key."
            logger.error(self._last_error, exc_info=True)
            # Return empty or defaults, as fetch failed
        except Exception as e:  # Catch other errors during fetch
            self._last_error = f"Error fetching Gemini models: {type(e).__name__} - {e}"
            logger.warning(self._last_error, exc_info=True)
            # Return empty or defaults

        # Always provide some default candidates, especially if fetch failed or timed out
        default_candidates = [
            "models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-latest",  # Newer models
            "models/gemini-pro", "models/gemini-1.0-pro",  # Older but common
            "models/gemini-pro-vision"  # If vision is a common use case
        ]

        # Combine fetched models with defaults, ensuring uniqueness
        combined_list = list(set(fetched_models_ids + default_candidates))

        # Custom sort key for Gemini models
        def sort_key_gemini(model_name_str: str) -> Tuple[int, int, int, str]:
            name_lower = model_name_str.lower()
            # Primary sort: 1.5 models first, then 1.0, then others
            is_1_5 = "1.5" in name_lower
            is_1_0 = "1.0" in name_lower and not is_1_5  # Avoid double counting if "1.0" is in "1.5" string

            # Secondary sort: "pro" before "flash" before others
            is_pro = "pro" in name_lower
            is_flash = "flash" in name_lower and not is_pro

            # Tertiary sort: "latest" variants first
            is_latest = "latest" in name_lower

            # Sorting order: (1.5 > 1.0 > other), (pro > flash > other), (latest > not latest)
            return (
                0 if is_1_5 else 1 if is_1_0 else 2,  # Version priority
                0 if is_pro else 1 if is_flash else 2,  # Model type priority
                0 if is_latest else 1,  # "latest" tag priority
                name_lower  # Alphabetical as final tie-breaker
            )

        final_model_list = sorted(combined_list, key=sort_key_gemini)

        # If a model is currently configured, ensure it's in the list and at the top
        if self._model_name and self._is_configured and self._model_name not in final_model_list:
            final_model_list.insert(0, self._model_name)
            # Re-sort if we inserted, to maintain overall order but keep current at top if possible
            final_model_list.sort(key=lambda x: (x != self._model_name, *sort_key_gemini(x)))

        logger.debug(f"Returning Gemini models: {final_model_list}")
        return final_model_list

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        # Return None or (0,0) if you prefer to always return a tuple
        return None