# app/llm/ollama_adapter.py
import asyncio
import base64
import logging
import sys
import time

import requests
import re
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

try:
    import ollama

    try:
        from ollama._types import Model as _OllamaModelType, ChatResponse as _OllamaChatResponseType  # type: ignore
    except ImportError:
        _OllamaModelType = object
        _OllamaChatResponseType = object
        logging.getLogger(__name__).debug("OllamaAdapter: Could not import specific types from ollama._types.")
    OLLAMA_LIBRARY_AVAILABLE = True
except ImportError:
    ollama = None  # type: ignore
    _OllamaModelType = object
    _OllamaChatResponseType = object
    OLLAMA_LIBRARY_AVAILABLE = False
    logging.getLogger(__name__).warning("OllamaAdapter: 'ollama' library not found. Install: pip install ollama")

try:
    from .backend_interface import BackendInterface
    from models.chat_message import ChatMessage, MODEL_ROLE, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE
    # Import the new stream handler
    from .ollama_stream_handler import OllamaStreamHandler, OllamaStreamTimeoutError
except ImportError as e_imp:
    logging.getLogger(__name__).critical(f"OllamaAdapter: Critical import error: {e_imp}", exc_info=True)
    BackendInterface = type("BackendInterface", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MODEL_ROLE, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE = "model", "user", "system", "error"
    OllamaStreamHandler = type("OllamaStreamHandler", (object,), {})  # type: ignore
    OllamaStreamTimeoutError = type("OllamaStreamTimeoutError", (Exception,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class OllamaAdapter(BackendInterface):
    DEFAULT_OLLAMA_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "llama3:latest"

    def __init__(self):
        super().__init__()
        self._sync_client: Optional[ollama.Client] = None  # type: ignore
        self._model_name: str = self.DEFAULT_MODEL
        self._system_prompt: Optional[str] = None
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._ollama_host: str = self.DEFAULT_OLLAMA_HOST
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None
        self._startup_mode: bool = True

        # Instantiate the stream handler
        self._stream_handler: OllamaStreamHandler = OllamaStreamHandler(
            # Adjust timeouts as needed, these are examples
            chunk_timeout=45.0,  # Increased from 30s in previous handler version
            total_request_timeout=300.0,  # 5 minutes total
            max_retries=2,  # Retry twice on failure (3 attempts total)
            retry_delay_seconds=2.0
        )
        logger.info("OllamaAdapter initialized with OllamaStreamHandler.")

    def configure(self,
                  api_key: Optional[str],
                  model_name: Optional[str],
                  system_prompt: Optional[str] = None) -> bool:

        effective_model_name = model_name if model_name and model_name.strip() else self.DEFAULT_MODEL
        logger.info(f"OllamaAdapter: Configuring. Host: {self._ollama_host}, Model: {effective_model_name}")

        self._sync_client = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not OLLAMA_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library ('ollama') not installed."
            logger.error(self._last_error)
            return False

        self._model_name = effective_model_name
        self._system_prompt = system_prompt.strip() if isinstance(system_prompt,
                                                                  str) and system_prompt.strip() else None

        try:
            logger.info(f"Creating Ollama client for host {self._ollama_host}...")
            self._sync_client = ollama.Client(host=self._ollama_host)  # type: ignore
            self._is_configured = True
            logger.info(f"OllamaAdapter instance configured for model '{self._model_name}' at {self._ollama_host}.")

            if self._startup_mode:
                logger.info("OllamaAdapter: Startup mode - quick connection ping.")
                try:
                    response = requests.get(f"{self._ollama_host}/api/version", timeout=2.0)
                    response.raise_for_status()
                    logger.info(f"Quick ping to Ollama server at {self._ollama_host} successful.")
                except requests.exceptions.RequestException as e_ping:
                    self._last_error = f"Quick connection test to Ollama ({self._ollama_host}) failed: {e_ping}. Will retry on first request."
                    logger.warning(self._last_error)
                self._startup_mode = False
            else:
                logger.info("OllamaAdapter: Non-startup mode - attempting model list for connection test.")
                try:
                    models_list = self._sync_client.list(timeout=5)  # type: ignore
                    logger.info(
                        f"Connection test to Ollama at {self._ollama_host} successful. Found {len(models_list.get('models', []))} models.")
                except Exception as e_conn_test:
                    self._last_error = f"Connection test to Ollama ({self._ollama_host}) during config failed: {e_conn_test}. Will retry."
                    logger.warning(self._last_error)
            return True
        except Exception as e_client_create:
            self._last_error = f"Failed to create Ollama client for host '{self._ollama_host}': {type(e_client_create).__name__} - {e_client_create}"
            logger.error(self._last_error, exc_info=True)
            self._sync_client = None
            self._is_configured = False
            return False

    def is_configured(self) -> bool:
        return self._is_configured and self._sync_client is not None

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self,
                                  history: List[ChatMessage],  # type: ignore
                                  options: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        logger.info(
            f"OllamaAdapter: get_response_stream called. Model: {self._model_name}, History: {len(history)} items")
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._sync_client:
            self._last_error = "OllamaAdapter not configured or client missing."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)

        messages_for_api = self._format_history_for_api(history)
        if not messages_for_api and not self._system_prompt:
            self._last_error = "Cannot send request: No valid messages in history and no system prompt."
            logger.error(self._last_error)
            raise ValueError(self._last_error)

        logger.debug(f"Sending {len(messages_for_api)} formatted messages to model '{self._model_name}'")

        ollama_api_options: Dict[str, Any] = {}
        if options:
            if "temperature" in options and isinstance(options["temperature"], (float, int)):
                ollama_api_options["temperature"] = float(options["temperature"])
            if "max_tokens" in options and isinstance(options["max_tokens"], int):
                ollama_api_options["num_predict"] = options["max_tokens"]
            # Add other Ollama specific options like 'top_p', 'top_k', 'stop' etc.
            if "stop" in options and isinstance(options["stop"], list):
                ollama_api_options["stop"] = options["stop"]

        request_id = f"ollama_{self._model_name.replace(':', '_')}_{int(time.time() * 1000)}"
        final_response_data: Optional[Dict[str, Any]] = None

        try:
            async for chunk_dict in self._stream_handler.stream_with_timeout_and_retries(
                    ollama_client_instance=self._sync_client,  # type: ignore
                    model_name=self._model_name,
                    messages_for_api=messages_for_api,
                    ollama_api_options=ollama_api_options,
                    request_id=request_id
            ):
                # OllamaStreamHandler now yields the raw dictionary chunk
                if chunk_dict.get('error'):
                    self._last_error = f"Ollama API error in stream: {chunk_dict['error']}"
                    logger.error(self._last_error)
                    yield f"[SYSTEM ERROR: {self._last_error}]"
                    return

                message_part = chunk_dict.get('message', {})
                if isinstance(message_part, dict) and message_part.get('content'):
                    yield message_part['content']

                if chunk_dict.get('done', False):
                    final_response_data = chunk_dict  # Store the final chunk for token processing
                    break

            # Process token usage from the final chunk after the loop
            if final_response_data:
                self._last_prompt_tokens = final_response_data.get('prompt_eval_count')
                self._last_completion_tokens = final_response_data.get('eval_count')
                if self._last_prompt_tokens is not None or self._last_completion_tokens is not None:
                    logger.info(
                        f"Ollama Token Usage ({self._model_name}): Prompt Eval Tokens={self._last_prompt_tokens or 'N/A'}, Completion Eval Tokens={self._last_completion_tokens or 'N/A'}")
                else:
                    logger.warning(
                        f"Ollama token usage data not fully available in final 'done' chunk for {self._model_name}.")
            else:
                logger.warning(
                    f"Ollama stream for {self._model_name} ended but no final 'done' chunk with stats was captured by adapter.")


        except OllamaStreamTimeoutError as e_timeout:
            self._last_error = f"Ollama streaming timed out: {e_timeout}"
            logger.error(self._last_error, exc_info=True)
            yield f"[SYSTEM ERROR: {self._last_error}]"
        except RuntimeError as e_runtime:  # Catch critical errors from handler or client
            self._last_error = f"Ollama runtime error during stream: {e_runtime}"
            logger.error(self._last_error, exc_info=True)
            yield f"[SYSTEM ERROR: {self._last_error}]"  # Yield error to stream, but also consider re-raising
            # raise # Re-raise if it's critical and should stop further processing
        except Exception as e_general:
            self._last_error = f"Unexpected error during Ollama stream ({self._model_name}): {type(e_general).__name__} - {e_general}"
            logger.error(self._last_error, exc_info=True)
            yield f"[SYSTEM ERROR: {self._last_error}]"

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:  # type: ignore
        ollama_messages: List[Dict[str, Any]] = []
        if self._system_prompt:
            ollama_messages.append({"role": "system", "content": self._system_prompt})

        for msg in history:
            role_for_api: Optional[str] = None
            if msg.role == USER_ROLE:
                role_for_api = "user"
            elif msg.role == MODEL_ROLE:
                role_for_api = "assistant"
            elif msg.role == SYSTEM_ROLE and not self._system_prompt:
                role_for_api = "system"
            elif msg.role == SYSTEM_ROLE and self._system_prompt:
                continue  # Already handled
            elif msg.role == ERROR_ROLE or (
                    hasattr(msg, 'metadata') and msg.metadata and msg.metadata.get("is_internal")):
                continue
            else:
                logger.warning(f"OllamaAdapter: Skipping message with unhandled role: {msg.role}"); continue

            message_payload: Dict[str, Any] = {"role": role_for_api}
            text_content = msg.text.strip() if msg.text else ""

            if text_content:
                message_payload["content"] = text_content
            elif role_for_api == "system" and not self._system_prompt:
                message_payload["content"] = ""

            if msg.has_images and msg.image_parts:
                if "llava" in self._model_name.lower() or "bakllava" in self._model_name.lower() or "moondream" in self._model_name.lower():
                    base64_image_list: List[str] = [
                        img_part_dict["data"] for img_part_dict in msg.image_parts
                        if
                        isinstance(img_part_dict, dict) and img_part_dict.get("type") == "image" and img_part_dict.get(
                            "data")
                    ]
                    if base64_image_list:
                        message_payload["images"] = base64_image_list
                        if not text_content: message_payload[
                            "content"] = ""  # Ensure content field exists if only images

            if "content" in message_payload or "images" in message_payload:
                ollama_messages.append(message_payload)
        return ollama_messages

    def get_available_models(self) -> List[str]:
        # This method remains largely the same as it was already quite robust
        self._last_error = None
        model_names: List[str] = []
        try:
            timeout = 3.0 if self._startup_mode else 7.0
            response = requests.get(f"{self._ollama_host}/api/tags", timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if 'models' in data and isinstance(data['models'], list):
                model_names = [item['name'] for item in data['models'] if isinstance(item, dict) and 'name' in item]
                logger.info(f"Fetched {len(model_names)} models from Ollama via /api/tags.")
        except requests.exceptions.RequestException as e_http:
            logger.warning(f"Direct API call to Ollama /api/tags failed: {e_http}. Trying library if configured.")
            if OLLAMA_LIBRARY_AVAILABLE and self._sync_client:
                try:
                    models_response = self._sync_client.list(timeout=7)  # type: ignore
                    if 'models' in models_response and isinstance(models_response['models'], list):
                        model_names = [item.get('name') or item.get('model') for item in models_response['models'] if
                                       item.get('name') or item.get('model')]
                        logger.info(f"Fetched {len(model_names)} models using Ollama library fallback.")
                except Exception as e_lib:
                    self._last_error = f"Error fetching models via Ollama library: {e_lib}"
                    logger.warning(self._last_error)
            else:
                self._last_error = f"Direct API failed and Ollama library/client not available for model listing."
        except Exception as e_json:
            self._last_error = f"Error processing Ollama model list response: {e_json}"
            logger.warning(self._last_error)

        default_ollama_models = [
            "llama3:latest", "llama3:8b", "codellama:latest", "mistral:latest",
            "qwen2:latest", "phi3:latest", "gemma:latest", "llava:latest", self.DEFAULT_MODEL
        ]
        combined_list = sorted(list(set(model_names + default_ollama_models)), key=self._sort_key_ollama)
        if self._model_name and self._is_configured:
            if self._model_name in combined_list: combined_list.remove(self._model_name)
            combined_list.insert(0, self._model_name)

        if not combined_list and not self._last_error: self._last_error = "No Ollama models found or loaded from defaults."
        return combined_list

    def _sort_key_ollama(self, model_name_str: str) -> Tuple[int, float, int, str]:
        name_lower = model_name_str.lower()
        priority_family = 10
        if "llama3" in name_lower:
            priority_family = 0
        elif "codellama" in name_lower:
            priority_family = 1
        elif "qwen" in name_lower:
            priority_family = 2
        elif "mistral" in name_lower or "mixtral" in name_lower:
            priority_family = 3
        elif "phi3" in name_lower:
            priority_family = 4
        size_param = 0.0
        size_match = re.search(r'(\d+(?:\.\d+)?)(b|B)', name_lower)
        if size_match:
            try:
                size_param = -float(size_match.group(1))
            except ValueError:
                pass
        is_latest = 0 if "latest" in name_lower else 1
        return (priority_family, size_param, is_latest, name_lower)

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None