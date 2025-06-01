# app/llm/ollama_stream_handler.py
import asyncio
import logging
import time  # For request_id generation if needed, and internal timing
from typing import Optional, AsyncGenerator, Any, Dict, TYPE_CHECKING
from contextlib import asynccontextmanager  # For potential context management features

if TYPE_CHECKING:
    # This helps with type hinting for the Ollama client without a hard import cycle risk
    # if ollama_adapter were to import this for patching.
    # However, this handler is more likely to be USED BY ollama_adapter.
    import ollama  # type: ignore

logger = logging.getLogger(__name__)


class OllamaStreamTimeoutError(Exception):
    """Custom exception for Ollama streaming timeouts encountered by this handler."""
    pass


class OllamaStreamHandler:
    """
    A dedicated handler for managing streaming responses from an Ollama server.
    This class aims to provide more robust timeout handling and potentially
    retry logic for Ollama streams, which can sometimes be prone to read timeouts
    with the standard client if the model takes a long time between chunks.

    It's designed to be used by the OllamaAdapter.
    """

    def __init__(self,
                 chunk_timeout: float = 45.0,  # Timeout for receiving a single chunk
                 total_request_timeout: float = 300.0,  # Max total time for the entire stream request (5 minutes)
                 max_retries: int = 1,  # Number of retries for the whole stream operation if timeout occurs
                 retry_delay_seconds: float = 3.0):  # Delay between retries
        self.chunk_timeout = chunk_timeout
        self.total_request_timeout = total_request_timeout
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # For tracking active streams if multiple instances of this handler were used (less common)
        # Or for more detailed internal state if needed.
        self._active_streams_debug: Dict[str, Dict[str, Any]] = {}
        logger.info(f"OllamaStreamHandler initialized with: chunk_timeout={chunk_timeout}s, "
                    f"total_timeout={total_request_timeout}s, max_retries={max_retries}")

    async def stream_with_timeout_and_retries(self,
                                              ollama_client_instance: 'ollama.Client',  # Type hint with forward ref
                                              model_name: str,
                                              messages_for_api: list,  # List of dicts for Ollama API
                                              ollama_api_options: Optional[Dict[str, Any]] = None,
                                              request_id: Optional[str] = None
                                              ) -> AsyncGenerator[Dict[str, Any], None]:  # Yields raw chunk dicts
        """
        Streams from Ollama with comprehensive timeout handling and retries for the entire operation.

        Args:
            ollama_client_instance: The configured Ollama client instance.
            model_name: The name of the Ollama model to use.
            messages_for_api: The list of message dictionaries formatted for Ollama.
            ollama_api_options: Optional dictionary of parameters for the Ollama API.
            request_id: An optional ID for tracking this stream request.

        Yields:
            Dict[str, Any]: Raw chunk dictionaries received from the Ollama stream.
                           The caller (OllamaAdapter) will be responsible for extracting text content.

        Raises:
            OllamaStreamTimeoutError: If streaming times out beyond recovery after all retries.
            RuntimeError: For other critical, non-timeout related errors during streaming.
        """
        if not request_id:
            request_id = f"ollama_stream_{int(time.time() * 1000)}_{model_name[:10]}"

        # Debug tracking for this stream attempt
        self._active_streams_debug[request_id] = {
            'start_time': time.monotonic(), 'last_chunk_time': time.monotonic(),
            'chunk_count': 0, 'total_chars_approx': 0, 'model': model_name, 'retries_done': 0
        }

        try:
            current_retry_attempt = 0
            last_exception_encountered: Optional[Exception] = None

            while current_retry_attempt <= self.max_retries:
                self._active_streams_debug[request_id]['retries_done'] = current_retry_attempt
                try:
                    logger.info(
                        f"Ollama Stream [{request_id}]: Starting attempt {current_retry_attempt + 1}/{self.max_retries + 1} for model '{model_name}'.")

                    # Use an overall timeout for the entire _single_stream_attempt call
                    async for chunk_dict in asyncio.wait_for(
                            self._single_stream_attempt(
                                ollama_client_instance, model_name, messages_for_api,
                                ollama_api_options, request_id
                            ),
                            timeout=self.total_request_timeout
                    ):
                        yield chunk_dict  # Yield the raw dictionary chunk

                    # If the loop completes without timeout, the stream was successful for this attempt
                    logger.info(
                        f"Ollama Stream [{request_id}]: Attempt {current_retry_attempt + 1} completed successfully.")
                    return  # Successful completion of the stream

                except asyncio.TimeoutError:  # Timeout for the whole _single_stream_attempt
                    last_exception_encountered = OllamaStreamTimeoutError(
                        f"Total request timeout ({self.total_request_timeout}s) exceeded for attempt {current_retry_attempt + 1} on stream {request_id}.")
                    logger.warning(f"Ollama Stream [{request_id}]: {last_exception_encountered}")
                except OllamaStreamTimeoutError as e_chunk_timeout:  # Specific chunk timeout from _single_stream_attempt
                    last_exception_encountered = e_chunk_timeout
                    logger.warning(
                        f"Ollama Stream [{request_id}]: {e_chunk_timeout} (Attempt {current_retry_attempt + 1})")
                except Exception as e_inner:  # Other errors from _single_stream_attempt
                    last_exception_encountered = e_inner
                    logger.error(
                        f"Ollama Stream [{request_id}]: Non-timeout error in attempt {current_retry_attempt + 1}: {e_inner}",
                        exc_info=True)
                    # For critical errors not related to timeout, might not want to retry
                    # For simplicity now, all exceptions lead to retry if attempts remain.
                    # Could add specific exception types to break early.
                    # If it's a connection error, retrying might make sense.
                    if isinstance(e_inner, RuntimeError) and "connection" in str(e_inner).lower():
                        pass  # Proceed to retry for connection issues
                    else:  # For other RuntimeErrors or unexpected errors, maybe break early
                        raise  # Re-raise to stop retries

                current_retry_attempt += 1
                if current_retry_attempt <= self.max_retries:
                    logger.info(f"Ollama Stream [{request_id}]: Retrying in {self.retry_delay_seconds}s...")
                    await asyncio.sleep(self.retry_delay_seconds)
                else:
                    logger.error(f"Ollama Stream [{request_id}]: All {self.max_retries + 1} retry attempts exhausted.")

            # If loop finishes, all retries failed
            if last_exception_encountered:
                if isinstance(last_exception_encountered, OllamaStreamTimeoutError):
                    raise last_exception_encountered  # Re-raise the specific timeout error
                else:  # Wrap other errors
                    raise OllamaStreamTimeoutError(
                        f"Ollama streaming failed after {self.max_retries + 1} attempts. Last error: {last_exception_encountered}") from last_exception_encountered
            else:  # Should not be reached if retries exhausted
                raise OllamaStreamTimeoutError(
                    f"Ollama streaming failed after {self.max_retries + 1} attempts (unknown final error).")

        finally:
            self._active_streams_debug.pop(request_id, None)  # Clean up tracking

    async def _single_stream_attempt(self,
                                     ollama_client: 'ollama.Client',
                                     model: str,
                                     messages: list,
                                     options: Optional[Dict[str, Any]],
                                     request_id: str  # For logging/debugging
                                     ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handles a single attempt to stream from Ollama.
        This runs the synchronous Ollama client's streaming method in an executor
        and yields chunks with individual chunk timeouts.
        """
        loop = asyncio.get_running_loop()
        stream_iterator = None

        try:
            # This call itself can block or timeout if Ollama server is unresponsive
            # Use a timeout for the initial call to ollama.chat
            logger.debug(f"Ollama Stream [{request_id}]: Initiating ollama.chat call in executor.")
            stream_iterator = await asyncio.wait_for(
                loop.run_in_executor(
                    None,  # Default thread pool
                    lambda: ollama_client.chat(model=model, messages=messages, stream=True, options=options or {})
                ),
                timeout=15.0  # Timeout for establishing the stream connection (e.g., 15 seconds)
            )
            logger.debug(f"Ollama Stream [{request_id}]: ollama.chat stream iterator obtained.")

        except asyncio.TimeoutError:
            raise OllamaStreamTimeoutError(f"Timeout establishing Ollama stream connection for request {request_id}.")
        except Exception as e_create_stream:  # Catch errors from ollama.chat call itself
            logger.error(f"Ollama Stream [{request_id}]: Failed to create stream: {e_create_stream}", exc_info=True)
            # Re-raise as a runtime error to be handled by the retry loop or caller
            raise RuntimeError(f"Failed to create Ollama stream: {e_create_stream}") from e_create_stream

        last_chunk_received_time = time.monotonic()
        stream_debug_info = self._active_streams_debug.get(request_id)

        while True:
            try:
                # Get the next chunk from the synchronous iterator in the executor
                # Apply chunk_timeout to this operation.
                next_chunk_dict = await asyncio.wait_for(
                    loop.run_in_executor(None, next, stream_iterator, None),  # Pass None as default to next()
                    timeout=self.chunk_timeout
                )

                if next_chunk_dict is None:  # Iterator exhausted (stream ended normally by Ollama)
                    logger.info(f"Ollama Stream [{request_id}]: Stream iterator exhausted (ended normally).")
                    break

                last_chunk_received_time = time.monotonic()
                if stream_debug_info:
                    stream_debug_info['last_chunk_time'] = last_chunk_received_time
                    stream_debug_info['chunk_count'] += 1
                    if isinstance(next_chunk_dict.get('message', {}).get('content'), str):
                        stream_debug_info['total_chars_approx'] += len(next_chunk_dict['message']['content'])

                yield next_chunk_dict  # Yield the raw dictionary

                if next_chunk_dict.get('done', False):
                    logger.info(f"Ollama Stream [{request_id}]: 'done' flag received in chunk.")
                    break  # Stream officially finished

            except asyncio.TimeoutError:  # Timeout waiting for a specific chunk
                raise OllamaStreamTimeoutError(f"No chunk received in {self.chunk_timeout}s for stream {request_id}.")
            except StopIteration:  # Should be caught by next(..., None) if iterator ends
                logger.info(f"Ollama Stream [{request_id}]: StopIteration received, stream ended.")
                break
            except Exception as e_next_chunk:  # Other errors while getting next chunk
                logger.error(f"Ollama Stream [{request_id}]: Error getting next chunk: {e_next_chunk}", exc_info=True)
                raise RuntimeError(f"Error iterating Ollama stream: {e_next_chunk}") from e_next_chunk

            await asyncio.sleep(0.001)  # Tiny sleep to allow other tasks to run, good for tight loops

    # --- Methods for managing or inspecting streams (examples) ---

    @asynccontextmanager
    async def managed_stream_context(self, request_id: str):
        """Example context manager for stream lifecycle (if needed for more complex state)."""
        logger.debug(f"Ollama Stream Context Manager: Entering for {request_id}")
        try:
            yield
        finally:
            logger.debug(f"Ollama Stream Context Manager: Exiting for {request_id}")
            self._active_streams_debug.pop(request_id, None)  # Ensure cleanup if used

    def get_stream_diagnostics(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get diagnostic info for an active stream (for debugging)."""
        return self._active_streams_debug.get(request_id)

    def list_active_stream_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Get diagnostics for all currently tracked active streams."""
        return self._active_streams_debug.copy()