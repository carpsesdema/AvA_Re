# app/services/terminal_service.py
import asyncio
import logging
import os
import sys  # For sys.stdout.encoding
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict  # Keep Any if needed for future extensions

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    # Assuming constants might be used later (e.g., for whitelisted commands or paths)
    from utils import constants
except ImportError as e:
    logging.getLogger(__name__).critical(f"Critical import error in TerminalService: {e}", exc_info=True)
    # Define fallbacks if necessary for the script to be parsable,
    # though it will likely fail at runtime if these core components are missing.
    EventBus = type("EventBus", (object,), {"get_instance": lambda: type("DummyBus", (object,), {
        "terminalCommandRequested": type("Signal", (object,), {"connect": lambda x: None})(),
        "terminalCommandStarted": type("Signal", (object,), {"emit": lambda *args: None})(),
        "terminalCommandOutput": type("Signal", (object,), {"emit": lambda *args: None})(),
        "terminalCommandCompleted": type("Signal", (object,), {"emit": lambda *args: None})(),
        "terminalCommandError": type("Signal", (object,), {"emit": lambda *args: None})(),
    })()})
    constants = type("constants", (object,), {}) # Fallback constants object
    raise # Re-raise after attempting to define fallbacks for parsing

logger = logging.getLogger(__name__)


class TerminalService(QObject):
    """
    Service for executing terminal commands safely and streaming output.
    Integrates with the EventBus to emit command results.
    """

    # Safe commands that can be executed without user confirmation
    # This list should be carefully curated and maintained.
    SAFE_COMMANDS = {
        # Python tools
        'python', 'python3', 'pip', 'pip3', 'pytest', 'black', 'mypy', 'flake8', 'isort',
        # Node/JS tools
        'node', 'npm', 'npx', 'yarn',
        # General development tools
        'git', 'ls', 'dir', 'cat', 'type', 'echo', 'pwd', 'cd', 'mkdir', 'touch',
        # Linting and formatting
        'pylint', 'autopep8', 'bandit', 'safety', 'ruff',
        # Testing
        'coverage', 'tox', 'nox',
        # Build tools
        'make', 'cmake', 'ninja', 'gradle', 'mvn',
        # Other common dev utilities
        'grep', 'find', 'awk', 'sed', 'curl', 'wget', 'unzip', 'tar', 'gzip', 'bzip2',
        'py_compile', # Explicitly add py_compile as safe
    }

    # Commands that should never be executed automatically due to potential harm
    FORBIDDEN_COMMANDS = {
        'rm', 'del', 'rmdir', 'rd', 'format', 'fdisk', 'sudo', 'su', 'chmod', 'chown',
        'kill', 'killall', 'pkill', 'taskkill', 'shutdown', 'reboot', 'halt', 'poweroff',
        'dd', 'mkfs', 'fsck', 'mount', 'umount', 'crontab', 'systemctl', 'service',
        # Add potentially dangerous network commands if not intended for general use
        'iptables', 'ufw',
        # Potentially dangerous environment modification commands
        'export', 'set', 'unset', # Be cautious with these, though often used benignly
    }

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._event_bus = EventBus.get_instance()
        self._active_processes: Dict[str, asyncio.subprocess.Process] = {}
        self._connect_signals()
        logger.info("TerminalService initialized")

    def _connect_signals(self):
        """Connect to EventBus signals"""
        self._event_bus.terminalCommandRequested.connect(self._handle_command_request)

    @Slot(str, str, str) # command, working_directory, command_id
    def _handle_command_request(self, command: str, working_directory: str, command_id: str):
        """Handle terminal command execution request from the EventBus."""
        logger.info(f"Terminal command requested: '{command}' in '{working_directory}' (ID: {command_id})")
        # Schedule the async execution of the command
        asyncio.create_task(self._execute_command_async(command, working_directory, command_id))

    async def _execute_command_async(self, command: str, working_directory: str, command_id: str):
        """Execute command asynchronously and stream output."""
        start_time = time.time()
        process: Optional[asyncio.subprocess.Process] = None

        try:
            if not self._is_command_safe(command):
                error_msg = f"Command execution denied for safety reasons: '{command.split(' ')[0]}'"
                logger.warning(error_msg)
                self._event_bus.terminalCommandError.emit(command_id, error_msg)
                return

            if not os.path.isdir(working_directory):
                error_msg = f"Working directory does not exist or is not a directory: {working_directory}"
                logger.error(error_msg)
                self._event_bus.terminalCommandError.emit(command_id, error_msg)
                return

            self._event_bus.terminalCommandStarted.emit(command_id, command)

            # Using asyncio.create_subprocess_shell for executing the command.
            # Ensure proper quoting/escaping if 'command' contains user input directly,
            # though safety checks should mitigate most direct risks.
            # For Windows, shell=True might be necessary for some commands.
            # For POSIX, shell=True is generally discouraged if command is constructed from parts.
            # If `command` is a single string, shell=True is often how users expect it to work.
            # The safety check _is_command_safe focuses on the base command.
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
                # text=False by default, which is correct for reading bytes
            )
            self._active_processes[command_id] = process

            async def stream_output(stream: Optional[asyncio.StreamReader], output_type: str):
                if stream is None:
                    logger.warning(f"Stream for {output_type} is None for command {command_id}. Cannot read.")
                    return
                try:
                    while True:
                        line_bytes = await stream.readline()
                        if not line_bytes:
                            break # End of stream
                        try:
                            # Try decoding with common encodings, prioritizing UTF-8
                            line_str = line_bytes.decode('utf-8', errors='replace')
                        except UnicodeDecodeError:
                            try: # Fallback to system's default console encoding
                                line_str = line_bytes.decode(sys.stdout.encoding or 'latin-1', errors='replace')
                            except Exception: # Ultimate fallback
                                line_str = line_bytes.decode('latin-1', errors='replace')
                        self._event_bus.terminalCommandOutput.emit(command_id, output_type, line_str.rstrip('\n'))
                except asyncio.CancelledError:
                    logger.info(f"Output streaming for {output_type} (command {command_id}) cancelled.")
                except Exception as e_stream:
                    logger.error(f"Error streaming {output_type} for command {command_id}: {e_stream}", exc_info=True)
                    self._event_bus.terminalCommandOutput.emit(command_id, "stderr", f"[Streaming Error: {e_stream}]")

            # Create tasks for streaming stdout and stderr concurrently
            stdout_task = asyncio.create_task(stream_output(process.stdout, "stdout"))
            stderr_task = asyncio.create_task(stream_output(process.stderr, "stderr"))

            # Wait for both streams to complete
            await asyncio.gather(stdout_task, stderr_task)

            # Wait for the process to terminate and get the exit code
            exit_code = await process.wait()
            execution_time = time.time() - start_time

            self._event_bus.terminalCommandCompleted.emit(command_id, exit_code, execution_time)
            logger.info(f"Command '{command}' (ID: {command_id}) completed. Exit code: {exit_code}, Time: {execution_time:.2f}s")

        except FileNotFoundError:
            error_msg = f"Command not found: '{command.split(' ')[0]}'. Ensure it's in your system's PATH."
            logger.error(error_msg, exc_info=False) # No need for full exc_info for FileNotFoundError
            self._event_bus.terminalCommandError.emit(command_id, error_msg)
        except PermissionError:
            error_msg = f"Permission denied to execute command: '{command}'"
            logger.error(error_msg, exc_info=True)
            self._event_bus.terminalCommandError.emit(command_id, error_msg)
        except Exception as e:
            error_msg = f"Error executing command '{command}' (ID: {command_id}): {type(e).__name__} - {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._event_bus.terminalCommandError.emit(command_id, error_msg)
        finally:
            if command_id in self._active_processes:
                del self._active_processes[command_id]
            # Ensure process is cleaned up if it exists and an error occurred before wait()
            if process and process.returncode is None: # If process still running
                try:
                    logger.warning(f"Process for command {command_id} (PID: {process.pid}) was still running after error/completion handling. Attempting to kill.")
                    process.kill()
                    await process.wait() # Wait for kill to complete
                except ProcessLookupError:
                    pass # Process already died
                except Exception as e_kill:
                    logger.warning(f"Error trying to kill process {command_id} during cleanup: {e_kill}")


    def _is_command_safe(self, command: str) -> bool:
        """Determines if a command is safe to execute based on predefined lists."""
        if not command or not command.strip():
            logger.warning("Attempted to check safety of an empty command.")
            return False

        # Split the command string into parts. Handles simple cases.
        # For complex shell commands with pipes, redirections, etc., this is a simplification.
        # A more robust solution might involve a proper shell parser or a more restrictive execution model.
        parts = command.strip().split()
        base_command_executable = parts[0] # The first part is usually the command/executable

        # Normalize the base command (e.g., resolve path, get executable name)
        # This helps if `base_command_executable` is a full path.
        base_command_name = Path(base_command_executable).name.lower()

        # Check against forbidden commands first (highest priority)
        if base_command_name in self.FORBIDDEN_COMMANDS:
            logger.warning(f"Command '{base_command_name}' is in FORBIDDEN_COMMANDS list.")
            return False

        # Check if it's an explicitly safe command
        if base_command_name in self.SAFE_COMMANDS:
            logger.debug(f"Command '{base_command_name}' is in SAFE_COMMANDS list.")
            return True

        # Special handling for Python module execution: python -m <module>
        if base_command_name in ('python', 'python3') and len(parts) > 2 and parts[1] == '-m':
            module_name_to_run = parts[2].lower()
            # Allow if the module itself is considered safe (e.g., 'pip', 'venv', 'pytest')
            if module_name_to_run in self.SAFE_COMMANDS:
                logger.debug(f"Allowing safe Python module execution: 'python -m {module_name_to_run}'")
                return True
            logger.warning(f"Python module '{module_name_to_run}' not in explicit SAFE_COMMANDS for '-m' execution.")
            return False # Default to not safe for unknown modules via -m

        # Special handling for pip install (generally safe, but source of packages is a concern)
        if base_command_name in ('pip', 'pip3') and len(parts) > 1 and parts[1].lower() == 'install':
            logger.debug(f"Allowing 'pip install' command: '{command}'")
            return True # Assuming user knows what they are installing. For stricter control, analyze package names.

        logger.warning(f"Command '{base_command_name}' (from '{command}') not in SAFE_COMMANDS and doesn't match allowed patterns. Defaulting to not safe.")
        return False # Default to not safe if not explicitly allowed

    def execute_command_request(self, command: str, working_directory: Optional[str] = None) -> str:
        """
        Public method to request command execution.
        Generates a command ID and emits it via EventBus.
        Returns the command_id for tracking.
        """
        command_id = f"cmd_{uuid.uuid4().hex[:8]}"
        work_dir = working_directory or os.getcwd() # Default to current working directory if not specified
        self._event_bus.terminalCommandRequested.emit(command, work_dir, command_id)
        logger.info(f"Public request to execute command: '{command}' in '{work_dir}'. Assigned ID: {command_id}")
        return command_id

    def cancel_command(self, command_id: str) -> bool:
        """Cancel a running command by its ID."""
        if command_id in self._active_processes:
            process = self._active_processes[command_id]
            try:
                if process.returncode is None:  # Check if process is still running
                    logger.info(f"Attempting to terminate process for command ID: {command_id} (PID: {process.pid})")
                    process.terminate()  # Send SIGTERM (graceful shutdown)
                    # For a more forceful stop, you might use process.kill() after a timeout.
                    # asyncio.create_task(self._ensure_process_killed(process, command_id))
                else:
                    logger.info(f"Process for command {command_id} already completed with code {process.returncode}.")
                # Removal from _active_processes happens in _execute_command_async's finally block or here if already done.
                if command_id in self._active_processes: # Check again in case it finished fast
                    del self._active_processes[command_id]
                return True
            except ProcessLookupError:
                logger.warning(f"Process for command {command_id} (PID: {process.pid if process else 'N/A'}) already exited before explicit cancellation.")
                if command_id in self._active_processes: del self._active_processes[command_id]
                return True # Considered successful cancellation if already exited
            except Exception as e:
                logger.error(f"Error cancelling command {command_id}: {e}", exc_info=True)
                return False
        logger.info(f"Command ID {command_id} not found in active processes for cancellation.")
        return False

    # Optional: Helper to ensure process is killed if terminate isn't enough
    # async def _ensure_process_killed(self, process: asyncio.subprocess.Process, command_id: str):
    #     try:
    #         await asyncio.wait_for(process.wait(), timeout=2.0) # Wait 2s for terminate
    #     except asyncio.TimeoutError:
    #         if process.returncode is None:
    #             logger.warning(f"Process {command_id} (PID: {process.pid}) did not terminate gracefully. Killing.")
    #             process.kill()
    #             await process.wait() # Wait for kill
    #     finally:
    #         if command_id in self._active_processes:
    #             del self._active_processes[command_id]


    def get_active_commands(self) -> List[str]:
        """Returns a list of IDs for currently active commands."""
        return list(self._active_processes.keys())

    def is_command_running(self, command_id: str) -> bool:
        """Checks if a command with the given ID is currently running."""
        process = self._active_processes.get(command_id)
        return process is not None and process.returncode is None