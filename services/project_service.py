# app/services/project_service.py
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional, Dict

from PySide6.QtCore import QObject, Signal

try:
    # Assuming models are in models now
    from models.project_models import Project, ChatSession
    from models.chat_message import ChatMessage
    # Enums are also in models
    from models.message_enums import MessageLoadingState
    from utils import constants
except ImportError as e:
    logging.getLogger(__name__).critical(f"Critical import error in ProjectService: {e}", exc_info=True)
    # Fallback for type hinting if imports fail
    Project = type("Project", (object,), {}) # type: ignore
    ChatSession = type("ChatSession", (object,), {}) # type: ignore
    ChatMessage = type("ChatMessage", (object,), {}) # type: ignore
    MessageLoadingState = type("MessageLoadingState", (object,), {}) # type: ignore
    constants = type("constants", (object,), {"USER_DATA_DIR": os.path.expanduser("~/.ava_pys6_data_p1_fallback")}) # type: ignore
    # Re-raise if critical for app function
    raise

logger = logging.getLogger(__name__)


class ProjectManager(QObject):
    projectCreated = Signal(str)  # project_id
    projectSwitched = Signal(str) # project_id
    sessionCreated = Signal(str, str) # project_id, session_id
    sessionSwitched = Signal(str, str) # project_id, session_id
    projectsLoaded = Signal(list)  # List[Project]
    projectDeleted = Signal(str) # project_id
    sessionRenamed = Signal(str, str, str) # project_id, session_id, new_name

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.projects_dir = os.path.join(constants.USER_DATA_DIR, "projects")
        self.projects_index_file = os.path.join(self.projects_dir, "projects_index.json")
        self._current_project: Optional[Project] = None
        self._current_session: Optional[ChatSession] = None
        self._projects_cache: Dict[str, Project] = {}
        self._sessions_cache: Dict[str, ChatSession] = {} # session_id -> ChatSession
        self._ensure_directories()
        self._load_projects_index()
        logger.info("ProjectManager initialized.")

    def _ensure_directories(self):
        try:
            os.makedirs(self.projects_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create projects directory: {e}")
            raise

    def _load_projects_index(self):
        if not os.path.exists(self.projects_index_file):
            logger.info(f"Projects index file not found at {self.projects_index_file}. Will be created on first save.")
            return
        try:
            with open(self.projects_index_file, 'r', encoding='utf-8') as f:
                projects_data = json.load(f)
            # Use Project.from_dict for proper deserialization
            self._projects_cache = {pid: Project.from_dict(data) for pid, data in projects_data.items()}
            logger.info(f"Loaded {len(self._projects_cache)} projects from index.")
            self.projectsLoaded.emit(list(self._projects_cache.values()))
        except Exception as e:
            logger.error(f"Failed to load projects index: {e}", exc_info=True)
            self._projects_cache = {} # Start fresh if index is corrupt

    def _save_projects_index(self):
        try:
            with open(self.projects_index_file, 'w', encoding='utf-8') as f:
                # Use project.to_dict() for proper serialization
                json.dump({pid: p.to_dict() for pid, p in self._projects_cache.items()}, f, indent=2, ensure_ascii=False)
            logger.debug("Projects index saved.")
        except Exception as e:
            logger.error(f"Failed to save projects index: {e}")

    def create_project(self, name: str, description: str = "") -> Project:
        project_id = str(uuid.uuid4())
        project = Project(id=project_id, name=name, description=description)
        self._projects_cache[project_id] = project

        project_dir_path = os.path.join(self.projects_dir, project_id)
        try:
            os.makedirs(project_dir_path, exist_ok=True)
            os.makedirs(os.path.join(project_dir_path, "sessions"), exist_ok=True)
            os.makedirs(os.path.join(project_dir_path, "generated_files"), exist_ok=True) # For RAG / code output
        except OSError as e:
            logger.error(f"Error creating directories for project {project_id}: {e}")
            if project_id in self._projects_cache: del self._projects_cache[project_id] # Rollback cache
            raise

        try:
            default_session = self.create_session(project_id, "Main Chat")
            project.current_session_id = default_session.id
        except Exception as e_sess:
            logger.error(f"Failed to create default session for project {project_id}: {e_sess}")
            project.current_session_id = None

        self._save_projects_index()
        logger.info(f"Created project '{name}' with ID: {project_id}")
        self.projectCreated.emit(project_id)
        return project

    def create_session(self, project_id: str, name: str) -> ChatSession:
        if project_id not in self._projects_cache:
            logger.error(f"Attempted to create session for non-existent project ID: {project_id}")
            raise ValueError(f"Project {project_id} not found")

        session_id = str(uuid.uuid4())
        session = ChatSession(id=session_id, project_id=project_id, name=name)
        self._save_session(session) # Save new session to its file
        self._sessions_cache[session_id] = session # Add to runtime cache
        logger.info(f"Created session '{name}' (ID: {session_id}) in project {project_id}")
        self.sessionCreated.emit(project_id, session_id)
        return session

    def switch_to_project(self, project_id: str) -> bool:
        if project_id not in self._projects_cache:
            logger.error(f"Cannot switch to unknown project: {project_id}")
            return False

        new_project = self._projects_cache[project_id]
        if self._current_project and self._current_project.id == project_id:
            logger.debug(f"Project {project_id} is already current.")
            # Ensure its current session is also active if one is set
            if new_project.current_session_id and \
               (not self._current_session or self._current_session.id != new_project.current_session_id):
                self.switch_to_session(new_project.current_session_id)
            elif not new_project.current_session_id: # If project has no current session, load first available
                sessions = self.get_project_sessions(project_id)
                if sessions: self.switch_to_session(sessions[0].id)
            self.projectSwitched.emit(project_id) # Emit even if already current, to refresh UI if needed
            return True

        self._current_project = new_project
        self._current_session = None # Reset current session when project changes

        if self._current_project.current_session_id:
            self.switch_to_session(self._current_project.current_session_id)
        else: # If project has no current_session_id, load its first session or let UI handle it
            sessions = self.get_project_sessions(project_id)
            if sessions:
                self.switch_to_session(sessions[0].id)
            # Else, the project has no sessions, UI should reflect this (e.g., prompt to create one)

        logger.info(f"Switched to project: {self._current_project.name}")
        self.projectSwitched.emit(project_id)
        return True

    def switch_to_session(self, session_id: str) -> bool:
        if not self._current_project:
            logger.error("No current project to switch session in.")
            return False

        session = self._sessions_cache.get(session_id)
        if not session: # If not in runtime cache, try loading from disk
            session = self._load_session(self._current_project.id, session_id)
            if not session:
                logger.error(f"Session {session_id} not found or failed to load for project {self._current_project.id}")
                return False
            self._sessions_cache[session_id] = session # Add to runtime cache

        if session.project_id != self._current_project.id:
            logger.error(f"Session {session_id} (project: {session.project_id}) does not belong to current project {self._current_project.id}")
            return False

        self._current_session = session
        if self._current_project.current_session_id != session_id:
            self._current_project.current_session_id = session_id
            self._save_projects_index() # Persist the change in current_session_id for the project

        logger.info(f"Switched to session: {self._current_session.name} in project {self._current_project.name}")
        self.sessionSwitched.emit(self._current_project.id, session_id)
        return True

    def _get_session_file_path(self, project_id: str, session_id: str) -> str:
        return os.path.join(self.projects_dir, project_id, "sessions", f"{session_id}.json")

    def _load_session(self, project_id: str, session_id: str) -> Optional[ChatSession]:
        session_file = self._get_session_file_path(project_id, session_id)
        if not os.path.exists(session_file):
            logger.warning(f"Session file not found: {session_file}")
            return None
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            # Use ChatSession.from_dict for proper deserialization, including ChatMessage objects
            return ChatSession.from_dict(session_data)
        except json.JSONDecodeError as e_json:
            logger.error(f"Failed to load session {session_id} from {session_file}: JSON Decode Error: {e_json}")
            self._handle_corrupted_session_file(session_file)
            return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from {session_file}: {e}", exc_info=True)
            return None

    def _handle_corrupted_session_file(self, session_file_path: str):
        try:
            corrupted_dir = os.path.join(os.path.dirname(session_file_path), "corrupted_sessions")
            os.makedirs(corrupted_dir, exist_ok=True)
            backup_file_name = f"{os.path.basename(session_file_path)}.{datetime.now().strftime('%Y%m%d%H%M%S')}.corrupted"
            backup_file_path = os.path.join(corrupted_dir, backup_file_name)
            shutil.move(session_file_path, backup_file_path) # Move instead of copy
            logger.warning(f"Moved corrupted session file '{session_file_path}' to '{backup_file_path}'")
        except Exception as e_backup:
            logger.error(f"Failed to backup/move corrupted session file '{session_file_path}': {e_backup}")


    def _save_session(self, session: ChatSession):
        if not session.project_id:
            logger.error("Cannot save session without project_id.")
            return
        project_sessions_dir = os.path.join(self.projects_dir, session.project_id, "sessions")
        os.makedirs(project_sessions_dir, exist_ok=True) # Ensure directory exists
        session_file = self._get_session_file_path(session.project_id, session.id)
        try:
            with open(session_file, 'w', encoding='utf-t8') as f:
                # Use session.to_dict() for proper serialization
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Session {session.id} saved to {session_file}")
        except Exception as e:
            logger.error(f"Failed to save session {session.id} to {session_file}: {e}")

    def update_current_session_history(self, messages: List[ChatMessage]): # type: ignore
        if not self._current_session:
            logger.warning("No current session to update history for.")
            return
        self._current_session.message_history = messages # type: ignore
        self._save_session(self._current_session)
        logger.debug(f"Updated history for session {self._current_session.id} and saved.")

    def rename_session(self, project_id: str, session_id: str, new_name: str) -> bool:
        if not project_id or not session_id or not new_name.strip():
            logger.warning("Invalid parameters for renaming session.")
            return False

        session = self._sessions_cache.get(session_id)
        if not session:
            session = self._load_session(project_id, session_id)

        if not session or session.project_id != project_id:
            logger.error(f"Session {session_id} not found or does not belong to project {project_id}.")
            return False

        old_name = session.name
        session.name = new_name.strip()
        self._save_session(session)
        logger.info(f"Renamed session {session_id} from '{old_name}' to '{session.name}' in project {project_id}.")
        self.sessionRenamed.emit(project_id, session_id, session.name)
        return True


    def get_project_by_id(self, project_id: str) -> Optional[Project]:
        return self._projects_cache.get(project_id)

    def get_session_by_id(self, session_id: str) -> Optional[ChatSession]:
        # Prioritize runtime cache
        if session_id in self._sessions_cache:
            return self._sessions_cache[session_id]
        # If not in cache, try to load it (requires knowing its project_id)
        # This might be slow if we have to iterate all projects.
        # Consider if session_id should be globally unique or if project_id context is always available.
        # For now, assuming session_id is unique enough or a project context is implied.
        if self._current_project: # Check current project first
            session = self._load_session(self._current_project.id, session_id)
            if session: self._sessions_cache[session_id] = session; return session
        # Fallback: iterate all projects (could be slow if many projects/sessions)
        for pid_iter in self._projects_cache:
            session = self._load_session(pid_iter, session_id)
            if session: self._sessions_cache[session_id] = session; return session
        logger.warning(f"Session {session_id} not found in cache or on disk.")
        return None

    def get_current_project(self) -> Optional[Project]:
        return self._current_project

    def get_current_session(self) -> Optional[ChatSession]:
        return self._current_session

    def get_all_projects(self) -> List[Project]:
        return list(self._projects_cache.values())

    def get_project_sessions(self, project_id: str) -> List[ChatSession]: # type: ignore
        if project_id not in self._projects_cache:
            logger.warning(f"Attempted to get sessions for non-existent project: {project_id}")
            return []
        sessions_dir = os.path.join(self.projects_dir, project_id, "sessions")
        if not os.path.isdir(sessions_dir):
            logger.debug(f"Sessions directory not found for project {project_id}, returning empty list.")
            return []

        sessions: List[ChatSession] = []
        session_ids_seen: set[str] = set()

        for filename in os.listdir(sessions_dir):
            if filename.endswith('.json'):
                session_id_from_file = filename[:-5] # Remove .json
                if session_id_from_file in session_ids_seen: continue # Avoid duplicates if any
                session_ids_seen.add(session_id_from_file)

                # Try cache first
                session_obj = self._sessions_cache.get(session_id_from_file)
                if not session_obj: # If not in cache, load from disk
                    loaded_session = self._load_session(project_id, session_id_from_file)
                    if loaded_session:
                        self._sessions_cache[session_id_from_file] = loaded_session
                        session_obj = loaded_session
                if session_obj:
                    sessions.append(session_obj)
        # Sort sessions, e.g., by creation date
        sessions.sort(key=lambda s: s.created_at if s.created_at else "")
        return sessions

    def get_project_files_dir(self, project_id: Optional[str] = None) -> str:
        target_pid = project_id or (self._current_project.id if self._current_project else None)
        if not target_pid:
            logger.warning("get_project_files_dir called with no project context, returning generic dir.")
            # Fallback path if no project context
            return os.path.join(constants.USER_DATA_DIR, "default_project_files")
        return os.path.join(self.projects_dir, target_pid, "generated_files")


    def delete_project(self, project_id: str) -> bool:
        if project_id not in self._projects_cache:
            logger.warning(f"Attempted to delete non-existent project: {project_id}")
            return False
        try:
            # Delete all sessions associated with this project first
            sessions_to_delete = self.get_project_sessions(project_id)
            for session in sessions_to_delete:
                self.delete_session(project_id, session.id, emit_project_deleted_signal=False) # Avoid redundant signals

            # Delete the project directory itself
            project_path = os.path.join(self.projects_dir, project_id)
            if os.path.exists(project_path):
                shutil.rmtree(project_path) # Robustly remove directory and its contents
                logger.info(f"Removed project directory: {project_path}")

            # Remove from cache and save index
            del self._projects_cache[project_id]
            self._save_projects_index()

            # If it was the current project, clear current project/session
            if self._current_project and self._current_project.id == project_id:
                self._current_project = None
                self._current_session = None
                logger.info("Current project was deleted. Active project/session cleared.")

            self.projectDeleted.emit(project_id)
            logger.info(f"Project {project_id} deleted successfully.")
            return True
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}", exc_info=True)
            return False

    def delete_session(self, project_id: str, session_id: str, emit_project_deleted_signal: bool = True) -> bool:
        session_file = self._get_session_file_path(project_id, session_id)
        if os.path.exists(session_file):
            try:
                os.remove(session_file)
                if session_id in self._sessions_cache:
                    del self._sessions_cache[session_id]

                project = self.get_project_by_id(project_id)
                if project and project.current_session_id == session_id:
                    project.current_session_id = None # Clear current session for the project
                    self._save_projects_index() # Save change to project index
                    if self._current_session and self._current_session.id == session_id:
                        self._current_session = None # Also clear if it was the globally current session
                logger.info(f"Session {session_id} deleted from project {project_id}.")
                # Note: No direct signal for session deletion yet, could be added if UI needs it.
                return True
            except Exception as e:
                logger.error(f"Error deleting session file {session_file}: {e}", exc_info=True)
                return False
        logger.warning(f"Session file not found for deletion: {session_file}")
        return False