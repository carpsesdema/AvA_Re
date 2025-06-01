# app/llm/rag_system.py
import logging
import os
import re
from typing import List, Optional, Set, Tuple, Dict, Any

try:
    # Assuming UploadService and VectorDBService are in app.services
    from app.services.upload_service import UploadService
    from app.services.vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID  # type: ignore
    # Constants from utils
    from utils import constants
except ImportError as e_rag:
    logging.getLogger(__name__).critical(f"RagSystem: Critical import error: {e_rag}", exc_info=True)
    # Fallback types for type hinting
    UploadService = type("UploadService", (object,), {})  # type: ignore
    VectorDBService = type("VectorDBService", (object,), {})  # type: ignore
    GLOBAL_COLLECTION_ID = "global_knowledge_fallback_rag"  # Fallback
    constants = type("constants", (object,), {  # type: ignore
        "RAG_NUM_RESULTS": 5, "RAG_CHUNK_SIZE": 1000, "RAG_CHUNK_OVERLAP": 150,
        "ALLOWED_TEXT_EXTENSIONS": set(), "DEFAULT_IGNORED_DIRS": set(),
        "MAX_SCAN_DEPTH": 5, "RAG_MAX_FILE_SIZE_MB": 50
    })
    raise

logger = logging.getLogger(__name__)


class RagSystem:  # Renamed from RagHandler
    """
    Handles Retrieval Augmented Generation (RAG) processes.
    Determines when to perform RAG, extracts relevant entities from queries,
    queries the vector database for context, and formats it for the LLM.
    """

    # Keywords that strongly suggest a technical or code-related query where RAG might be useful.
    _TECHNICAL_KEYWORDS = {
        'python', 'javascript', 'java', 'c#', 'c++', 'php', 'ruby', 'go', 'swift', 'kotlin', 'typescript',
        'code', 'error', 'fix', 'implement', 'debug', 'refactor', 'algorithm', 'function', 'class',
        'method', 'module', 'library', 'package', 'framework', 'api', 'sdk',
        'bug', 'issue', 'exception', 'traceback', 'syntax', 'logic',
        'install', 'pip', 'npm', 'maven', 'gradle', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
        'sql', 'database', 'query', 'schema', 'model', 'entity',
        'test', 'unittest', 'pytest', 'selenium', 'cypress',
        'git', 'merge', 'commit', 'branch', 'pull request',
        'config', 'yaml', 'json', 'xml', 'html', 'css', 'react', 'angular', 'vue',
        'pyside', 'pyqt', 'qt', 'widget', 'signal', 'slot',
        'self.', 'this.', 'super()', 'async', 'await', 'yield', 'lambda', 'decorator',
        'numpy', 'pandas', 'tensorflow', 'pytorch', 'scikit-learn',
        'vector db', 'embedding', 'chunking', 'chroma', 'similarity search',
        'my code', 'my project', 'in my file', 'the provided context', 'search docs', 'find relevant'
    }
    # Simple greetings or very short phrases that usually don't warrant RAG.
    _GREETING_PATTERNS = re.compile(
        r"^\s*(hi|hello|hey|yo|sup|good\s+(morning|afternoon|evening)|how\s+are\s+you|ok|thanks|thank you)\b.*",
        re.IGNORECASE)
    _CODE_FENCE_PATTERN = re.compile(r"```")  # Presence of code fences might indicate user providing context.

    # Boost factors for re-ranking search results
    EXPLICIT_FOCUS_BOOST_FACTOR = 0.50  # Stronger boost for explicitly focused files (e.g., currently open in IDE)
    IMPLICIT_FOCUS_BOOST_FACTOR = 0.70  # Moderate boost for implicitly focused (e.g., recently accessed)
    ENTITY_BOOST_FACTOR = 0.80  # Slight boost if query entities match chunk entities

    def __init__(self, upload_service: Optional[UploadService], vector_db_service: Optional[VectorDBService]):
        self._upload_service: Optional[UploadService] = None
        self._vector_db_service: Optional[VectorDBService] = None

        if not isinstance(upload_service, UploadService):  # type: ignore
            logger.warning("RagSystem initialized with invalid or missing UploadService. RAG will be non-functional.")
        else:
            self._upload_service = upload_service

        if not isinstance(vector_db_service, VectorDBService):  # type: ignore
            logger.warning("RagSystem initialized with invalid or missing VectorDBService. RAG will be non-functional.")
        else:
            self._vector_db_service = vector_db_service

        services_status = f"UploadService={'OK' if self._upload_service else 'Missing'}, VectorDBService={'OK' if self._vector_db_service else 'Missing'}"
        logger.info(f"RagSystem initialized. {services_status}")

    def should_perform_rag(self, query: str, rag_globally_available: bool,
                           current_context_rag_initialized: bool) -> bool:
        """
        Determines if RAG should be performed for a given query.
        Considers query content, RAG availability, and whether the current context (project/global) has RAG data.
        """
        if not rag_globally_available:  # If the core RAG system (embedder, DB client) isn't ready
            return False
        if not current_context_rag_initialized:  # If the specific collection (project or global) has no data or isn't ready
            logger.debug("RAG not performed: Current context (project/global collection) is not initialized or empty.")
            return False
        if not query or not query.strip():
            return False

        query_lower = query.lower().strip()

        # Don't use RAG for very short queries or simple greetings
        if len(query_lower) < 15 and self._GREETING_PATTERNS.match(query_lower):
            return False
        if len(query_lower) < 10 and ' ' not in query_lower:  # Single short word
            return False

        # If user is pasting code, they are providing context, RAG might be less critical or even confusing
        # if self._CODE_FENCE_PATTERN.search(query):
        #     logger.debug("Query contains code fences, potentially skipping RAG or using it cautiously.")
        #     return False # Or return True with a flag for cautious RAG

        # Check for technical keywords
        if any(keyword in query_lower for keyword in self._TECHNICAL_KEYWORDS):
            logger.debug(f"RAG indicated by technical keyword in query: '{query_lower[:50]}...'")
            return True

        # Heuristic: if query contains characters common in code or technical discussion
        if re.search(r"[_.(){}\[\]=:/\\#]", query) and len(query) > 15:  # Added more symbols
            logger.debug(f"RAG indicated by special characters in query: '{query_lower[:50]}...'")
            return True

        logger.debug(f"RAG not indicated for query: '{query_lower[:50]}...'")
        return False

    def extract_code_entities(self, query: str) -> Set[str]:
        """
        Extracts potential code-related entities (variable names, function names, filenames) from a query.
        """
        entities: Set[str] = set()
        if not query: return entities

        # Regex for typical identifiers (variable, function, class names)
        # Allows underscores, alphanumeric, must start with letter or underscore.
        identifier_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b'  # Min length 3 to reduce noise
        # Regex for filenames (e.g., main.py, utils.js, MyClass.java)
        filename_pattern = r'\b([a-zA-Z0-9_.-]+\.[a-zA-Z0-9_]+)\b'  # More specific for filenames
        # Regex for CamelCase or PascalCase (often class names)
        class_name_pattern = r'\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*)\b'

        try:
            for pattern in [identifier_pattern, filename_pattern, class_name_pattern]:
                for match in re.finditer(pattern, query):
                    entity = match.group(1)
                    # Filter out very common Python keywords or generic terms if they are not likely entities
                    if entity.lower() not in ['def', 'class', 'self', 'init', 'return', 'true', 'false', 'none',
                                              'import', 'from', 'is', 'in', 'for', 'while', 'if', 'else', 'elif', 'try',
                                              'except', 'with', 'as', 'and', 'or', 'not', 'str', 'int', 'list', 'dict',
                                              'file', 'path', 'module', 'error', 'type', 'name', 'data', 'value',
                                              'result', 'info', 'test', 'this']:
                        entities.add(entity)
        except Exception as e_regex:
            logger.warning(f"Regex error during query entity extraction: {e_regex}")

        if entities: logger.debug(f"Extracted potential code entities from query: {entities}")
        return entities

    def get_formatted_context(
            self,
            query: str,
            query_entities: Set[str],
            project_id: Optional[str],  # Project ID for project-specific RAG
            explicit_focus_paths: Optional[List[str]] = None,  # Files currently open/focused in IDE
            implicit_focus_paths: Optional[List[str]] = None,  # Files recently accessed/related
            is_modification_request: bool = False  # If true, might fetch more context or different types
    ) -> Tuple[str, List[str]]:  # Returns (formatted_context_string, list_of_queried_collection_names)
        """
        Retrieves relevant context from vector DB, re-ranks based on focus, and formats it.
        """
        if not self._upload_service or not self._vector_db_service:
            logger.warning("RagSystem: UploadService or VectorDBService not available. Cannot retrieve RAG context.")
            return "", []

        context_str = ""
        queried_collections_for_log: List[str] = []

        # Determine which collections to query: project-specific and/or global
        collections_to_query: List[str] = []
        if project_id and project_id != GLOBAL_COLLECTION_ID and self._vector_db_service.is_ready(
                project_id):  # type: ignore
            collections_to_query.append(project_id)
        if self._vector_db_service.is_ready(GLOBAL_COLLECTION_ID):  # type: ignore
            if GLOBAL_COLLECTION_ID not in collections_to_query:  # Avoid duplicate if project_id is global
                collections_to_query.append(GLOBAL_COLLECTION_ID)  # type: ignore

        if not collections_to_query:
            logger.warning("RAG context requested, but no ready collections (project or global) to query.")
            return "", []

        logger.info(
            f"RagSystem: Attempting RAG retrieval from collections: {collections_to_query} for query: '{query[:50]}...'")

        try:
            if not hasattr(self._upload_service, 'query_vector_db'):
                logger.error("UploadService is missing the 'query_vector_db' method.")
                return "", []

            num_initial_results = getattr(constants, 'RAG_NUM_RESULTS', 5) * (3 if is_modification_request else 2)
            num_final_results = getattr(constants, 'RAG_NUM_RESULTS', 5)

            # UploadService now queries all listed collections and returns combined results
            # Each result should have metadata indicating its source collection.
            relevant_chunks = self._upload_service.query_vector_db(
                query_text=query,
                collection_ids=collections_to_query,
                n_results=num_initial_results
            )

            queried_collections_for_log = list(set(
                chunk.get("metadata", {}).get("retrieved_from_collection", "N/A") for chunk in relevant_chunks
                if chunk.get("metadata", {}).get("retrieved_from_collection") != "N/A"
            ))

            # Re-ranking logic
            if relevant_chunks:
                normalized_explicit_focus = {os.path.normcase(os.path.abspath(p)) for p in
                                             explicit_focus_paths} if explicit_focus_paths else set()
                normalized_implicit_focus = {os.path.normcase(os.path.abspath(p)) for p in
                                             implicit_focus_paths} if implicit_focus_paths else set()

                boost_counts = {"explicit": 0, "implicit": 0, "entity": 0}

                for chunk in relevant_chunks:
                    metadata = chunk.get('metadata', {})
                    if not isinstance(metadata, dict): continue

                    boost_applied_this_chunk = False
                    chunk_source_path = metadata.get('source')  # Absolute path from DB
                    norm_chunk_path = os.path.normcase(chunk_source_path) if chunk_source_path else None

                    # Apply boosts (lower distance is better)
                    if norm_chunk_path:
                        if normalized_explicit_focus and any(
                                norm_chunk_path.startswith(os.path.normcase(focus_dir) + os.sep) if os.path.isdir(
                                        focus_dir) else norm_chunk_path == os.path.normcase(focus_dir) for focus_dir in
                                normalized_explicit_focus):
                            chunk['distance'] = chunk.get('distance', 1.0) * self.EXPLICIT_FOCUS_BOOST_FACTOR
                            chunk['boost_reason'] = 'explicit_focus'
                            boost_counts["explicit"] += 1
                            boost_applied_this_chunk = True
                        elif not boost_applied_this_chunk and normalized_implicit_focus and any(
                                norm_chunk_path.startswith(os.path.normcase(focus_dir) + os.sep) if os.path.isdir(
                                        focus_dir) else norm_chunk_path == os.path.normcase(focus_dir) for focus_dir in
                                normalized_implicit_focus):
                            chunk['distance'] = chunk.get('distance', 1.0) * self.IMPLICIT_FOCUS_BOOST_FACTOR
                            chunk['boost_reason'] = 'implicit_focus'
                            boost_counts["implicit"] += 1
                            boost_applied_this_chunk = True

                    if not boost_applied_this_chunk and query_entities:
                        chunk_entities_str = metadata.get('code_entities', "")  # Comma-separated string
                        if chunk_entities_str:
                            chunk_entities_set = set(e.strip() for e in chunk_entities_str.split(',') if e.strip())
                            if not query_entities.isdisjoint(chunk_entities_set):
                                chunk['distance'] = chunk.get('distance', 1.0) * self.ENTITY_BOOST_FACTOR
                                chunk['boost_reason'] = 'entity_match'
                                boost_counts["entity"] += 1

                if sum(boost_counts.values()) > 0:
                    logger.info(
                        f"RAG re-ranking boosts applied: Explicit={boost_counts['explicit']}, Implicit={boost_counts['implicit']}, Entity={boost_counts['entity']}")

                valid_chunks = [r for r in relevant_chunks if isinstance(r.get('distance'), (int, float))]
                sorted_results = sorted(valid_chunks, key=lambda x: x.get('distance', float('inf')))
                final_results = sorted_results[:num_final_results]

                context_parts = []
                retrieved_details_for_log = []
                for i, chunk_data in enumerate(final_results):
                    meta = chunk_data.get("metadata", {})
                    fname = meta.get("filename", os.path.basename(meta.get("source", "unknown_source")))
                    coll_id_disp = meta.get("retrieved_from_collection", meta.get("collection_id", "N/A"))
                    content = chunk_data.get("content", "[Content Missing]")
                    dist = chunk_data.get('distance', -1.0)
                    s_line = meta.get('start_line', 'N/A')
                    e_line = meta.get('end_line', 'N/A')
                    boost_info = f", Boost: {chunk_data['boost_reason']}" if 'boost_reason' in chunk_data else ""

                    debug_info = f"Lines {s_line}-{e_line}, Dist: {dist:.4f}{boost_info}"
                    context_parts.append(
                        f"--- Snippet {i + 1} from `{fname}` (Source: {coll_id_disp}) ({debug_info}) ---\n"
                        f"```\n{content}\n```\n"  # Assuming text/code, not specifying language for simplicity
                    )
                    retrieved_details_for_log.append(f"{fname}({coll_id_disp}, {debug_info})")

                if context_parts:
                    context_str = "--- Relevant Code Context Start ---\n" + "\n".join(
                        context_parts) + "--- Relevant Code Context End ---"
                    logger.info(
                        f"Final RAG context: {len(final_results)} chunks from {len(queried_collections_for_log)} collection(s): [{', '.join(retrieved_details_for_log)}]")
                else:
                    logger.info("No valid chunks remained after re-ranking/processing.")
            else:
                logger.info(f"No relevant RAG context found in collections {collections_to_query}.")

        except Exception as e_rag_get:
            logger.error(f"Error during RAG context retrieval/formatting: {e_rag_get}", exc_info=True)
            context_str = "[Error retrieving RAG context]"  # Placeholder for LLM

        return context_str, queried_collections_for_log