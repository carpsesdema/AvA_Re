# app/services/upload_service.py
import asyncio
import datetime
import logging
import os
from html import escape
from typing import List, Tuple, Optional, Set, Dict, Any

# Ensure numpy is imported if used, typically by SentenceTransformer or direct embedding manipulation.
# It's listed in requirements.txt, so it should be available.
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False
    logging.getLogger(__name__).error(
        "UploadService: Numpy library not found. RAG DB cannot function if embeddings are numpy arrays.", exc_info=True)

try:
    from sentence_transformers import SentenceTransformer

    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # A common default
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    SentenceTransformer = None  # type: ignore
    EMBEDDINGS_AVAILABLE = False
    DEFAULT_EMBEDDING_MODEL = "fallback_model_name"
    logging.getLogger(__name__).error(
        f"UploadService: SentenceTransformer library import failed ({e}). RAG will likely fail.", exc_info=True)

try:
    from core.event_bus import EventBus  # Assuming EventBus might be used for status updates
    from utils import constants
    from models.chat_message import ChatMessage
    from models.message_enums import SYSTEM_ROLE, ERROR_ROLE
    from .chunking_service import ChunkingService
    from .vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID
    from .file_handler_service import FileHandlerService
    from .code_analysis_service import CodeAnalysisService
except ImportError as e_upload_service:
    logging.getLogger(__name__).critical(f"Critical import error in UploadService: {e_upload_service}", exc_info=True)
    # Define fallbacks for type hinting if imports fail
    EventBus = type("EventBus", (object,), {})
    constants = type("constants", (object,),
                     {"RAG_CHUNK_SIZE": 1000, "RAG_CHUNK_OVERLAP": 150, "ALLOWED_TEXT_EXTENSIONS": set(),
                      "DEFAULT_IGNORED_DIRS": set(), "MAX_SCAN_DEPTH": 5, "RAG_MAX_FILE_SIZE_MB": 50})
    ChatMessage = type("ChatMessage", (object,), {})
    SYSTEM_ROLE, ERROR_ROLE = "system", "error"
    ChunkingService = type("ChunkingService", (object,), {})
    VectorDBService = type("VectorDBService", (object,), {})
    GLOBAL_COLLECTION_ID = "global_knowledge_fallback_us"
    FileHandlerService = type("FileHandlerService", (object,), {})
    CodeAnalysisService = type("CodeAnalysisService", (object,), {})
    raise  # Re-raise to ensure the issue is visible

logger = logging.getLogger(__name__)
CHROMA_DB_UPLOAD_BATCH_SIZE = 4000  # Default, can be moved to constants


class UploadService:
    def __init__(self):
        logger.info("UploadService initializing...")
        self._embedder: Optional[SentenceTransformer] = None
        self._chunking_service: Optional[ChunkingService] = None
        self._vector_db_service: Optional[VectorDBService] = None
        self._file_handler_service: Optional[FileHandlerService] = None
        self._code_analysis_service: Optional[CodeAnalysisService] = None
        self._index_dim: int = -1  # Will be set after embedder initialization
        self._dependencies_ready: bool = False
        self._embedder_init_task: Optional[asyncio.Task] = None
        self._embedder_ready: bool = False

        if not all([EMBEDDINGS_AVAILABLE, ChunkingService, VectorDBService, FileHandlerService, CodeAnalysisService,
                    NUMPY_AVAILABLE]):
            logger.critical(
                "UploadService cannot initialize due to missing critical dependencies (SentenceTransformer, Numpy, or other services). RAG functionality will be disabled.")
            return

        try:
            self._file_handler_service = FileHandlerService()
            self._chunking_service = ChunkingService(
                chunk_size=getattr(constants, 'RAG_CHUNK_SIZE', 1000),
                chunk_overlap=getattr(constants, 'RAG_CHUNK_OVERLAP', 150)
            )
            self._code_analysis_service = CodeAnalysisService()
            # Initialize VectorDB with a placeholder dimension; it will be updated
            self._vector_db_service = VectorDBService(index_dimension=384)  # Default for MiniLM

            self._start_embedder_initialization()  # Initialize embedder in the background

        except Exception as e:
            logger.exception(f"CRITICAL FAILURE during UploadService component initialization: {e}")
            self._dependencies_ready = False

    def _start_embedder_initialization(self):
        if not EMBEDDINGS_AVAILABLE:
            logger.error("Cannot initialize embedder: SentenceTransformers library not available.")
            self._embedder_ready = False
            self._dependencies_ready = False
            self._emit_rag_status(False, "RAG: Embedder Lib Missing", "#ef4444")
            return

        logger.info("Starting background embedder initialization...")
        self._embedder_init_task = asyncio.create_task(self._initialize_embedder_async())

    async def _initialize_embedder_async(self):
        try:
            logger.info(f"Initializing SentenceTransformer ('{DEFAULT_EMBEDDING_MODEL}') in background...")
            loop = asyncio.get_event_loop()
            self._embedder = await loop.run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                lambda: SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            )

            if self._embedder:
                logger.info("SentenceTransformer initialized successfully.")
                # Test embedding and get dimension
                test_embedding = await loop.run_in_executor(None,
                                                            lambda: self._embedder.encode(["test"]))  # type: ignore
                if test_embedding is not None and hasattr(test_embedding, 'shape') and len(test_embedding.shape) > 1:
                    self._index_dim = test_embedding.shape[1]
                    logger.info(f"Detected embedding dimension: {self._index_dim}")
                    if self._vector_db_service:
                        self._vector_db_service._index_dim = self._index_dim  # Update VDB service
                    self._embedder_ready = True
                    # Check if VDB service is also ready (it should be if client init was okay)
                    self._dependencies_ready = self._vector_db_service.is_ready() if self._vector_db_service else False
                    if self._dependencies_ready:
                        logger.info("UploadService fully ready after background embedder initialization.")
                        self._emit_rag_status(True, "RAG: Ready ✓", "#4ade80")
                    else:
                        logger.error("VectorDBService not ready after embedder initialization.")
                        self._emit_rag_status(False, "RAG: VectorDB Error", "#ef4444")
                else:
                    logger.error("Failed to get valid embedding shape after SentenceTransformer init.")
                    raise ValueError("Failed to get valid embedding shape.")
            else:
                logger.error("SentenceTransformer failed to initialize (returned None).")
                raise ValueError("SentenceTransformer failed to initialize.")

        except Exception as e:
            logger.exception(f"Failed to initialize embedder in background: {e}")
            self._embedder_ready = False
            self._dependencies_ready = False
            self._emit_rag_status(False, "RAG: Embedder Init Failed", "#ef4444")

    def _emit_rag_status(self, is_ready: bool, message: str, color: str):
        try:
            event_bus = EventBus.get_instance()
            event_bus.ragStatusChanged.emit(is_ready, message, color)
        except Exception as e_emit:
            logger.error(f"Failed to emit RAG status change: {e_emit}")

    def is_vector_db_ready(self, collection_id: Optional[str] = None) -> bool:
        if not self._embedder_ready or not self._dependencies_ready or not self._vector_db_service:
            return False
        return self._vector_db_service.is_ready(collection_id)

    async def wait_for_embedder_ready(self, timeout_seconds: float = 30.0) -> bool:
        if self._embedder_ready: return True
        if not self._embedder_init_task:
            logger.error("No embedder initialization task running to wait for.")
            return False
        try:
            await asyncio.wait_for(self._embedder_init_task, timeout=timeout_seconds)
            return self._embedder_ready
        except asyncio.TimeoutError:
            logger.error(f"Embedder initialization timed out after {timeout_seconds} seconds.")
            return False
        except Exception as e:
            logger.error(f"Error waiting for embedder: {e}")
            return False

    def _send_batch_to_db(self, collection_id: str,
                          batch_contents: List[str],
                          batch_embeddings: List[List[float]],  # Assuming embeddings are List[float]
                          batch_metadatas: List[Dict[str, Any]],
                          files_in_this_batch_names: Set[str]) -> Tuple[bool, Set[str], int]:
        if not batch_contents: return True, set(), 0
        if not self._vector_db_service:
            logger.error("UploadService: _vector_db_service is None in _send_batch_to_db.")
            return False, set(), 0

        num_embeddings_in_batch = len(batch_embeddings)
        try:
            logger.info(
                f"Sending batch of {len(batch_contents)} docs from {len(files_in_this_batch_names)} files to coll '{collection_id}'...")
            success = self._vector_db_service.add_embeddings(collection_id, batch_contents, batch_embeddings,
                                                             batch_metadatas)
            if success:
                logger.info(
                    f"Successfully added {len(batch_contents)} documents ({num_embeddings_in_batch} embeddings) to '{collection_id}'.")
                return True, files_in_this_batch_names, num_embeddings_in_batch
            else:
                logger.error(
                    f"DB service reported failure for batch to coll '{collection_id}'. Files: {files_in_this_batch_names}")
                return False, set(), 0
        except Exception as e:
            logger.exception(
                f"Exception during batch add to coll '{collection_id}': {e}. Files: {files_in_this_batch_names}")
            return False, set(), 0

    async def process_files_for_context_async(self, file_paths: List[str], collection_id: str) -> Optional[ChatMessage]:
        if not await self.wait_for_embedder_ready():
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[Error: RAG embedder not ready. Please try again.]"])  # type: ignore
        return self.process_files_for_context(file_paths, collection_id)

    def process_files_for_context(self, file_paths: List[str], collection_id: str) -> Optional[
        ChatMessage]:  # type: ignore
        if not collection_id:
            logger.error("UploadService: process_files_for_context called without a collection_id.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: Collection ID for RAG processing is missing.]"])  # type: ignore
        if not isinstance(file_paths, list):
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: Invalid input provided for file paths.]"])  # type: ignore

        num_input_files = len(file_paths)
        logger.info(f"Processing {num_input_files} files for RAG collection '{collection_id}'...")

        if not self._embedder_ready or not self._embedder:
            logger.error("UploadService: Embedder not ready for file processing.")
            return ChatMessage(role=ERROR_ROLE, parts=[
                "[Error: RAG embedder not ready. Please wait for initialization.]"])  # type: ignore
        if not self._dependencies_ready:
            logger.error("UploadService: Core dependencies (VDB, services) not ready. Cannot process files for RAG.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[Error: RAG core components failed to initialize.]"])  # type: ignore
        if not self.is_vector_db_ready(collection_id):
            logger.error(f"UploadService: Vector DB/collection '{collection_id}' not ready. Cannot process files.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=[f"[Error: RAG DB/collection '{collection_id}' not ready.]"])  # type: ignore
        if not all([self._file_handler_service, self._chunking_service, self._code_analysis_service]):
            logger.error("UploadService: One or more internal services are None.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: RAG processing sub-components not ready.]"])  # type: ignore

        overall_successfully_added_files: Set[str] = set()
        processing_error_files_dict: Dict[str, str] = {}
        db_failed_files_exclusive_set: Set[str] = set()
        binary_skipped_files: Set[str] = set()
        no_content_files: Set[str] = set()
        total_chunks_generated = 0
        total_embeddings_in_successful_batches = 0
        total_embeddings_in_failed_batches = 0
        any_db_batch_add_failure_occurred = False

        current_batch_contents: List[str] = []
        current_batch_embeddings: List[List[float]] = []
        current_batch_metadatas: List[Dict[str, Any]] = []
        current_batch_file_names_involved: Set[str] = set()

        for i, file_path in enumerate(file_paths):
            display_name = os.path.basename(file_path)
            if not os.path.exists(file_path): processing_error_files_dict[escape(display_name)] = "Not Found"; continue
            if not os.path.isfile(file_path): processing_error_files_dict[escape(display_name)] = "Not a File"; continue

            logger.info(f"  Processing [{i + 1}/{num_input_files}]: {display_name} for collection '{collection_id}'")
            read_result = self._file_handler_service.read_file_content(file_path)
            if not read_result or len(read_result) != 3: processing_error_files_dict[
                escape(display_name)] = "Internal Read Error"; continue
            content, file_type, error_msg = read_result
            if file_type == "error": processing_error_files_dict[
                escape(display_name)] = error_msg or "Read Error"; continue
            if file_type == "binary": binary_skipped_files.add(escape(display_name)); continue

            code_structures: List[Dict[str, Any]] = []
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.py' and self._code_analysis_service and content:
                try:
                    code_structures = self._code_analysis_service.parse_python_structures(content, file_path)
                except Exception as e_code_parse:
                    logger.warning(f"Error parsing code structures for {display_name}: {e_code_parse}", exc_info=True)
                    processing_error_files_dict[escape(display_name)] = "Code Parse Error"

            if escape(display_name) in processing_error_files_dict and "Code Parse Error" not in \
                    processing_error_files_dict[escape(display_name)]:
                continue  # Skip if other critical error occurred before parsing

            if not content and not code_structures:  # If no text content and no structures (e.g. empty .py file)
                no_content_files.add(escape(display_name))
                continue

            # Use content for chunking, even if code parsing had issues but there's text
            text_to_chunk = content if content else ""

            try:
                chunks = self._chunking_service.chunk_document(text_to_chunk, source_id=file_path, file_ext=file_ext)
                if not chunks and text_to_chunk.strip():  # If content existed but no chunks were made
                    no_content_files.add(escape(display_name))
                    continue
                elif not chunks:  # No content and no chunks
                    continue

                total_chunks_generated += len(chunks)
                file_specific_chunk_contents: List[str] = []
                file_specific_metadatas: List[Dict[str, Any]] = []

                for chunk_data in chunks:
                    if not isinstance(chunk_data,
                                      dict) or 'metadata' not in chunk_data or 'content' not in chunk_data: continue
                    md = chunk_data['metadata']
                    # Associate code entities with chunks based on line numbers
                    entities_in_chunk = [
                        s["name"] for s in code_structures
                        if s.get("start_line", -1) <= md.get('end_line', 0) and \
                           s.get("end_line", 0) >= md.get('start_line', -1) and \
                           s.get("name")
                    ]
                    md['code_entities'] = ", ".join(sorted(list(set(entities_in_chunk)))) if entities_in_chunk else ""
                    md['collection_id'] = collection_id  # Ensure collection_id is in metadata for each chunk
                    file_specific_chunk_contents.append(chunk_data['content'])
                    file_specific_metadatas.append(md)

                if not file_specific_chunk_contents:
                    if text_to_chunk.strip(): no_content_files.add(escape(display_name))
                    continue

                embeddings_np = self._embedder.encode(file_specific_chunk_contents,
                                                      show_progress_bar=False)  # type: ignore
                if not isinstance(embeddings_np, np.ndarray) or embeddings_np.ndim != 2:
                    logger.error(f"Embedder did not return a 2D numpy array for {display_name}. Skipping file.")
                    processing_error_files_dict[escape(display_name)] = "Embedding Error"
                    continue

                current_batch_contents.extend(file_specific_chunk_contents)
                current_batch_embeddings.extend(embeddings_np.tolist())
                current_batch_metadatas.extend(file_specific_metadatas)
                current_batch_file_names_involved.add(escape(display_name))

                if len(current_batch_contents) >= CHROMA_DB_UPLOAD_BATCH_SIZE:
                    batch_ok, files_added_names, embs_count = self._send_batch_to_db(collection_id,
                                                                                     current_batch_contents,
                                                                                     current_batch_embeddings,
                                                                                     current_batch_metadatas,
                                                                                     current_batch_file_names_involved)
                    if batch_ok:
                        overall_successfully_added_files.update(
                            files_added_names); total_embeddings_in_successful_batches += embs_count
                    else:
                        any_db_batch_add_failure_occurred = True
                        total_embeddings_in_failed_batches += embs_count
                        for f_name_failed in current_batch_file_names_involved:
                            if f_name_failed not in processing_error_files_dict: db_failed_files_exclusive_set.add(
                                f_name_failed)
                    current_batch_contents, current_batch_embeddings, current_batch_metadatas, current_batch_file_names_involved = [], [], [], set()
            except Exception as e_proc:
                logger.exception(f"Error processing file {display_name} for collection '{collection_id}': {e_proc}")
                processing_error_files_dict[escape(display_name)] = "Processing Error"

        if current_batch_contents:  # Process any remaining batch
            batch_ok, files_added_names, embs_count = self._send_batch_to_db(collection_id, current_batch_contents,
                                                                             current_batch_embeddings,
                                                                             current_batch_metadatas,
                                                                             current_batch_file_names_involved)
            if batch_ok:
                overall_successfully_added_files.update(
                    files_added_names); total_embeddings_in_successful_batches += embs_count
            else:
                any_db_batch_add_failure_occurred = True
                total_embeddings_in_failed_batches += embs_count
                for f_name_failed in current_batch_file_names_involved:
                    if f_name_failed not in processing_error_files_dict: db_failed_files_exclusive_set.add(
                        f_name_failed)

        # Construct summary message
        if num_input_files == 0: return ChatMessage(role=SYSTEM_ROLE, parts=[
            f"[RAG Upload: No files provided for collection '{collection_id}'.]"])  # type: ignore

        status_notes = []
        if overall_successfully_added_files: status_notes.append(
            f"{len(overall_successfully_added_files)} file(s) added ({total_embeddings_in_successful_batches} embeddings)")
        if db_failed_files_exclusive_set:
            status_notes.append(
                f"{len(db_failed_files_exclusive_set)} file(s) failed DB add ({total_embeddings_in_failed_batches} attempted embeddings)")
        elif any_db_batch_add_failure_occurred and not overall_successfully_added_files:
            status_notes.append(f"All DB batches failed ({total_embeddings_in_failed_batches} attempted embeddings)")
        if processing_error_files_dict: status_notes.append(
            f"{len(processing_error_files_dict)} file(s) with processing errors")
        if binary_skipped_files: status_notes.append(f"{len(binary_skipped_files)} binary file(s) ignored")

        actual_no_content_files = no_content_files - set(processing_error_files_dict.keys())  # Don't double-count
        if actual_no_content_files: status_notes.append(
            f"{len(actual_no_content_files)} file(s) yielded no RAG content")

        message_role = ERROR_ROLE if processing_error_files_dict or any_db_batch_add_failure_occurred else SYSTEM_ROLE
        summary_parts = [f"RAG Upload: Processed {num_input_files} item(s) for Collection ID '{collection_id}'."]
        if status_notes:
            summary_parts.append("Summary: " + "; ".join(status_notes) + ".")
        else:
            summary_parts.append("Summary: Processing complete. No items added or errors noted (check logs).")

        issue_details_list = [f"'{fname} ({reason})'" for fname, reason in processing_error_files_dict.items()]
        for fname_db_failed in db_failed_files_exclusive_set: issue_details_list.append(
            f"'{escape(fname_db_failed)} (DB Batch Add Failed)'")
        if issue_details_list:
            issue_details_str = ", ".join(sorted(list(set(issue_details_list))))
            if len(issue_details_str) > 250: issue_details_str = issue_details_str[:247] + "...'"  # Truncate
            summary_parts.append(f"Issue details: {issue_details_str}")

        summary_text = " ".join(summary_parts)
        logger.info(f"UploadService: Finished processing for collection '{collection_id}'. Final: {summary_text}")
        return ChatMessage(role=message_role, parts=[summary_text], timestamp=datetime.datetime.now().isoformat(),
                           # type: ignore
                           metadata={"upload_summary_v5_final": {
                               # Changed key to avoid potential type conflicts if old keys exist
                               "input_files": num_input_files,
                               "successfully_added_files": len(overall_successfully_added_files),
                               "embeddings_in_successful_batches": total_embeddings_in_successful_batches,
                               "processing_error_files": len(processing_error_files_dict),
                               "db_failed_files_exclusive": len(db_failed_files_exclusive_set),
                               "embeddings_in_failed_db_batches": total_embeddings_in_failed_batches,
                               "binary_skipped_files": len(binary_skipped_files),
                               "no_content_files": len(actual_no_content_files),
                               "collection_id": collection_id,
                               "total_chunks_generated": total_chunks_generated
                           }})

    def process_directory_for_context(self, dir_path: str, collection_id: str) -> Optional[ChatMessage]:  # type: ignore
        logger.info(f"Processing directory '{dir_path}' for RAG collection '{collection_id}'")
        if not collection_id:
            logger.error("UploadService: process_directory_for_context called without a collection_id.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: Collection ID for RAG processing is missing.]"])  # type: ignore
        try:
            if not os.path.isdir(dir_path):
                return ChatMessage(role=ERROR_ROLE,
                                   parts=[f"[Error: Not a directory: '{os.path.basename(dir_path)}']"])  # type: ignore

            if not self._file_handler_service:  # Add this check
                logger.error("UploadService: FileHandlerService not initialized in process_directory_for_context.")
                return ChatMessage(role=ERROR_ROLE,
                                   parts=["[System Error: File handling service not ready.]"])  # type: ignore

            valid_files, skipped_scan_info = self._scan_directory(dir_path)
            if not valid_files:
                msg = f"[RAG Scan: No suitable files in '{os.path.basename(dir_path)}' for collection '{collection_id}'."
                if skipped_scan_info:
                    msg += f" {len(skipped_scan_info)} items skipped/errored during scan.]"
                else:
                    msg += "]"
                return ChatMessage(role=SYSTEM_ROLE, parts=[msg])  # type: ignore

            process_msg_obj = self.process_files_for_context(valid_files, collection_id=collection_id)
            if process_msg_obj and skipped_scan_info:  # Add scan issue info to the processing message
                scan_issue_txt = f"{len(skipped_scan_info)} items skipped/error in dir scan."
                if process_msg_obj.metadata: process_msg_obj.metadata[
                    "dir_scan_issues"] = scan_issue_txt  # type: ignore
                if process_msg_obj.parts and isinstance(process_msg_obj.parts[0], str):  # type: ignore
                    process_msg_obj.parts[0] = process_msg_obj.parts[0].rstrip(
                        ' .]') + f" | DirScan: {scan_issue_txt}]"  # type: ignore
            return process_msg_obj
        except Exception as e:
            logger.exception(f"CRITICAL ERROR processing directory '{dir_path}' for collection '{collection_id}': {e}")
            return ChatMessage(role=ERROR_ROLE, parts=[
                f"[System: Critical error processing directory '{os.path.basename(dir_path)}' for collection '{collection_id}'.]"])  # type: ignore

    def query_vector_db(self, query_text: str, collection_ids: List[str], n_results: int = constants.RAG_NUM_RESULTS) -> \
    List[Dict[str, Any]]:  # type: ignore
        if not self._embedder_ready or not self._embedder or not self._vector_db_service:
            logger.warning("query_vector_db: Core dependencies (embedder/VDB service) not ready. Returning empty list.")
            return []
        if not query_text.strip(): return []
        n_results = max(1, n_results)  # Ensure k is at least 1
        if not collection_ids:
            logger.warning(
                "UploadService.query_vector_db called with empty collection_ids, defaulting to GLOBAL_COLLECTION_ID.")
            collection_ids = [GLOBAL_COLLECTION_ID]

        all_results: List[Dict[str, Any]] = []
        try:
            query_embedding_np = self._embedder.encode([query_text])  # type: ignore
            if not isinstance(query_embedding_np, np.ndarray) or query_embedding_np.ndim != 2 or \
                    query_embedding_np.shape[1] != self._index_dim:
                logger.error(
                    f"Failed to generate valid query embedding or dimension mismatch. Expected {self._index_dim}, got shape {query_embedding_np.shape if isinstance(query_embedding_np, np.ndarray) else 'N/A'}")
                return []
            query_embedding = query_embedding_np.tolist()

            for coll_id in collection_ids:
                if not coll_id or not isinstance(coll_id, str):
                    logger.warning(f"Skipping invalid collection_id in query: {coll_id}")
                    continue
                if self.is_vector_db_ready(coll_id):
                    logger.debug(f"Querying collection: {coll_id} for '{query_text[:30]}...'")
                    coll_results = self._vector_db_service.search(coll_id, query_embedding, k=n_results)
                    if coll_results:
                        for res_item in coll_results:  # Add source collection info
                            if 'metadata' not in res_item or res_item['metadata'] is None: res_item['metadata'] = {}
                            res_item['metadata']['retrieved_from_collection'] = coll_id
                        all_results.extend(coll_results)
                else:
                    logger.warning(f"Collection '{coll_id}' not ready, skipping in query.")
            # Optionally, re-rank or merge results from multiple collections if needed
            # For now, just concatenating. If distances are comparable, sorting by distance might be useful.
            if len(collection_ids) > 1 and all_results:
                all_results.sort(
                    key=lambda x: x.get('distance', float('inf')))  # Sort by distance if multiple collections
                all_results = all_results[:n_results]  # Trim to original n_results after merging

            return all_results
        except Exception as e:
            logger.exception(f"Error querying Vector DB: {e}")
            return []

    def _scan_directory(self, root_dir: str,
                        allowed_extensions: Set[str] = constants.ALLOWED_TEXT_EXTENSIONS,  # type: ignore
                        ignored_dirs: Set[str] = constants.DEFAULT_IGNORED_DIRS) -> Tuple[
        List[str], List[str]]:  # type: ignore
        valid_files: List[str] = []
        skipped_info: List[str] = []
        logger.info(f"Scanning directory: {root_dir}")

        ignored_dirs_lower = {d.lower() for d in ignored_dirs if isinstance(d, str)}
        allowed_extensions_lower = {e.lower() for e in allowed_extensions if isinstance(e, str)}
        max_depth = getattr(constants, 'MAX_SCAN_DEPTH', 5)
        max_size_mb = getattr(constants, 'RAG_MAX_FILE_SIZE_MB', 50)
        max_size_bytes = max_size_mb * 1024 * 1024

        if not os.path.isdir(root_dir):
            skipped_info.append(f"Error: Root path is not a directory: {root_dir}")
            return [], skipped_info
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True, onerror=None):
                try:
                    # Calculate current depth relative to root_dir
                    rel_dirpath = os.path.relpath(dirpath, root_dir)
                except ValueError:  # Happens if dirpath is not under root_dir (e.g. symlink outside)
                    logger.warning(
                        f"Cannot get relative path for {dirpath} from {root_dir}. Assuming depth 0 for this path.")
                    depth = 0  # Effectively, don't descend further into this odd path
                    rel_dirpath = dirpath  # Use absolute for logging if relpath fails

                if rel_dirpath != '.':  # relpath gives '.' for root_dir itself
                    depth = rel_dirpath.count(os.sep)
                else:
                    depth = 0

                if depth >= max_depth:
                    skipped_info.append(f"Max Depth ({max_depth}) reached: '{rel_dirpath}'")
                    dirnames[:] = []  # Don't descend further
                    continue

                # Filter dirnames in-place to prevent os.walk from traversing them
                original_dirnames = list(dirnames)  # Iterate over a copy
                dirnames[:] = []  # Clear current dirnames to rebuild
                for d in original_dirnames:
                    if d.startswith('.') or d.lower() in ignored_dirs_lower:
                        skipped_info.append(f"Ignored Directory: '{os.path.join(rel_dirpath, d)}'")
                    else:
                        dirnames.append(d)  # Keep this directory for traversal

                for filename in filenames:
                    rel_filepath = os.path.join(rel_dirpath, filename) if rel_dirpath != '.' else filename
                    full_path = os.path.join(dirpath, filename)

                    if filename.startswith('.'):
                        skipped_info.append(f"Ignored Hidden File: '{rel_filepath}'")
                        continue
                    if os.path.splitext(filename)[1].lower() not in allowed_extensions_lower:
                        skipped_info.append(f"Unsupported Extension: '{rel_filepath}'")
                        continue
                    try:
                        if not os.access(full_path, os.R_OK):  # Check readability
                            skipped_info.append(f"Unreadable File: '{rel_filepath}'")
                            continue
                        file_size = os.path.getsize(full_path)
                        if file_size == 0:
                            skipped_info.append(f"Empty File: '{rel_filepath}'")
                            continue
                        if file_size > max_size_bytes:
                            skipped_info.append(f"File Too Large ({file_size / (1024 * 1024):.1f}MB): '{rel_filepath}'")
                            continue
                    except OSError as e_stat:  # Catch errors during stat (e.g. broken symlink)
                        skipped_info.append(f"OS Error Accessing File ('{rel_filepath}'): {e_stat.strerror}")
                        continue

                    valid_files.append(full_path)
        except OSError as e_walk:  # Error during the os.walk itself
            skipped_info.append(f"OS Error During Directory Walk ('{root_dir}'): {e_walk.strerror}")
        except Exception as e_scan_unexpected:  # Catch any other unexpected errors
            skipped_info.append(f"Unexpected Error During Scan ('{root_dir}'): {type(e_scan_unexpected).__name__}")
            logger.error(f"Unexpected scan error in {root_dir}: {e_scan_unexpected}", exc_info=True)

        logger.info(
            f"Scan of '{os.path.basename(root_dir)}' found {len(valid_files)} processable files. Skipped/Errors reported for {len(skipped_info)} items.")
        if skipped_info:
            logger.debug(f"Skipped/Error items during scan of '{root_dir}':\n" + "\n".join(
                f"  - {s}" for s in skipped_info[:20]))  # Log first 20 skipped
        return valid_files, skipped_info