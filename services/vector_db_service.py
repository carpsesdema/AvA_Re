# app/services/vector_db_service.py
import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.api.models.Collection import Collection as ChromaCollection # Specific import for type hint
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None # type: ignore
    ChromaCollection = type("ChromaCollection", (object,), {}) # Fallback type hint
    CHROMADB_AVAILABLE = False
    logging.critical(
        "VectorDBService: ChromaDB library not found. RAG DB cannot function. Install: pip install chromadb")

try:
    from utils import constants
except ImportError:
    # Fallback if constants are not available (e.g. during isolated testing or initial setup)
    class constants_fallback:
        RAG_COLLECTIONS_PATH = os.path.join(os.path.expanduser("~"), ".ava_pys6_data_p1", "rag_collections")
        GLOBAL_COLLECTION_ID = "global_knowledge_fallback_vdb"
    constants = constants_fallback # type: ignore
    logging.getLogger(__name__).warning("VectorDBService: utils.constants not found, using fallback paths.")


GLOBAL_COLLECTION_ID = constants.GLOBAL_COLLECTION_ID
logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self, index_dimension: int, base_persist_directory: Optional[str] = None):
        logger.info("VectorDBService initializing (ChromaDB implementation)...")
        self._client: Optional[chromadb.PersistentClient] = None
        self._index_dim = index_dimension # This will be updated by UploadService once embedder is ready

        if not CHROMADB_AVAILABLE:
            logger.critical("ChromaDB library not available. VectorDBService cannot initialize.")
            return

        if not isinstance(index_dimension, int) or index_dimension <= 0:
            logger.critical(f"Invalid initial index dimension provided: {index_dimension}. Using a default of 384.")
            self._index_dim = 384 # Default for many common sentence transformers

        self.base_persist_directory = os.path.abspath(base_persist_directory or constants.RAG_COLLECTIONS_PATH)
        logger.info(f"VectorDBService: ChromaDB base_persist_directory set to: {self.base_persist_directory}")

        try:
            os.makedirs(self.base_persist_directory, exist_ok=True)
            logger.info(f"ChromaDB base directory ensured: {self.base_persist_directory}")
        except OSError as e:
            logger.critical(f"Failed to create ChromaDB base directory '{self.base_persist_directory}': {e}", exc_info=True)
            return

        try:
            logger.info(f"Attempting to initialize chromadb.PersistentClient with path: {self.base_persist_directory}")
            self._client = chromadb.PersistentClient(path=self.base_persist_directory)
            logger.info("ChromaDB PersistentClient initialized successfully.")

            # Ensure global collection exists (or can be created) upon initialization
            if not self.get_or_create_collection(GLOBAL_COLLECTION_ID): # type: ignore
                logger.error(
                    f"VectorDBService initialized, but the global collection ('{GLOBAL_COLLECTION_ID}') "
                    "could not be properly created or loaded. "
                    "RAG functionality relying on the global context may be impaired."
                )
            else:
                logger.info("VectorDBService fully initialized and global collection is ready.")

        except Exception as e:
            logger.critical(f"CRITICAL FAILURE: Failed to initialize ChromaDB client with path '{self.base_persist_directory}': {e}", exc_info=True)
            self._client = None

    def get_or_create_collection(self, collection_id: str) -> Optional[ChromaCollection]:
        if not CHROMADB_AVAILABLE or self._client is None:
            logger.error(
                f"Cannot get/create collection '{collection_id}': ChromaDB not available or client not initialized.")
            return None
        if not isinstance(collection_id, str) or not collection_id.strip():
            logger.error("Cannot get/create collection: Invalid or empty collection_id provided.")
            return None
        try:
            # ChromaDB's get_or_create_collection uses a default embedding function if none is specified,
            # but we handle embeddings externally, so we pass embedding_function=None (though it might be implicit).
            # The metadata={'hnsw:space': 'cosine'} is a common setting for sentence embeddings.
            collection = self._client.get_or_create_collection(
                name=collection_id,
                embedding_function=None, # We provide embeddings manually
                metadata={"hnsw:space": "cosine"} # Use cosine distance for similarity
            )
            logger.debug(f"Collection '{collection_id}' accessed/created successfully.")
            return collection
        except Exception as e:
            logger.error(f"Error getting/creating collection '{collection_id}': {type(e).__name__} - {e}", exc_info=True)
            try:
                if self._client:
                    logger.info(f"Current collections in DB: {[col.name for col in self._client.list_collections()]}")
            except Exception as list_e:
                logger.error(f"Failed to list collections after error: {list_e}")
            return None

    def is_ready(self, collection_id: Optional[str] = None) -> bool:
        if not CHROMADB_AVAILABLE or self._client is None:
            logger.debug("VectorDBService.is_ready: ChromaDB not available or client is None. Returning False.")
            return False
        if collection_id is None: # General client health check
            try:
                self._client.heartbeat()
                logger.debug("VectorDBService.is_ready: Client heartbeat successful. Returning True for general readiness.")
                return True
            except Exception as e:
                logger.error(f"VectorDBService.is_ready: Client heartbeat failed: {e}. Returning False.", exc_info=True)
                return False

        # Specific collection check
        try:
            # A more robust check is to try and get the collection. list_collections might not be enough
            # if the collection exists but is somehow corrupted or inaccessible for queries.
            collection = self._client.get_collection(name=collection_id)
            logger.debug(f"VectorDBService.is_ready: Collection '{collection_id}' exists and was retrieved.")
            return True
        except chromadb.errors.CollectionNotFoundError: # More specific exception
            logger.debug(f"VectorDBService.is_ready: Collection '{collection_id}' not found.")
            return False
        except Exception as e:
            logger.error(f"Error checking if collection '{collection_id}' exists: {e}", exc_info=True)
            return False


    def add_embeddings(self, collection_id: str, contents: List[str], embeddings: List[List[float]],
                       metadatas: List[Dict[str, Any]]) -> bool:
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot add embeddings: Collection '{collection_id}' not found or created.")
            return False

        if not (isinstance(contents, list) and isinstance(embeddings, list) and isinstance(metadatas, list)):
            logger.error("Invalid input: contents, embeddings, and metadatas must be lists.")
            return False
        if not (len(contents) == len(embeddings) == len(metadatas)):
            logger.error(
                f"Input length mismatch: contents ({len(contents)}), embeddings ({len(embeddings)}), metadatas ({len(metadatas)}).")
            return False
        if not contents:
            logger.warning(f"No contents provided to add to collection '{collection_id}'.")
            return True # Technically successful as there was nothing to add.

        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list) or len(emb) != self._index_dim:
                logger.error(
                    f"Embedding at index {i} for collection '{collection_id}' has incorrect dimension or type: "
                    f"length {len(emb) if isinstance(emb, list) else 'N/A'} (type: {type(emb)}). Expected {self._index_dim} floats."
                )
                return False

        logger.info(f"Adding {len(contents)} documents to collection '{collection_id}'...")
        # Generate unique IDs for ChromaDB if not provided in metadata
        ids = [metadata.get('id', f"doc-{uuid.uuid4().hex}") for metadata in metadatas]
        # Ensure all metadatas are dicts, even if empty
        sanitized_metadatas = [md if isinstance(md, dict) else {} for md in metadatas]


        try:
            collection.add(
                documents=contents,
                embeddings=embeddings,
                metadatas=sanitized_metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(contents)} documents to '{collection_id}'.")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to collection '{collection_id}': {e}", exc_info=True)
            return False

    def remove_document_chunks_by_source(self, collection_id: str, source_path_to_remove: str) -> bool:
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot remove documents: Collection '{collection_id}' not found.")
            return False

        logger.info(f"Removing documents with source '{source_path_to_remove}' from '{collection_id}'...")
        try:
            # This assumes 'source' is a key in your metadata
            collection.delete(where={"source": source_path_to_remove})
            logger.info(f"Successfully requested deletion of documents with source '{source_path_to_remove}' from '{collection_id}'.")
            return True
        except Exception as e:
            logger.error(f"Error removing documents by source from '{collection_id}': {e}", exc_info=True)
            return False

    def search(self, collection_id: str, query_embedding: List[List[float]], k: int = 5) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot search: Collection '{collection_id}' not found.")
            return []

        if not isinstance(query_embedding, list) or len(query_embedding) != 1 or \
                not isinstance(query_embedding[0], list) or len(query_embedding[0]) != self._index_dim:
            logger.error(
                f"Invalid query embedding format for collection '{collection_id}'. "
                f"Expected a list containing one list of {self._index_dim} floats. Got: {len(query_embedding[0]) if query_embedding and query_embedding[0] else 'N/A'}"
            )
            return []

        try:
            current_count = collection.count()
            logger.debug(f"Searching collection '{collection_id}' (count: {current_count}) with k={k}.")
            if current_count == 0:
                logger.debug(f"Collection '{collection_id}' is empty, no results to search.")
                return []
            effective_k = min(k, current_count) # Cannot request more results than available
            if effective_k == 0: return []

            query_results = collection.query(
                query_embeddings=query_embedding,
                n_results=effective_k,
                include=['documents', 'metadatas', 'distances'] # Request all useful info
            )
            results_list = []
            # ChromaDB query_results structure is a dict with lists for each included field
            if query_results and query_results.get('documents') and query_results.get('metadatas') and query_results.get('distances'):
                # Results are nested in a list because query_embeddings can be a list of embeddings
                for i in range(len(query_results['documents'][0])):
                    doc_content = query_results['documents'][0][i]
                    doc_metadata = query_results['metadatas'][0][i] if query_results['metadatas'][0][i] is not None else {}
                    doc_distance = query_results['distances'][0][i]
                    results_list.append({
                        'content': doc_content,
                        'metadata': doc_metadata,
                        'distance': doc_distance
                    })
            logger.debug(f"Search in '{collection_id}' returned {len(results_list)} results.")
            return results_list
        except Exception as e:
            logger.error(f"Error searching collection '{collection_id}': {e}", exc_info=True)
            return []


    def get_all_metadata(self, collection_id: str) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot get all metadata: Collection '{collection_id}' not found.")
            return []
        try:
            # Fetch all items with only metadata. `ids=None` is not standard for get.
            # A common way is to fetch a large number of items if no direct "get all metadata" exists.
            # Or, if you know the IDs or can filter for all (where={}), that's better.
            # ChromaDB's `get` can take `ids`, `where`, `where_document`.
            # If you want all, you might need to iterate or fetch with a very large limit if IDs are unknown.
            # A more efficient "get all metadata" might depend on ChromaDB version or specific client features.
            # For now, let's assume `get()` without specific filters returns all if possible, or adjust.
            # A common pattern for "all" is a where clause that's always true, but Chroma might not support that directly.
            # Let's try getting with a limit and a simple filter if "all" isn't direct.
            # A safer approach for "all" with ChromaDB is to simply fetch with include=['metadatas'] and a large limit.
            # However, if you don't specify IDs or a where clause, collection.get() returns ALL items.
            all_data = collection.get(include=['metadatas']) # This gets all items.
            return all_data.get('metadatas', []) # type: ignore
        except Exception as e:
            logger.error(f"Error retrieving all metadata from collection '{collection_id}': {e}", exc_info=True)
            return []

    def get_collection_size(self, collection_id: str) -> int:
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot get collection size: Collection '{collection_id}' not found.")
            return -1 # Indicate error or collection not found
        try:
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting count for collection '{collection_id}': {e}", exc_info=True)
            return -1

    def clear_collection(self, collection_id: str) -> bool:
        if collection_id == GLOBAL_COLLECTION_ID: # Safety net
            logger.warning("Clearing the GLOBAL_COLLECTION is a destructive operation and is typically disallowed "
                           "via this direct method for safety. Consider manual deletion or a more specific admin tool.")
            return False

        logger.warning(f"Attempting to clear collection '{collection_id}' by deleting and re-creating it.")
        try:
            if self._client:
                self._client.delete_collection(name=collection_id) # Delete it
                self.get_or_create_collection(collection_id) # Recreate it (empty)
                logger.info(f"Collection '{collection_id}' cleared and re-created successfully.")
                return True
            logger.error("Cannot clear collection: ChromaDB client not initialized.")
            return False
        except Exception as e:
            logger.error(f"Error clearing collection '{collection_id}': {e}", exc_info=True)
            # It might be partially deleted or in an inconsistent state. Try to ensure it's creatable again.
            try:
                self.get_or_create_collection(collection_id)
            except Exception as e_recreate:
                 logger.error(f"Failed to ensure collection '{collection_id}' exists after clear error: {e_recreate}")
            return False

    def get_available_collections(self) -> List[str]:
        if not self._client:
            logger.error("Cannot list collections: ChromaDB client not initialized.")
            return []
        try:
            return [coll.name for coll in self._client.list_collections()]
        except Exception as e:
            logger.error(f"Error listing collections from ChromaDB: {e}", exc_info=True)
            return []

    def delete_collection(self, collection_id: str) -> bool:
        if not self._client:
            logger.error(f"Cannot delete collection '{collection_id}': Client not initialized.")
            return False
        if collection_id == GLOBAL_COLLECTION_ID: # Safety net
            logger.warning("Deleting the GLOBAL_COLLECTION is a destructive operation and is typically disallowed "
                           "via this direct method for safety. Consider manual deletion or a more specific admin tool.")
            return False

        logger.info(f"Deleting collection '{collection_id}' from ChromaDB.")
        try:
            self._client.delete_collection(name=collection_id)
            # ChromaDB's delete_collection should handle removing associated on-disk files for PersistentClient.
            # However, if you suspect it doesn't always, you might add manual deletion of the sub-folder:
            collection_disk_path = Path(self.base_persist_directory) / collection_id
            if collection_disk_path.exists() and collection_disk_path.is_dir():
                try:
                    # Be VERY careful with rmtree.
                    # shutil.rmtree(collection_disk_path)
                    logger.info(f"PersistentClient should handle disk cleanup for {collection_id}. Manual rmtree commented out.")
                except OSError as e_rm:
                    logger.warning(f"Could not remove collection directory {collection_disk_path} from disk after DB delete: {e_rm}. "
                                   "This might be expected if ChromaDB handles it.")
            logger.info(f"Collection '{collection_id}' processing for deletion complete from DB client perspective.")
            return True
        except Exception as e: # chromadb.errors.CollectionNotFound Error is a possibility
            logger.error(f"Error deleting collection '{collection_id}': {e}", exc_info=True)
            return False