# app/services/chunking_service.py
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Consider moving these defaults to utils.constants if they are widely used
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


class ChunkingService:
    """
    Service for chunking text documents into smaller pieces for RAG processing.
    Enhanced with basic Python code structure awareness.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """
        Initialize the chunking service.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if chunk_overlap >= chunk_size:
            logger.warning(
                f"Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}). Setting overlap to chunk_size / 4.")
            self.chunk_overlap = chunk_size // 4

        logger.info(
            f"ChunkingService initialized with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def chunk_document(self, content: str, source_id: str, file_ext: str = "") -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces.

        Args:
            content: The text content to chunk.
            source_id: Identifier for the source (usually file path).
            file_ext: File extension to help determine chunking strategy.

        Returns:
            List of chunk dictionaries with 'content' and 'metadata' keys.
        """
        if not content or not content.strip():
            logger.warning(f"Empty content provided for chunking: {source_id}")
            return []

        chunks: List[Dict[str, Any]] = []
        file_extension = file_ext.lower().strip('.')

        # Attempt to use langchain's text splitters if available and appropriate
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter
            LANGCHAIN_AVAILABLE = True
        except ImportError:
            LANGCHAIN_AVAILABLE = False
            logger.info("Langchain text splitters not available. Using fallback chunking methods.")

        if LANGCHAIN_AVAILABLE:
            if file_extension == 'py':
                try:
                    python_splitter = PythonCodeTextSplitter(chunk_size=self.chunk_size,
                                                             chunk_overlap=self.chunk_overlap)
                    text_chunks = python_splitter.split_text(content)
                    chunks = self._create_chunk_dicts_from_list(text_chunks, source_id,
                                                                0)  # Start line num unknown for this splitter
                    logger.debug(
                        f"Chunked Python code {source_id} using Langchain PythonCodeTextSplitter into {len(chunks)} pieces.")
                    return chunks
                except Exception as e_lc_py:
                    logger.warning(f"Langchain PythonCodeTextSplitter failed for {source_id}: {e_lc_py}. Falling back.")

            # Generic recursive splitter for other text types if Langchain is available
            try:
                # Customize separators for better general text splitting
                separators = [
                    "\n\n\n",  # Triple newlines (often separate major sections)
                    "\n\n",  # Paragraphs
                    "\n",  # Lines
                    ". ",  # Sentences
                    ", ",  # Clauses
                    " ",  # Words
                    "",  # Characters
                ]
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                    separators=separators
                )
                text_chunks = text_splitter.split_text(content)
                chunks = self._create_chunk_dicts_from_list(text_chunks, source_id, 0)  # Start line num unknown
                logger.debug(
                    f"Chunked {source_id} using Langchain RecursiveCharacterTextSplitter into {len(chunks)} pieces.")
                return chunks
            except Exception as e_lc_rec:
                logger.warning(
                    f"Langchain RecursiveCharacterTextSplitter failed for {source_id}: {e_lc_rec}. Falling back to manual.")

        # Fallback to manual chunking if Langchain is not available or failed
        if file_extension == 'py':
            chunks = self._chunk_python_code_manually(content, source_id)
        else:
            chunks = self._chunk_text_manually(content, source_id)

        logger.debug(f"Manually chunked {source_id} into {len(chunks)} pieces (fallback).")
        return chunks

    def _create_chunk_dicts_from_list(self, text_chunks: List[str], source_id: str, base_start_line: int) -> List[
        Dict[str, Any]]:
        """Helper to create chunk dictionaries from a list of text strings."""
        chunks_dicts = []
        current_line_offset = 0
        for i, text_chunk in enumerate(text_chunks):
            if not text_chunk.strip():
                continue
            num_lines_in_chunk = text_chunk.count('\n')
            start_line = base_start_line + current_line_offset + 1
            end_line = start_line + num_lines_in_chunk
            chunk_dict = {
                'content': text_chunk,
                'metadata': {
                    'source': source_id,
                    'chunk_index': i,
                    'start_line': start_line,
                    'end_line': end_line,
                    'filename': os.path.basename(source_id)
                }
            }
            chunks_dicts.append(chunk_dict)
            current_line_offset += (num_lines_in_chunk + 1) if num_lines_in_chunk > 0 else 1
        return chunks_dicts

    def _chunk_python_code_manually(self, content: str, source_id: str) -> List[Dict[str, Any]]:
        """
        Manually chunk Python code, trying to preserve logical boundaries like functions/classes.
        This is a fallback if Langchain's PythonCodeTextSplitter is unavailable or fails.
        """
        lines = content.splitlines()
        chunks = []
        current_chunk_lines: List[str] = []
        current_char_count = 0
        chunk_start_line_num = 1

        for i, line_text in enumerate(lines):
            line_num = i + 1
            line_len = len(line_text) + 1  # +1 for newline character

            # Potential split points: start of a class or function, or a blank line separating blocks
            is_potential_split_point = line_text.strip().startswith(('def ', 'class ')) or not line_text.strip()

            if current_char_count + line_len > self.chunk_size and current_chunk_lines:
                # If adding this line exceeds chunk_size, and we have content in current_chunk_lines:
                # If current line is a good split point, or if chunk is already quite full, finalize current chunk.
                if is_potential_split_point or current_char_count > self.chunk_size - self.chunk_overlap:
                    chunk_content = "\n".join(current_chunk_lines)
                    chunks.append({
                        'content': chunk_content,
                        'metadata': {
                            'source': source_id,
                            'chunk_index': len(chunks),
                            'start_line': chunk_start_line_num,
                            'end_line': line_num - 1,
                            'filename': os.path.basename(source_id)
                        }
                    })
                    # Start new chunk, considering overlap
                    # Find how many lines to carry over for overlap
                    overlap_chars = 0
                    overlap_line_idx_start = len(current_chunk_lines)
                    for k in range(len(current_chunk_lines) - 1, -1, -1):
                        overlap_chars += len(current_chunk_lines[k]) + 1
                        if overlap_chars >= self.chunk_overlap:
                            overlap_line_idx_start = k
                            break

                    current_chunk_lines = current_chunk_lines[overlap_line_idx_start:]
                    current_char_count = sum(len(l) + 1 for l in current_chunk_lines)
                    chunk_start_line_num = chunk_start_line_num + overlap_line_idx_start

            current_chunk_lines.append(line_text)
            current_char_count += line_len

        # Add any remaining lines as the final chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            if chunk_content.strip():
                chunks.append({
                    'content': chunk_content,
                    'metadata': {
                        'source': source_id,
                        'chunk_index': len(chunks),
                        'start_line': chunk_start_line_num,
                        'end_line': len(lines),
                        'filename': os.path.basename(source_id)
                    }
                })
        return chunks

    def _chunk_text_manually(self, content: str, source_id: str) -> List[Dict[str, Any]]:
        """
        Manually chunk regular text content by characters, trying to respect sentence/paragraph boundaries.
        This is a fallback if Langchain's RecursiveCharacterTextSplitter is unavailable or fails.
        """
        chunks = []
        start_char_idx = 0
        chunk_idx_counter = 0
        content_len = len(content)
        current_line_num_abs = 1  # Absolute line number in the original document

        while start_char_idx < content_len:
            end_char_idx = min(start_char_idx + self.chunk_size, content_len)

            # If not the last chunk, try to find a better split point backward from end_char_idx
            if end_char_idx < content_len:
                # Prefer double newlines, then single, then sentence endings, then spaces
                potential_split_points = []
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:  # Order matters
                    sep_idx = content.rfind(sep, start_char_idx, end_char_idx)
                    if sep_idx != -1 and sep_idx > start_char_idx + (
                            self.chunk_size - self.chunk_overlap):  # Ensure split is within overlap region
                        potential_split_points.append(sep_idx + len(sep))

                if potential_split_points:
                    end_char_idx = max(potential_split_points)  # Take the latest good split point
                else:  # If no good separator found, try space
                    space_idx = content.rfind(" ", start_char_idx, end_char_idx)
                    if space_idx != -1 and space_idx > start_char_idx + (self.chunk_size - self.chunk_overlap):
                        end_char_idx = space_idx + 1

            chunk_text = content[start_char_idx:end_char_idx]

            if chunk_text.strip():
                num_lines_in_chunk = chunk_text.count('\n')
                chunk_dict = {
                    'content': chunk_text,
                    'metadata': {
                        'source': source_id,
                        'chunk_index': chunk_idx_counter,
                        'start_line': current_line_num_abs,
                        'end_line': current_line_num_abs + num_lines_in_chunk,
                        'filename': os.path.basename(source_id)
                    }
                }
                chunks.append(chunk_dict)
                chunk_idx_counter += 1

            current_line_num_abs += (num_lines_in_chunk + 1) if chunk_text.strip() else 0

            # Determine start of next chunk with overlap
            start_char_idx = end_char_idx - self.chunk_overlap
            if start_char_idx < 0: start_char_idx = 0  # Should not happen if overlap < chunk_size
            if start_char_idx >= content_len: break  # Reached end

            # Adjust current_line_num_abs for the overlapped part for the next chunk's start_line
            if chunk_text.strip():  # Only adjust if current chunk was not empty
                overlap_text_for_next_start = content[
                                              start_char_idx:end_char_idx]  # Text that will start the next chunk due to overlap
                lines_in_overlap_for_next_start = overlap_text_for_next_start.count('\n')
                current_line_num_abs -= lines_in_overlap_for_next_start

        return chunks