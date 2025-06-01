# app/services/file_handler_service.py
import logging
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class FileHandlerService:
    """
    Service for reading file content with support for various file types.
    """

    def __init__(self):
        """Initialize the file handler service."""
        self.supported_text_extensions = {
            '.txt', '.md', '.markdown', '.rst',
            '.py', '.js', '.ts', '.html', '.css', '.json', '.xml', '.yaml', '.yml',
            '.toml', '.ini', '.cfg', '.conf', '.env',
            '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.swift', '.php', '.rb'
        }
        # Note: PDF and DOCX require external libraries (pypdf, mammoth)
        # Ensure these are installed if you need to support these formats.
        self.supported_doc_extensions = {'.pdf', '.docx'}
        logger.info("FileHandlerService initialized")

    def read_file_content(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Read content from a file.

        Args:
            file_path: Path to the file to read

        Returns:
            Tuple of (content, file_type, error_message)
            - content: The file content as string, or None if error/binary
            - file_type: "text", "binary", or "error"
            - error_message: Error description if file_type is "error", else None
        """
        if not os.path.exists(file_path):
            return None, "error", f"File not found: {file_path}"

        if not os.path.isfile(file_path):
            return None, "error", f"Path is not a file: {file_path}"

        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return "", "text", None # Empty file is considered text with no content

            # Check file size (e.g., 50MB limit, configurable via constants if needed)
            # Using a hardcoded limit here for simplicity, but constants.RAG_MAX_FILE_SIZE_MB would be better.
            max_size = 50 * 1024 * 1024  # 50MB
            if file_size > max_size:
                return None, "error", f"File too large: {file_size / (1024 * 1024):.1f}MB > 50MB"

        except OSError as e:
            logger.error(f"Cannot access file {file_path}: {e}")
            return None, "error", f"Cannot access file: {e}"

        file_ext = os.path.splitext(file_path)[1].lower()

        # Handle different file types
        if file_ext in self.supported_text_extensions:
            return self._read_text_file(file_path)
        elif file_ext in self.supported_doc_extensions:
            return self._read_document_file(file_path, file_ext)
        else:
            # Try to read as text, but check if it's binary
            return self._read_unknown_file(file_path)

    def _read_text_file(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Read a text file with encoding detection."""
        # Try common encodings
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                return content, "text", None
            except UnicodeDecodeError:
                logger.debug(f"UnicodeDecodeError with {encoding} for {file_path}, trying next.")
                continue # Try next encoding
            except Exception as e:
                logger.error(f"Error reading text file {file_path} with {encoding}: {e}")
                continue # Try next encoding, or fail if all fail

        logger.warning(f"Could not decode file {file_path} with any supported encoding.")
        return None, "error", f"Could not decode file with any supported encoding"

    def _read_document_file(self, file_path: str, file_ext: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Read document files (PDF, DOCX, etc.). Placeholder for actual implementation."""
        try:
            if file_ext == '.pdf':
                return self._read_pdf(file_path)
            elif file_ext == '.docx':
                return self._read_docx(file_path)
            else:
                logger.warning(f"Document type {file_ext} read attempt, but not specifically handled.")
                return None, "error", f"Unsupported document type by this method: {file_ext}"
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return None, "error", f"Document read error: {e}"

    def _read_pdf(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Read PDF file content using pypdf."""
        try:
            import pypdf # Local import to avoid making it a hard dependency if not used often

            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                text_parts = []

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip(): # Ensure text is not empty
                            text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                    except Exception as e_page:
                        logger.warning(f"Error extracting text from page {page_num + 1} of {file_path}: {e_page}")
                        continue # Skip problematic pages

                if text_parts:
                    return '\n\n'.join(text_parts), "text", None
                else:
                    logger.info(f"No text content extracted from PDF: {file_path}")
                    return "", "text", "No text content extracted from PDF" # Return empty string if no text

        except ImportError:
            logger.warning("pypdf library not available. PDF reading functionality disabled. Install with: pip install pypdf")
            return None, "error", "PDF support not available (pypdf not installed)"
        except Exception as e:
            logger.error(f"Generic PDF read error for {file_path}: {e}")
            return None, "error", f"PDF read error: {e}"

    def _read_docx(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Read DOCX file content using mammoth."""
        try:
            import mammoth # Local import

            with open(file_path, 'rb') as f:
                result = mammoth.extract_raw_text(f)
                if result.value:
                    return result.value, "text", None
                else:
                    logger.info(f"No text content extracted from DOCX: {file_path}")
                    return "", "text", "No text content in DOCX file" # Return empty string

        except ImportError:
            logger.warning("mammoth library not available. DOCX reading functionality disabled. Install with: pip install mammoth")
            return None, "error", "DOCX support not available (mammoth not installed)"
        except Exception as e:
            logger.error(f"Generic DOCX read error for {file_path}: {e}")
            return None, "error", f"DOCX read error: {e}"

    def _read_unknown_file(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Try to read an unknown file type, detecting if it's binary."""
        try:
            # Read a small sample to check if it's binary
            with open(file_path, 'rb') as f:
                sample = f.read(1024) # Read first 1KB

            # Simple binary detection - if there are many null bytes or non-printable chars
            # Text files typically have very few null bytes.
            null_count = sample.count(b'\x00')
            if null_count > len(sample) * 0.1:  # More than 10% null bytes -> likely binary
                logger.info(f"Detected binary file (high null count): {file_path}")
                return None, "binary", None

            # If not obviously binary by null count, try to decode as text
            try:
                # Try UTF-8 first as it's most common
                with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                    content = f.read()
                return content, "text", None
            except UnicodeDecodeError:
                # If UTF-8 strict fails, try with replacement characters to see if it's mostly text
                logger.debug(f"UTF-8 strict decode failed for {file_path}, trying with replacement.")
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                # If too many replacement characters, it might still be binary or a very different encoding
                replacement_char_count = content.count('\ufffd')
                if replacement_char_count > len(content) * 0.05 and len(content) > 100:  # More than 5% replacement chars in a decent sized file
                    logger.info(f"Detected binary file (high replacement char count): {file_path}")
                    return None, "binary", None
                return content, "text", None # Consider it text with some encoding issues

        except Exception as e:
            logger.error(f"Unknown file read error for {file_path}: {e}")
            return None, "error", f"Unknown file read error: {e}"