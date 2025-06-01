# utils/code_output_processor.py
import ast  # For validating Python syntax
import logging
import re
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict, Sized  # Added Union

logger = logging.getLogger(__name__)


class CodeExtractionStrategy(Enum):
    FENCED_BLOCK = auto()
    MARKED_SECTION = auto()
    LARGEST_BLOCK = auto()
    FULL_RESPONSE = auto()
    HYBRID_EXTRACTION = auto()


class CodeQualityLevel(Enum):
    EXCELLENT = auto()  # Perfect syntax, good structure, comments, docs
    GOOD = auto()  # Valid syntax, decent structure
    ACCEPTABLE = auto()  # Valid syntax, but basic or messy
    POOR = auto()  # Syntax errors present but potentially recoverable
    UNUSABLE = auto()  # Too many issues, or not identifiable as code


class CodeOutputProcessor:
    """
    Enhanced processor for extracting, validating, and cleaning code from LLM responses.
    Aims for a high success rate by using multiple strategies and robust fallbacks.
    """

    def __init__(self):
        self.logger = logger.getChild('CodeProcessor')  # Using a child logger

        # Common fenced block patterns, ordered by strictness/commonality
        self._fenced_patterns = [
            re.compile(r"```(?:python|py)\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # Standard Python
            re.compile(r"```(?:javascript|js)\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # JavaScript
            re.compile(r"```(?:typescript|ts)\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # TypeScript
            re.compile(r"```java\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # Java
            re.compile(r"```(?:csharp|cs)\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # C#
            re.compile(r"```html\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # HTML
            re.compile(r"```css\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # CSS
            re.compile(r"```json\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # JSON
            re.compile(r"```(?:yaml|yml)\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # YAML
            re.compile(r"```xml\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # XML
            re.compile(r"```sql\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # SQL
            re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # Shell/Bash
            re.compile(r"```\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),  # Generic, no language specified
            # Variations LLMs might use
            re.compile(r"~~~(?:python|py)?\s*\n(.*?)\n?\s*~~~", re.DOTALL | re.IGNORECASE),  # Tilde fences
            re.compile(r"```(?:python|py)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE),  # No newline after lang
            re.compile(r"`{3,}(?:python|py)?\s*\n?(.*?)\n?`{3,}", re.DOTALL | re.IGNORECASE),  # 3+ backticks
        ]

        # Patterns for text that often precedes or follows code blocks, to be cleaned
        self._cleanup_patterns = [
            re.compile(
                r"^Here'?s?\s+(?:the\s+)?(?:complete\s+)?(?:updated\s+)?(?:python\s+|[\w\s]+)?(?:code|implementation|solution|script|file).*?[:]\s*\n?",
                re.IGNORECASE | re.MULTILINE),
            re.compile(
                r"^(?:The\s+)?(?:complete\s+)?(?:updated\s+)?(?:Python\s+|[\w\s]+)?(?:code\s+)?(?:implementation\s+)?(?:for\s+.*?\s+)?(?:is|would\s+be|looks\s+like).*?[:]\s*\n?",
                re.IGNORECASE | re.MULTILINE),
            re.compile(r"^I'?(?:ll|ve|d)\s+(?:create|write|implement|update|provide|give\s+you).*?[:]\s*\n?",
                       re.IGNORECASE | re.MULTILINE),
            re.compile(r"\n\s*(?:This|That)\s+(?:code|implementation|solution|script|file).*?(?:\.|!|\n|$)",
                       re.IGNORECASE | re.MULTILINE | re.DOTALL),
            re.compile(r"\n\s*(?:Hope|Let\s+me\s+know|Feel\s+free|Enjoy|Happy\s+coding).*?(?:\.|!|\n|$)",
                       re.IGNORECASE | re.MULTILINE | re.DOTALL),
            re.compile(r"\n\s*(?:This\s+should|This\s+will|This\s+implementation).*?(?:\.|!|\n|$)",
                       re.IGNORECASE | re.MULTILINE | re.DOTALL),
            re.compile(
                r"^(?:You\s+can\s+save\s+this\s+as|Save\s+the\s+following\s+content\s+to).*?['\"]?([\w\-./]+)['\"]?.*?\n",
                re.IGNORECASE | re.MULTILINE),
            re.compile(r"^(?:Note|Important|Remember|Explanation|Key\s+points|Summary)[\s:]*.*?\n",
                       re.IGNORECASE | re.MULTILINE),
        ]

        # Python-specific indicators for validation and intelligent block detection
        self._python_keywords = {
            'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif',
            'for', 'while', 'try', 'except', 'with', 'as', 'in', 'is', 'and',
            'or', 'not', 'lambda', 'yield', 'async', 'await', 'pass', 'break',
            'continue', 'global', 'nonlocal', 'assert', 'del', 'raise', 'finally'
        }
        self._python_patterns = [
            re.compile(r'^\s*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:', re.MULTILINE),
            re.compile(r'^\s*class\s+[a-zA-Z_][a-zA-Z0-9_]*.*?:', re.MULTILINE),
            re.compile(r'^\s*import\s+[a-zA-Z_][a-zA-Z0-9_.,\s]*', re.MULTILINE),
            re.compile(r'^\s*from\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import', re.MULTILINE),
            re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']', re.MULTILINE),
            re.compile(r'^\s*@\w+', re.MULTILINE),  # Decorators
        ]
        self._validation_cache: Dict[Tuple[int, int], bool] = {}  # (len(code), hash(code_sample)) -> is_valid
        self._max_cache_size = 100  # Max items in validation cache
        logger.info("CodeOutputProcessor initialized with enhanced strategies.")

    def process_llm_response(self, raw_response: str, filename: str,
                             expected_language: str = "python") -> Tuple[Optional[str], CodeQualityLevel, List[str]]:
        """
        Main entry point to extract, clean, and validate code from an LLM response.
        """
        processing_notes: List[str] = []
        if not raw_response or not raw_response.strip():
            return None, CodeQualityLevel.UNUSABLE, ["Empty LLM response"]

        original_length = len(raw_response)
        processing_notes.append(f"Original response length: {original_length} chars.")

        # Strategy 1: Direct fenced block extraction (most common and reliable)
        extracted_code = self._extract_fenced_code_blocks(raw_response, expected_language)
        if extracted_code:
            cleaned_code = self.clean_and_format_code(extracted_code)
            is_valid, syntax_error = self._is_valid_python_code(cleaned_code)
            if is_valid:
                quality = self._assess_code_quality(cleaned_code, filename, is_python=True)
                processing_notes.append("Extracted from primary fenced block.")
                return cleaned_code, quality, processing_notes
            else:
                processing_notes.append(f"Primary fenced block invalid: {syntax_error}")

        # Strategy 2: Clean entire response then try fenced extraction again
        cleaned_full_response = self._perform_aggressive_cleanup(raw_response)
        if cleaned_full_response != raw_response:
            processing_notes.append("Aggressive cleanup applied to full response.")
            extracted_code_after_cleanup = self._extract_fenced_code_blocks(cleaned_full_response, expected_language)
            if extracted_code_after_cleanup:
                cleaned_code = self.clean_and_format_code(extracted_code_after_cleanup)
                is_valid, syntax_error = self._is_valid_python_code(cleaned_code)
                if is_valid:
                    quality = self._assess_code_quality(cleaned_code, filename, is_python=True)
                    processing_notes.append("Extracted fenced block after full response cleanup.")
                    return cleaned_code, quality, processing_notes
                else:
                    processing_notes.append(f"Fenced block after cleanup invalid: {syntax_error}")

        # Strategy 3: Intelligent block detection (if Python expected)
        if expected_language.lower() == "python":
            intelligent_block = self._extract_python_block_intelligently(cleaned_full_response)  # Use cleaned response
            if intelligent_block:
                cleaned_code = self.clean_and_format_code(intelligent_block)
                is_valid, syntax_error = self._is_valid_python_code(cleaned_code)
                if is_valid:
                    quality = self._assess_code_quality(cleaned_code, filename, is_python=True)
                    processing_notes.append("Extracted via intelligent Python block detection.")
                    return cleaned_code, quality, processing_notes
                else:
                    processing_notes.append(f"Intelligent block invalid: {syntax_error}")

        # Strategy 4: Last resort - treat the cleaned full response as code (if it looks like code)
        # This is risky but can sometimes salvage code if fences are completely missing or malformed.
        if self._looks_like_code(cleaned_full_response, expected_language):
            is_valid_full, syntax_error_full = self._is_valid_python_code(
                cleaned_full_response)  # Assuming python for now
            if is_valid_full:
                quality = self._assess_code_quality(cleaned_full_response, filename, is_python=True)
                processing_notes.append("Used cleaned full response as code (passed validation).")
                return cleaned_full_response, quality, processing_notes
            else:
                processing_notes.append(f"Cleaned full response invalid as code: {syntax_error_full}")

        # If all strategies fail
        processing_notes.append("All code extraction strategies failed.")
        logger.warning(f"Failed to extract valid code for '{filename}'. Notes: {processing_notes}")
        return None, CodeQualityLevel.UNUSABLE, processing_notes

    def _extract_fenced_code_blocks(self, text: str, expected_language: str) -> Optional[str]:
        """Extracts code from the first, or most prominent, fenced code block."""
        candidates: List[Tuple[str, int]] = []  # (code, score)
        for pattern in self._fenced_patterns:
            for match in pattern.finditer(text):
                lang_in_fence = ""
                # Check if the pattern captures language (depends on regex group structure)
                if len(match.groups()) > 1 and match.group(1) and not match.group(
                        1).isspace():  # First group might be lang
                    lang_in_fence = match.group(1).lower().strip()
                    code_content = match.group(2).strip() if len(match.groups()) > 1 else match.group(1).strip()
                else:  # Simpler regex with only one capturing group for content
                    code_content = match.group(1).strip()

                if not code_content: continue

                score = len(code_content)  # Base score on length
                # Boost score if language in fence matches expected language
                if lang_in_fence and expected_language and lang_in_fence == expected_language.lower():
                    score += 1000  # High boost for matching language
                elif lang_in_fence:  # Penalize if language is specified but doesn't match
                    score -= 500

                # If no language in fence, check content for Python keywords
                elif not lang_in_fence and expected_language.lower() == "python":
                    if any(keyword in code_content for keyword in self._python_keywords):
                        score += 200  # Moderate boost if Python keywords found

                candidates.append((code_content, score))

        if not candidates: return None
        candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by score desc
        return candidates[0][0]  # Return the highest scored candidate

    def _perform_aggressive_cleanup(self, text: str) -> str:
        """Applies multiple cleanup patterns to remove explanatory text."""
        cleaned_text = text
        for pattern in self._cleanup_patterns:
            cleaned_text = pattern.sub('', cleaned_text)

        # Additional common cleanup: remove lines that are just "python", "code:", etc.
        lines = cleaned_text.split('\n')
        filtered_lines = [
            line for line in lines
            if not (line.strip().lower() in ["python", "code", "code:", "python:", "output:", "result:", "example:"] or
                    line.strip().endswith(":") and len(line.strip()) < 20)  # Remove short lines ending with colon
        ]
        cleaned_text = '\n'.join(filtered_lines)

        return cleaned_text.strip()

    def _extract_python_block_intelligently(self, text: str) -> Sized | None:
        """Tries to find a coherent block of Python code without fences."""
        lines = text.split('\n')
        potential_blocks: List[List[str]] = []
        current_block: List[str] = []
        in_code_block = False
        indent_level = 0

        for line in lines:
            stripped_line = line.strip()
            current_indent = len(line) - len(line.lstrip())

            is_python_indicator = any(
                stripped_line.startswith(kw) for kw in ['def ', 'class ', '@', 'import ', 'from ']) or \
                                  any(pattern.match(line) for pattern in self._python_patterns)

            if is_python_indicator and not in_code_block:
                if current_block:  # Save previous block if any
                    potential_blocks.append(list(current_block))
                current_block = [line]
                in_code_block = True
                indent_level = current_indent
            elif in_code_block:
                # Continue block if line is indented, empty, a comment, or contains keywords
                if stripped_line == "" or \
                        stripped_line.startswith('#') or \
                        current_indent >= indent_level or \
                        any(kw in stripped_line for kw in self._python_keywords):
                    current_block.append(line)
                else:  # Likely end of block
                    if current_block: potential_blocks.append(list(current_block))
                    current_block = []
                    in_code_block = False
                    indent_level = 0
            # If not in a code block and line looks like a start, start a new one
            elif is_python_indicator:
                if current_block: potential_blocks.append(list(current_block))
                current_block = [line]
                in_code_block = True
                indent_level = current_indent

        if current_block:  # Add the last collected block
            potential_blocks.append(current_block)

        if not potential_blocks: return None

        # Filter blocks: must have min lines and look like Python code
        valid_blocks = []
        for block_lines in potential_blocks:
            if len(block_lines) >= 3:  # Minimum 3 lines for a meaningful block
                block_text = "\n".join(block_lines)
                if self._looks_like_code(block_text, "python"):
                    is_valid, _ = self._is_valid_python_code(block_text)
                    if is_valid:
                        valid_blocks.append(block_text)

        if not valid_blocks: return None

        # Return the largest valid block
        return max(valid_blocks, key=len)

    def _looks_like_code(self, text: str, language: str) -> bool:
        """Basic heuristic to check if a string looks like code of a given language."""
        if not text.strip(): return False
        if language.lower() == "python":
            # Check for significant Python keywords or syntax elements
            num_keywords = sum(1 for kw in self._python_keywords if kw in text)
            num_patterns = sum(1 for pattern in self._python_patterns if pattern.search(text))
            # If many lines start with typical Python indentation or keywords
            python_lines = 0
            total_lines = 0
            for line in text.split('\n'):
                if line.strip():
                    total_lines += 1
                    if line.strip().startswith(tuple(self._python_keywords)) or \
                            line.startswith(('    ', '\t')) or \
                            any(p.match(line.strip()) for p in self._python_patterns):
                        python_lines += 1

            if total_lines == 0: return False
            # Heuristic: if >30% lines look like Python, or high keyword/pattern count
            return (python_lines / total_lines > 0.3) or (num_keywords > 5) or (num_patterns > 2)
        # Add heuristics for other languages if needed
        return False  # Default for unhandled languages

    def _is_valid_python_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validates Python code using AST parsing. Returns (isValid, error_message)."""
        if not code or not code.strip():
            return False, "Empty code"

        # Cache key based on code length and a sample hash to avoid hashing huge strings
        code_sample_for_hash = code[:100] + code[-100:] if len(code) > 200 else code
        cache_key = (len(code), hash(code_sample_for_hash))
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key], None  # Assume cached implies no new error msg

        try:
            ast.parse(code)
            self._update_validation_cache(cache_key, True)
            return True, None
        except SyntaxError as e:
            error_detail = f"SyntaxError: {e.msg} (line {e.lineno}, offset {e.offset})"
            self._update_validation_cache(cache_key, False)
            return False, error_detail
        except Exception as e_other:  # Catch other potential parsing errors
            error_detail = f"AST Parsing Error: {type(e_other).__name__} - {e_other}"
            self._update_validation_cache(cache_key, False)
            return False, error_detail

    def _update_validation_cache(self, key: Tuple[int, int], value: bool):
        if len(self._validation_cache) >= self._max_cache_size:
            # Simple FIFO eviction
            self._validation_cache.pop(next(iter(self._validation_cache)))
        self._validation_cache[key] = value

    def _assess_code_quality(self, code: str, filename: str, is_python: bool = True) -> CodeQualityLevel:
        """Assesses the quality of the extracted code."""
        if not code.strip(): return CodeQualityLevel.UNUSABLE

        if is_python:
            is_valid, syntax_error_msg = self._is_valid_python_code(code)
            if not is_valid:
                # If there's a syntax error, try to determine if it's minor or major
                if syntax_error_msg and ("unexpected EOF" in syntax_error_msg or "indent" in syntax_error_msg.lower()):
                    return CodeQualityLevel.POOR  # Often recoverable
                return CodeQualityLevel.UNUSABLE

            # Basic checks for Python quality beyond just syntax
            has_docstrings = "def " in code and '"""' in code or "class " in code and '"""' in code
            has_comments = "#" in code
            line_count = len(code.split('\n'))

            # Simple heuristic for quality
            if line_count < 5 and not (has_docstrings or has_comments):
                return CodeQualityLevel.ACCEPTABLE  # Too short to be "good" without context
            if has_docstrings and (has_comments or line_count > 20):
                return CodeQualityLevel.EXCELLENT
            if has_docstrings or has_comments or line_count > 10:
                return CodeQualityLevel.GOOD
            return CodeQualityLevel.ACCEPTABLE
        else:
            # For non-Python, a simpler check: if it's not empty, assume acceptable for now
            return CodeQualityLevel.ACCEPTABLE

    def clean_and_format_code(self, code: str) -> str:
        """Removes leading/trailing empty lines and extraneous whitespace from code block."""
        if not code: return ""

        lines = code.split('\n')

        # Remove leading and trailing blank lines
        start_index = 0
        while start_index < len(lines) and not lines[start_index].strip():
            start_index += 1

        end_index = len(lines) - 1
        while end_index >= 0 and not lines[end_index].strip():
            end_index -= 1

        if start_index > end_index: return ""  # All lines were blank

        cleaned_lines = lines[start_index: end_index + 1]

        # Optionally, attempt to unindent common leading whitespace if it's excessive
        # This is a simple heuristic and might not be perfect for all cases.
        if cleaned_lines:
            # Find minimum leading whitespace in non-empty lines
            min_indent = float('inf')
            for line in cleaned_lines:
                if line.strip():  # Only consider non-empty lines
                    leading_space = len(line) - len(line.lstrip(' '))
                    leading_tabs_as_space = (len(line) - len(line.lstrip('\t'))) * 4  # Assuming tab = 4 spaces
                    current_line_indent = min(leading_space, leading_tabs_as_space)  # Use the smaller if mixed
                    min_indent = min(min_indent, current_line_indent)

            if min_indent != float('inf') and min_indent > 0:
                # Unindent lines by the minimum common indent
                # This is tricky because it assumes consistent indentation was intended.
                # A more robust solution would use an AST formatter like Black if possible,
                # but that's a heavier dependency.
                # For now, a simpler unindent if all lines share a common prefix.

                # Check if all non-empty lines share this common prefix
                # We need to be careful if tabs and spaces are mixed.
                # This simple common prefix check might not be robust enough.
                # Let's refine to check if *all* lines can be unindented by this amount.

                can_unindent_all = True
                temp_unindented_lines = []

                for line_idx, line in enumerate(cleaned_lines):
                    if line.strip():  # Only operate on non-empty lines for unindenting
                        # Try to remove the min_indent amount of spaces
                        if line.startswith(" " * min_indent):
                            temp_unindented_lines.append(line[min_indent:])
                        # Else, if tabs were the source of min_indent, this logic is too simple.
                        # For now, if it doesn't start with that many spaces, we can't uniformly unindent.
                        # A more sophisticated approach would convert all leading tabs to spaces first.
                        else:  # Fallback: if a line doesn't have that prefix, don't unindent the block
                            can_unindent_all = False
                            break
                    else:  # Keep empty lines as they are
                        temp_unindented_lines.append(line)

                if can_unindent_all:
                    cleaned_lines = temp_unindented_lines

        return '\n'.join(cleaned_lines)

    def suggest_fixes_for_poor_code(self, raw_response_text: str, filename: str) -> List[str]:
        """Provides suggestions if code extraction results in poor quality or unusable code."""
        suggestions: List[str] = []

        if "```" not in raw_response_text:
            suggestions.append("The LLM response might be missing code fences (e.g., ```python ... ```).")
        elif raw_response_text.count("```") % 2 != 0:
            suggestions.append("The code fences in the LLM response seem mismatched or incomplete.")

        # Try to parse the raw response (or cleaned version) to see if it contains syntax errors
        cleaned_for_syntax_check = self._perform_aggressive_cleanup(raw_response_text)
        is_valid, syntax_error = self._is_valid_python_code(cleaned_for_syntax_check)
        if not is_valid and syntax_error:
            suggestions.append(f"The response contains syntax errors: {syntax_error}. "
                               "The LLM might need a clearer prompt or to be asked to fix the errors.")

        if len(raw_response_text.strip()) < 50 and "```" in raw_response_text:  # Very short but fenced
            suggestions.append(
                "The extracted code block is very short. The LLM might have truncated its response or the prompt was too narrow.")

        if not suggestions:  # Generic advice if no specific issue is pinpointed
            suggestions.append("Could not reliably extract code. Consider the following:")
            suggestions.append("- Check LLM's raw output for formatting issues.")
            suggestions.append("- Simplify the prompt to the LLM.")
            suggestions.append("- Ask the LLM to be very specific about using code fences.")

        return suggestions[:5]  # Limit to a few useful suggestions