# app/core/code_output_processor.py
import ast
import re
import logging
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum, auto
import autopep8  # type: ignore

logger = logging.getLogger(__name__)


class CodeQualityLevel(Enum):
    """Enumeration for code quality assessment"""
    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    NEEDS_IMPROVEMENT = auto()
    POOR = auto()


class CodeOutputProcessor:
    """
    Enhanced processor for LLM-generated code with intelligent analysis,
    validation, and formatting capabilities.
    """

    def __init__(self):
        self._code_block_patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```py\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'(?:^|\n)```python\s*\n(.*?)```(?:\s*$|\n)',
            r'(?:^|\n)```\s*\n(.*?)```(?:\s*$|\n)'
        ]
        logger.info("Enhanced CodeOutputProcessor initialized")

    def process_llm_response(self,
                             llm_response: str,
                             expected_filename: str,
                             expected_language: str = "python") -> Tuple[
        Optional[str], Optional[CodeQualityLevel], List[str]]:
        """
        Process LLM response and extract high-quality code.

        Args:
            llm_response: Raw response from LLM
            expected_filename: Expected filename for context
            expected_language: Expected programming language

        Returns:
            Tuple of (extracted_code, quality_level, processing_notes)
        """
        processing_notes = []

        try:
            # Step 1: Extract code from response
            extracted_code = self._extract_code_block(llm_response)
            if not extracted_code:
                processing_notes.append("No code block found in LLM response")
                return None, None, processing_notes

            processing_notes.append(f"Successfully extracted {len(extracted_code)} characters of code")

            # Step 2: Clean and normalize the code
            cleaned_code = self._clean_extracted_code(extracted_code)
            processing_notes.append("Code cleaned and normalized")

            # Step 3: Validate syntax
            is_valid, syntax_error = self._is_valid_python_code(cleaned_code)
            if not is_valid:
                processing_notes.append(f"Syntax validation failed: {syntax_error}")
                return cleaned_code, CodeQualityLevel.POOR, processing_notes

            processing_notes.append("Syntax validation passed")

            # Step 4: Assess code quality
            quality_level = self._assess_code_quality(cleaned_code, expected_filename)
            processing_notes.append(f"Quality assessment: {quality_level.name}")

            # Step 5: Apply intelligent formatting
            formatted_code = self._apply_intelligent_formatting(cleaned_code)
            processing_notes.append("Intelligent formatting applied")

            return formatted_code, quality_level, processing_notes

        except Exception as e:
            logger.error(f"Error processing LLM response: {e}", exc_info=True)
            processing_notes.append(f"Processing error: {e}")
            return None, None, processing_notes

    def _extract_code_block(self, text: str) -> Optional[str]:
        """Extract code from various markdown code block formats"""
        for pattern in self._code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the longest match (most likely to be the complete code)
                return max(matches, key=len).strip()

        # Fallback: look for Python-like content without markdown
        if self._looks_like_python_code(text):
            return text.strip()

        return None

    def _looks_like_python_code(self, text: str) -> bool:
        """Heuristic to determine if text looks like Python code"""
        python_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if __name__',
            'try:', 'except:', 'with ', 'for ', 'while ', 'return ',
            'yield ', 'async def', 'await '
        ]

        lines = text.split('\n')
        code_like_lines = 0

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if any(indicator in stripped for indicator in python_indicators):
                code_like_lines += 1
            elif ':' in stripped and (stripped.endswith(':') or 'def ' in stripped or 'class ' in stripped):
                code_like_lines += 1

        return code_like_lines >= 2  # At least 2 Python-like lines

    def _clean_extracted_code(self, code: str) -> str:
        """Clean and normalize extracted code"""
        # Remove common LLM artifacts
        cleaned = code

        # Remove explanatory comments at the start that aren't part of the code
        lines = cleaned.split('\n')
        start_index = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('#') and
                    any(phrase in stripped.lower() for phrase in ['here is', 'here\'s', 'this code', 'the following'])):
                start_index = i + 1
            elif stripped and not stripped.startswith('#'):
                break

        if start_index > 0:
            cleaned = '\n'.join(lines[start_index:])

        # Remove trailing explanatory text
        lines = cleaned.split('\n')
        end_index = len(lines)

        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped and not stripped.startswith('#'):
                break
            if (stripped.startswith('#') and
                    any(phrase in stripped.lower() for phrase in ['this code', 'the above', 'explanation'])):
                end_index = i

        if end_index < len(lines):
            cleaned = '\n'.join(lines[:end_index])

        # Normalize whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
        cleaned = cleaned.strip()

        return cleaned

    def _is_valid_python_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def _assess_code_quality(self, code: str, filename: str) -> CodeQualityLevel:
        """Assess the quality of the generated code"""
        try:
            tree = ast.parse(code)
            quality_score = 0
            total_checks = 0

            # Check for docstrings
            if self._has_module_docstring(tree):
                quality_score += 1
            total_checks += 1

            # Check for type hints
            type_hint_ratio = self._calculate_type_hint_ratio(tree)
            if type_hint_ratio > 0.8:
                quality_score += 2
            elif type_hint_ratio > 0.5:
                quality_score += 1
            total_checks += 2

            # Check for error handling
            if self._has_error_handling(tree):
                quality_score += 1
            total_checks += 1

            # Check for logging
            if self._has_logging(code):
                quality_score += 1
            total_checks += 1

            # Check for function docstrings
            docstring_ratio = self._calculate_docstring_ratio(tree)
            if docstring_ratio > 0.8:
                quality_score += 2
            elif docstring_ratio > 0.5:
                quality_score += 1
            total_checks += 2

            # Check for meaningful names
            if self._has_meaningful_names(tree):
                quality_score += 1
            total_checks += 1

            # Calculate final quality
            quality_percentage = quality_score / total_checks if total_checks > 0 else 0

            if quality_percentage >= 0.9:
                return CodeQualityLevel.EXCELLENT
            elif quality_percentage >= 0.7:
                return CodeQualityLevel.GOOD
            elif quality_percentage >= 0.5:
                return CodeQualityLevel.ACCEPTABLE
            elif quality_percentage >= 0.3:
                return CodeQualityLevel.NEEDS_IMPROVEMENT
            else:
                return CodeQualityLevel.POOR

        except Exception as e:
            logger.warning(f"Error assessing code quality: {e}")
            return CodeQualityLevel.ACCEPTABLE

    def _has_module_docstring(self, tree: ast.AST) -> bool:
        """Check if module has a docstring"""
        if (isinstance(tree, ast.Module) and
                tree.body and
                isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str)):
            return True
        return False

    def _calculate_type_hint_ratio(self, tree: ast.AST) -> float:
        """Calculate ratio of functions with type hints"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)

        if not functions:
            return 1.0  # No functions to check

        typed_functions = 0
        for func in functions:
            has_return_hint = func.returns is not None
            has_arg_hints = any(arg.annotation is not None for arg in func.args.args)

            if has_return_hint or has_arg_hints:
                typed_functions += 1

        return typed_functions / len(functions)

    def _has_error_handling(self, tree: ast.AST) -> bool:
        """Check if code has error handling"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                return True
        return False

    def _has_logging(self, code: str) -> bool:
        """Check if code includes logging"""
        logging_indicators = ['import logging', 'logger.', 'logging.']
        return any(indicator in code for indicator in logging_indicators)

    def _calculate_docstring_ratio(self, tree: ast.AST) -> float:
        """Calculate ratio of functions/classes with docstrings"""
        items = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                items.append(node)

        if not items:
            return 1.0  # No items to check

        documented_items = 0
        for item in items:
            if (item.body and
                    isinstance(item.body[0], ast.Expr) and
                    isinstance(item.body[0].value, ast.Constant) and
                    isinstance(item.body[0].value.value, str)):
                documented_items += 1

        return documented_items / len(items)

    def _has_meaningful_names(self, tree: ast.AST) -> bool:
        """Check for meaningful variable/function names"""
        poor_names = {'a', 'b', 'c', 'x', 'y', 'z', 'temp', 'var', 'data', 'info'}
        total_names = 0
        poor_name_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                total_names += 1
                if node.id.lower() in poor_names:
                    poor_name_count += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_names += 1
                if node.name.lower() in poor_names:
                    poor_name_count += 1

        if total_names == 0:
            return True

        return (poor_name_count / total_names) < 0.2  # Less than 20% poor names

    def _apply_intelligent_formatting(self, code: str) -> str:
        """Apply intelligent formatting to the code"""
        try:
            # Use autopep8 for basic PEP 8 formatting
            formatted = autopep8.fix_code(code, options={
                'max_line_length': 88,  # Black's default
                'aggressive': 2,
                'experimental': True
            })

            # Additional intelligent formatting
            formatted = self._apply_custom_formatting(formatted)

            return formatted
        except Exception as e:
            logger.warning(f"Error applying formatting: {e}")
            return code  # Return original if formatting fails

    def _apply_custom_formatting(self, code: str) -> str:
        """Apply custom formatting rules"""
        lines = code.split('\n')
        formatted_lines = []

        for line in lines:
            # Ensure proper spacing around operators
            line = re.sub(r'([^=!<>])=([^=])', r'\1 = \2', line)
            line = re.sub(r'([^=!<>])==([^=])', r'\1 == \2', line)

            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def clean_and_format_code(self, code: str) -> str:
        """Clean and format code for final output"""
        try:
            # Apply intelligent formatting
            formatted = self._apply_intelligent_formatting(code)

            # Ensure file ends with single newline
            formatted = formatted.rstrip() + '\n'

            return formatted
        except Exception as e:
            logger.warning(f"Error in final cleanup: {e}")
            return code

    def extract_imports_from_code(self, code: str) -> List[str]:
        """Extract import statements from code"""
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    if node.level > 0:  # Relative import
                        module = "." * node.level + module
                    imports.append(f"from {module} import {', '.join(names)}")
        except Exception as e:
            logger.warning(f"Error extracting imports: {e}")

        return imports

    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze the structure of the code"""
        structure = {
            'classes': [],
            'functions': [],
            'imports': [],
            'constants': [],
            'lines_of_code': len([line for line in code.split('\n') if line.strip()]),
            'complexity_score': 1
        }

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure['classes'].append({
                        'name': node.name,
                        'methods': [n.name for n in node.body if
                                    isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
                        'line': node.lineno
                    })
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Only top-level functions (not methods)
                    if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                               if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                        structure['functions'].append({
                            'name': node.name,
                            'line': node.lineno,
                            'is_async': isinstance(node, ast.AsyncFunctionDef)
                        })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Handled by extract_imports_from_code
                    pass
                elif isinstance(node, ast.Assign):
                    # Look for module-level constants
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            structure['constants'].append(target.id)

            # Calculate complexity score
            structure['complexity_score'] = min(
                1 + len(structure['classes']) + len(structure['functions']) // 3,
                10
            )

        except Exception as e:
            logger.warning(f"Error analyzing code structure: {e}")

        return structure