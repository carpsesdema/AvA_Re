# app/services/code_analysis_service.py
import ast
import logging
from typing import List, Dict, Any, Optional, Union # Added Union

logger = logging.getLogger(__name__)


class CodeAnalysisService:
    """
    Service for analyzing code structure to extract functions, classes, and other entities.
    """

    def __init__(self):
        """Initialize the code analysis service."""
        logger.info("CodeAnalysisService initialized")

    def parse_python_structures(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse Python code to extract structural information about functions, classes, etc.

        Args:
            content: The Python source code content
            file_path: Path to the file (for error reporting and context)

        Returns:
            List of dictionaries containing structure information
        """
        structures = []

        if not content or not content.strip():
            logger.debug(f"No content to parse for {file_path}")
            return structures

        try:
            # Parse the Python code into an AST
            tree = ast.parse(content)

            # Extract structures
            structures.extend(self._extract_structures_from_ast(tree, content, file_path))

            logger.debug(f"Extracted {len(structures)} code structures from {file_path}")

        except SyntaxError as e:
            logger.warning(f"Syntax error parsing Python file {file_path}: {e}")
            # Try to extract some basic info even with syntax errors
            structures.extend(self._extract_structures_fallback(content, file_path))
        except Exception as e:
            logger.error(f"Error parsing Python structures in {file_path}: {e}", exc_info=True)
            structures.extend(self._extract_structures_fallback(content, file_path))

        return structures

    def _extract_structures_from_ast(self, tree: ast.AST, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract structures using AST parsing."""
        structures = []
        # lines = content.splitlines() # Not directly used here, but can be useful for context

        for node in ast.walk(tree):
            structure_info: Optional[Dict[str, Any]] = None

            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                structure_info = {
                    'type': 'function',
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    'args': [arg.arg for arg in node.args.args] if hasattr(node.args, 'args') else [],
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'docstring': ast.get_docstring(node, clean=False) # Get raw docstring
                }

            elif isinstance(node, ast.ClassDef):
                class_methods = []
                for class_node_item in node.body:
                    if isinstance(class_node_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_info = {
                            'name': class_node_item.name,
                            'start_line': class_node_item.lineno,
                            'end_line': class_node_item.end_lineno if hasattr(class_node_item, 'end_lineno') else class_node_item.lineno,
                            'is_async': isinstance(class_node_item, ast.AsyncFunctionDef),
                            'docstring': ast.get_docstring(class_node_item, clean=False)
                        }
                        class_methods.append(method_info)

                structure_info = {
                    'type': 'class',
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    'bases': [self._get_name_from_node(base) for base in node.bases],
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'methods': class_methods,
                    'docstring': ast.get_docstring(node, clean=False)
                }


            elif isinstance(node, ast.Import):
                for alias in node.names:
                    import_info_item = { # Renamed to avoid conflict
                        'type': 'import',
                        'name': alias.name,
                        'alias': alias.asname,
                        'start_line': node.lineno,
                        'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                        'module': None # For simple imports, module is the name itself
                    }
                    structures.append(import_info_item)
                continue  # Skip adding to structures again below for this node type

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or '' # Module can be None for relative imports like 'from . import foo'
                for alias in node.names:
                    import_from_info_item = { # Renamed
                        'type': 'import_from',
                        'module': module_name,
                        'name': alias.name,
                        'alias': alias.asname,
                        'start_line': node.lineno,
                        'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                    }
                    structures.append(import_from_info_item)
                continue  # Skip adding to structures again below

            if structure_info:
                # Add file_path to each structure for easier context later
                structure_info['file_path'] = file_path
                structures.append(structure_info)

        return structures

    def _extract_structures_fallback(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Fallback method to extract basic structures using regex when AST fails."""
        import re # Local import for fallback only
        structures = []
        lines = content.splitlines()

        # Simple regex patterns for basic structure detection
        function_pattern = re.compile(r'^(\s*)(async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*:', re.MULTILINE)
        class_pattern = re.compile(r'^(\s*)class\s+([a-zA-Z_][a-zA-Z0-9_]*)(\s*\((.*?)\))?\s*:', re.MULTILINE)
        import_pattern = re.compile(r'^(\s*)import\s+([a-zA-Z0-9_.,\s]+)', re.MULTILINE)
        from_import_pattern = re.compile(r'^(\s*)from\s+([a-zA-Z0-9_.]+)\s+import\s+([a-zA-Z0-9_.,\s\*]+)', re.MULTILINE)


        for line_num, line_text in enumerate(lines, 1):
            # Check for function definitions
            func_match = function_pattern.match(line_text)
            if func_match:
                indent, is_async_str, func_name, args_str = func_match.groups()
                args_list = [arg.strip().split(':')[0].strip() for arg in args_str.split(',') if arg.strip()]
                structures.append({
                    'type': 'function',
                    'name': func_name,
                    'start_line': line_num,
                    'end_line': line_num,  # Can't determine end line easily with regex
                    'is_async': bool(is_async_str and is_async_str.strip()),
                    'args': args_list,
                    'fallback_parsing': True,
                    'file_path': file_path
                })
                continue

            # Check for class definitions
            class_match = class_pattern.match(line_text)
            if class_match:
                indent, class_name, _, bases_str = class_match.groups()
                bases_list = [base.strip() for base in bases_str.split(',') if base.strip()] if bases_str else []
                structures.append({
                    'type': 'class',
                    'name': class_name,
                    'start_line': line_num,
                    'end_line': line_num,  # Can't determine end line easily
                    'bases': bases_list,
                    'fallback_parsing': True,
                    'file_path': file_path
                })
                continue

            # Check for 'import ...'
            import_match_simple = import_pattern.match(line_text)
            if import_match_simple:
                indent, import_names_str = import_match_simple.groups()
                imported_items = [item.strip().split(' as ')[0].strip() for item in import_names_str.split(',') if item.strip()]
                for item_name in imported_items:
                    structures.append({
                        'type': 'import',
                        'name': item_name,
                        'module': item_name, # For simple import, module is the name
                        'alias': None, # Regex doesn't easily capture alias here
                        'start_line': line_num,
                        'end_line': line_num,
                        'fallback_parsing': True,
                        'file_path': file_path
                    })
                continue

            # Check for 'from ... import ...'
            from_import_match = from_import_pattern.match(line_text)
            if from_import_match:
                indent, module_name, import_names_str = from_import_match.groups()
                imported_items = [item.strip().split(' as ')[0].strip() for item in import_names_str.split(',') if item.strip()]
                for item_name in imported_items:
                    structures.append({
                        'type': 'import_from',
                        'module': module_name,
                        'name': item_name,
                        'alias': None, # Regex doesn't easily capture alias here
                        'start_line': line_num,
                        'end_line': line_num,
                        'fallback_parsing': True,
                        'file_path': file_path
                    })
                continue


        logger.debug(f"Fallback parsing extracted {len(structures)} structures from {file_path}")
        return structures

    def _get_decorator_name(self, decorator_node: ast.AST) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator_node, ast.Name):
            return decorator_node.id
        elif isinstance(decorator_node, ast.Attribute):
            return self._get_name_from_node(decorator_node) # Recursively get full attribute path
        elif isinstance(decorator_node, ast.Call): # Decorator with arguments
            if isinstance(decorator_node.func, ast.Name):
                return decorator_node.func.id
            elif isinstance(decorator_node.func, ast.Attribute):
                return self._get_name_from_node(decorator_node.func)
        return "unknown_decorator" # Fallback for complex decorator expressions

    def _get_name_from_node(self, node: Union[ast.Name, ast.Attribute, ast.Constant, ast.Call, ast.Subscript, ast.expr]) -> str:
        """Extract name from various AST node types, handling attributes recursively."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Recursively build the attribute path, e.g., "os.path.join"
            value_name = self._get_name_from_node(node.value)
            return f"{value_name}.{node.attr}"
        elif isinstance(node, ast.Constant): # For Python 3.8+
            return str(node.value)
        elif isinstance(node, ast.Str): # For older Python versions (docstrings, etc.)
            return node.s
        elif isinstance(node, ast.Num): # For older Python versions (numbers)
             return str(node.n)
        # Add more specific handlers if needed for other ast types like Call, Subscript, etc.
        else:
            # Fallback for other node types, might not be a simple "name"
            # but provides some representation.
            # Consider logging a warning if a more specific representation is needed.
            try:
                # Attempt to reconstruct a source segment if possible (requires unparse or similar logic)
                # For simplicity, we'll just stringify, but this might not be ideal.
                return ast.dump(node) # A more structured representation than str(node)
            except Exception:
                return "complex_node_type"