# app/core/python_code_assembler.py
import ast
import asyncio
import logging
import re
import uuid
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from PySide6.QtCore import QObject

try:
    from models.chat_message import ChatMessage, USER_ROLE
    from llm.backend_coordinator import BackendCoordinator
    from core.event_bus import EventBus
except ImportError as e:
    logging.getLogger(__name__).critical(f"PythonCodeAssembler: Import error: {e}")
    raise

logger = logging.getLogger(__name__)


@dataclass
class ImportGroup:
    """Represents a group of imports (stdlib, third-party, local)"""
    stdlib: Set[str]
    third_party: Set[str]
    local: Set[str]


@dataclass
class CodeSection:
    """Represents a section of organized code"""
    section_type: str  # "imports", "constants", "classes", "functions", "main"
    content: str
    priority: int  # Lower number = higher priority


class PythonCodeAssembler(QObject):
    """
    Intelligent assembler that creates production-ready Python files from atomic tasks.
    Handles proper organization, imports, documentation, and integration.
    """

    # Standard library modules for import classification
    STDLIB_MODULES = {
        'os', 'sys', 'json', 'logging', 'datetime', 'time', 'asyncio', 're', 'uuid',
        'pathlib', 'typing', 'dataclasses', 'enum', 'collections', 'itertools',
        'functools', 'operator', 'math', 'random', 'string', 'io', 'csv', 'sqlite3',
        'threading', 'multiprocessing', 'subprocess', 'socket', 'urllib', 'http',
        'email', 'html', 'xml', 'configparser', 'argparse', 'shlex', 'glob',
        'fnmatch', 'tempfile', 'shutil', 'zipfile', 'tarfile', 'gzip', 'pickle',
        'shelve', 'dbm', 'hashlib', 'hmac', 'secrets', 'base64', 'binascii',
        'struct', 'codecs', 'unicodedata', 'stringprep', 'readline', 'rlcompleter'
    }

    def __init__(self, backend_coordinator: BackendCoordinator, event_bus: EventBus, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._backend_coordinator = backend_coordinator
        self._event_bus = event_bus
        logger.info("PythonCodeAssembler initialized")

    async def assemble_production_file(self,
                                       filename: str,
                                       module_purpose: str,
                                       atomic_tasks: List[Any],  # AtomicTask objects
                                       sequence_id: str) -> Optional[str]:
        """
        Assembles atomic tasks into a production-ready Python file.

        Returns the complete, formatted file content or None if assembly fails.
        """
        try:
            logger.info(f"Assembler: Starting assembly of {filename} with {len(atomic_tasks)} atomic tasks")

            # Step 1: Organize atomic tasks by type
            organized_tasks = self._organize_atomic_tasks(atomic_tasks)

            # Step 2: Analyze and collect imports
            imports = self._analyze_imports(atomic_tasks)

            # Step 3: Detect integration needs
            integration_needs = self._detect_integration_needs(atomic_tasks)

            # Step 4: Request planner guidance if needed
            integration_guidance = None
            if integration_needs:
                integration_guidance = await self._request_integration_guidance(
                    filename, integration_needs, sequence_id
                )

            # Step 5: Assemble the complete file
            assembled_content = self._assemble_complete_file(
                filename=filename,
                module_purpose=module_purpose,
                organized_tasks=organized_tasks,
                imports=imports,
                integration_guidance=integration_guidance
            )

            # Step 6: Final validation and formatting
            final_content = self._apply_final_formatting(assembled_content)

            logger.info(f"Assembler: Successfully assembled {filename} ({len(final_content)} characters)")
            return final_content

        except Exception as e:
            logger.error(f"Assembler: Failed to assemble {filename}: {e}", exc_info=True)
            return None

    def _organize_atomic_tasks(self, atomic_tasks: List[Any]) -> Dict[str, List[Any]]:
        """Organize atomic tasks by their type and dependencies"""
        organized = {
            'imports': [],
            'constants': [],
            'classes': defaultdict(list),  # class_name -> list of methods
            'functions': [],
            'main_block': []
        }

        for task in atomic_tasks:
            if not task.generated_code:
                continue

            task_type = task.task_type.lower()

            if task_type == 'import':
                organized['imports'].append(task)
            elif task_type == 'constant':
                organized['constants'].append(task)
            elif task_type == 'class_def':
                organized['classes'][task.name].append(task)
            elif task_type == 'method' and task.parent_context:
                organized['classes'][task.parent_context].append(task)
            elif task_type in ['function', 'async_function']:
                organized['functions'].append(task)
            else:
                # Default to functions for unknown types
                organized['functions'].append(task)

        # Sort functions by dependencies and complexity
        organized['functions'] = self._sort_functions_by_dependencies(organized['functions'])

        # Sort methods within each class
        for class_name in organized['classes']:
            organized['classes'][class_name] = self._sort_class_methods(organized['classes'][class_name])

        return organized

    def _sort_functions_by_dependencies(self, functions: List[Any]) -> List[Any]:
        """Sort functions so dependencies come first"""
        # Create a dependency graph
        func_names = {f.name for f in functions}

        # Simple topological sort
        remaining = functions.copy()
        sorted_functions = []

        while remaining:
            # Find functions with no unresolved dependencies
            ready = []
            for func in remaining:
                deps_in_file = [dep for dep in func.dependencies if dep in func_names]
                resolved_deps = {f.name for f in sorted_functions}
                if all(dep in resolved_deps for dep in deps_in_file):
                    ready.append(func)

            if not ready:
                # If no functions are ready, add the one with fewest dependencies
                ready = [min(remaining, key=lambda f: len(f.dependencies))]

            # Add ready functions to sorted list
            for func in ready:
                sorted_functions.append(func)
                remaining.remove(func)

        return sorted_functions

    def _sort_class_methods(self, methods: List[Any]) -> List[Any]:
        """Sort class methods in logical order"""
        method_order = {
            '__new__': 0,
            '__init__': 1,
            '__post_init__': 2,
            '__str__': 90,
            '__repr__': 91,
            '__eq__': 92,
            '__hash__': 93,
            '__del__': 99
        }

        def method_priority(method):
            name = method.name
            if name in method_order:
                return method_order[name]
            elif name.startswith('__') and name.endswith('__'):
                return 95  # Other dunder methods
            elif name.startswith('_'):
                return 80  # Private methods
            else:
                return 50  # Public methods

        return sorted(methods, key=method_priority)

    def _analyze_imports(self, atomic_tasks: List[Any]) -> ImportGroup:
        """Analyze and organize imports from atomic tasks"""
        imports = ImportGroup(set(), set(), set())

        for task in atomic_tasks:
            if not task.generated_code:
                continue

            # Extract imports from the generated code
            task_imports = self._extract_imports_from_code(task.generated_code)
            imports.stdlib.update(task_imports.stdlib)
            imports.third_party.update(task_imports.third_party)
            imports.local.update(task_imports.local)

        return imports

    def _extract_imports_from_code(self, code: str) -> ImportGroup:
        """Extract imports from a piece of code"""
        imports = ImportGroup(set(), set(), set())

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        self._classify_import(module_name, imports)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        self._classify_import(module_name, imports)

        except SyntaxError:
            # If parsing fails, try regex extraction
            import_lines = re.findall(r'^(?:from\s+(\S+)\s+)?import\s+(.+)', code, re.MULTILINE)
            for from_module, import_names in import_lines:
                if from_module:
                    module_name = from_module.split('.')[0]
                    self._classify_import(module_name, imports)
                else:
                    # Direct imports
                    for name in import_names.split(','):
                        clean_name = name.strip().split('.')[0].split(' as ')[0]
                        self._classify_import(clean_name, imports)

        return imports

    def _classify_import(self, module_name: str, imports: ImportGroup):
        """Classify an import as stdlib, third-party, or local"""
        if module_name in self.STDLIB_MODULES:
            imports.stdlib.add(module_name)
        elif module_name.startswith('.') or module_name in ['app', 'core', 'models', 'services', 'utils']:
            imports.local.add(module_name)
        else:
            imports.third_party.add(module_name)

    def _detect_integration_needs(self, atomic_tasks: List[Any]) -> List[str]:
        """Detect if atomic tasks need integration guidance from planner"""
        integration_needs = []

        # Check for cross-references between tasks
        task_names = {task.name for task in atomic_tasks}

        for task in atomic_tasks:
            # Check if task references other tasks in its code
            if task.generated_code:
                for other_task_name in task_names:
                    if other_task_name != task.name and other_task_name in task.generated_code:
                        integration_needs.append(f"{task.name} references {other_task_name}")

        # Check for potential integration patterns
        class_tasks = [t for t in atomic_tasks if t.task_type == 'class_def']
        method_tasks = [t for t in atomic_tasks if t.task_type == 'method']

        if class_tasks and method_tasks:
            # Ensure methods are properly associated with classes
            for method in method_tasks:
                if not method.parent_context or method.parent_context not in [c.name for c in class_tasks]:
                    integration_needs.append(f"Method {method.name} needs proper class association")

        return integration_needs

    async def _request_integration_guidance(self,
                                            filename: str,
                                            integration_needs: List[str],
                                            sequence_id: str) -> Optional[str]:
        """Request guidance from planner LLM for complex integrations"""
        logger.info(f"Assembler: Requesting integration guidance for {filename}")

        guidance_prompt = f"""You are reviewing code assembly for {filename}. 

INTEGRATION ISSUES DETECTED:
{chr(10).join(f"- {need}" for need in integration_needs)}

Provide concise guidance on how to properly integrate these components:
1. Any missing imports or dependencies
2. Proper ordering of code elements  
3. Integration code needed between components
4. Error handling patterns to unify

Keep response focused and actionable."""

        try:
            request_id = f"integration_{sequence_id}_{uuid.uuid4().hex[:8]}"
            history = [ChatMessage(role=USER_ROLE, parts=[guidance_prompt])]

            # This would need to be implemented as a synchronous call or use a different pattern
            # For now, returning None to indicate no guidance available
            logger.warning("Integration guidance requested but not implemented yet")
            return None

        except Exception as e:
            logger.error(f"Assembler: Failed to get integration guidance: {e}")
            return None

    def _assemble_complete_file(self,
                                filename: str,
                                module_purpose: str,
                                organized_tasks: Dict[str, Any],
                                imports: ImportGroup,
                                integration_guidance: Optional[str]) -> str:
        """Assemble the complete file from organized components"""

        sections = []

        # 1. File header and module docstring
        sections.append(self._create_file_header(filename, module_purpose))

        # 2. Organized imports
        sections.append(self._create_imports_section(imports))

        # 3. Constants
        if organized_tasks['constants']:
            sections.append(self._create_constants_section(organized_tasks['constants']))

        # 4. Classes with their methods
        for class_name, methods in organized_tasks['classes'].items():
            sections.append(self._create_class_section(class_name, methods))

        # 5. Standalone functions
        if organized_tasks['functions']:
            sections.append(self._create_functions_section(organized_tasks['functions']))

        # 6. Main block if needed
        if self._needs_main_block(organized_tasks):
            sections.append(self._create_main_block())

        return '\n\n'.join(filter(None, sections))

    def _create_file_header(self, filename: str, module_purpose: str) -> str:
        """Create file header with module docstring"""
        return f'''"""
{module_purpose}

This module was generated by AvA's intelligent micro-task system.
Each component has been carefully crafted and assembled for production use.
"""

import logging

logger = logging.getLogger(__name__)'''

    def _create_imports_section(self, imports: ImportGroup) -> str:
        """Create properly organized imports section"""
        sections = []

        # Standard library imports
        if imports.stdlib:
            stdlib_imports = sorted(imports.stdlib)
            sections.append('\n'.join(f'import {module}' for module in stdlib_imports))

        # Third-party imports
        if imports.third_party:
            third_party_imports = sorted(imports.third_party)
            sections.append('\n'.join(f'import {module}' for module in third_party_imports))

        # Local imports
        if imports.local:
            local_imports = sorted(imports.local)
            sections.append('\n'.join(f'from {module} import *' for module in local_imports))

        return '\n\n'.join(sections)

    def _create_constants_section(self, constant_tasks: List[Any]) -> str:
        """Create constants section"""
        constants = []
        for task in constant_tasks:
            constants.append(f"# {task.description}")
            constants.append(task.generated_code.strip())

        return '\n'.join(constants)

    def _create_class_section(self, class_name: str, methods: List[Any]) -> str:
        """Create a complete class with all its methods"""
        if not methods:
            return ""

        # Find the class definition task
        class_def = next((m for m in methods if m.task_type == 'class_def'), None)
        method_tasks = [m for m in methods if m.task_type == 'method']

        class_parts = []

        if class_def:
            # Use the generated class definition
            class_parts.append(class_def.generated_code.strip())
        else:
            # Create a basic class definition
            class_parts.append(f"class {class_name}:")
            class_parts.append(f'    """Generated class for {class_name}."""')

        # Add methods with proper indentation
        for method in method_tasks:
            method_code = self._ensure_proper_indentation(method.generated_code, 1)
            class_parts.append("\n" + method_code)

        return '\n'.join(class_parts)

    def _create_functions_section(self, function_tasks: List[Any]) -> str:
        """Create standalone functions section"""
        functions = []

        for task in function_tasks:
            if task.generated_code:
                # Add a comment before each function
                functions.append(f"# {task.description}")
                functions.append(task.generated_code.strip())

        return '\n\n\n'.join(functions)

    def _ensure_proper_indentation(self, code: str, indent_level: int) -> str:
        """Ensure code has proper indentation"""
        lines = code.split('\n')
        indented_lines = []
        base_indent = '    ' * indent_level

        for line in lines:
            if line.strip():  # Don't indent empty lines
                # Remove any existing indentation and add correct indentation
                clean_line = line.lstrip()
                indented_lines.append(base_indent + clean_line)
            else:
                indented_lines.append('')

        return '\n'.join(indented_lines)

    def _needs_main_block(self, organized_tasks: Dict[str, Any]) -> bool:
        """Determine if the file needs a main block"""
        # Check if this looks like a script (has functions but no classes)
        has_functions = bool(organized_tasks['functions'])
        has_classes = bool(organized_tasks['classes'])

        # Simple heuristic: add main block if there are functions but no classes
        return has_functions and not has_classes

    def _create_main_block(self) -> str:
        """Create a main execution block"""
        return '''if __name__ == "__main__":
    # Main execution block
    logger.info("Starting application")
    main()'''

    def _apply_final_formatting(self, content: str) -> str:
        """Apply final formatting and validation"""
        try:
            # Validate syntax
            ast.parse(content)

            # Clean up extra blank lines (max 2 consecutive)
            content = re.sub(r'\n{3,}', '\n\n', content)

            # Ensure file ends with single newline
            content = content.rstrip() + '\n'

            logger.info("Assembler: Final formatting applied successfully")
            return content

        except SyntaxError as e:
            logger.error(f"Assembler: Syntax error in assembled file: {e}")
            # Return content anyway - better than nothing
            return content