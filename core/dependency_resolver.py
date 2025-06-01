# app/core/dependency_resolver.py
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    Intelligent dependency resolver for atomic tasks.
    Handles topological sorting, circular dependency detection,
    and optimal execution ordering.
    """

    def __init__(self):
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._reverse_graph: Dict[str, Set[str]] = {}
        self._execution_order: List[str] = []

    def analyze_and_order_tasks(self, atomic_tasks: List[Any]) -> List[Any]:
        """
        Analyze dependencies and return tasks in optimal execution order.

        Args:
            atomic_tasks: List of AtomicTask objects

        Returns:
            List of tasks ordered for optimal execution
        """
        logger.info(f"DependencyResolver: Analyzing {len(atomic_tasks)} atomic tasks")

        # Build dependency graphs
        self._build_dependency_graphs(atomic_tasks)

        # Detect and resolve circular dependencies
        circular_deps = self._detect_circular_dependencies()
        if circular_deps:
            logger.warning(f"DependencyResolver: Found circular dependencies: {circular_deps}")
            self._resolve_circular_dependencies(circular_deps, atomic_tasks)

        # Perform topological sort
        execution_order = self._topological_sort()

        # Order tasks according to execution order
        ordered_tasks = self._order_tasks_by_execution_order(atomic_tasks, execution_order)

        logger.info(f"DependencyResolver: Ordered {len(ordered_tasks)} tasks for execution")
        return ordered_tasks

    def are_dependencies_met(self, task: Any, all_tasks: List[Any]) -> bool:
        """
        Check if all dependencies for a task have been completed.

        Args:
            task: AtomicTask to check
            all_tasks: List of all tasks to check completion status

        Returns:
            True if all dependencies are met
        """
        if not task.dependencies:
            return True

        completed_tasks = {t.name for t in all_tasks
                           if t.generated_code and hasattr(t, 'status') and
                           str(t.status).endswith('COMPLETED')}

        return all(dep in completed_tasks for dep in task.dependencies)

    def get_next_available_tasks(self,
                                 remaining_tasks: List[Any],
                                 completed_tasks: Set[str]) -> List[Any]:
        """
        Get tasks that have all dependencies completed and can be executed.

        Args:
            remaining_tasks: Tasks not yet completed
            completed_tasks: Set of completed task names

        Returns:
            List of tasks ready for execution
        """
        available = []

        for task in remaining_tasks:
            if all(dep in completed_tasks for dep in task.dependencies):
                available.append(task)

        # Sort by complexity (simpler tasks first) and priority
        available.sort(key=lambda t: (t.estimated_complexity,
                                      len(t.dependencies),
                                      getattr(t, 'priority', 5)))

        return available

    def _build_dependency_graphs(self, atomic_tasks: List[Any]):
        """Build forward and reverse dependency graphs"""
        self._dependency_graph.clear()
        self._reverse_graph.clear()

        # Initialize graphs
        for task in atomic_tasks:
            self._dependency_graph[task.name] = set()
            self._reverse_graph[task.name] = set()

        # Build dependency relationships
        for task in atomic_tasks:
            for dep in task.dependencies:
                if dep in self._dependency_graph:
                    self._dependency_graph[task.name].add(dep)
                    self._reverse_graph[dep].add(task.name)
                else:
                    logger.warning(f"DependencyResolver: Unknown dependency '{dep}' for task '{task.name}'")

    def _detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies using DFS.

        Returns:
            List of cycles found (each cycle is a list of task names)
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {name: WHITE for name in self._dependency_graph}
        cycles = []

        def dfs(node: str, path: List[str]) -> bool:
            if colors[node] == GRAY:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True

            if colors[node] == BLACK:
                return False

            colors[node] = GRAY
            path.append(node)

            for neighbor in self._dependency_graph[node]:
                if dfs(neighbor, path):
                    # Don't return immediately - find all cycles
                    pass

            path.pop()
            colors[node] = BLACK
            return False

        for node in self._dependency_graph:
            if colors[node] == WHITE:
                dfs(node, [])

        return cycles

    def _resolve_circular_dependencies(self, cycles: List[List[str]], atomic_tasks: List[Any]):
        """
        Resolve circular dependencies by breaking the weakest links.

        Strategy:
        1. Identify the weakest dependency in each cycle
        2. Remove it from the dependency graph
        3. Log the resolution for manual review
        """
        logger.info(f"DependencyResolver: Resolving {len(cycles)} circular dependencies")

        for i, cycle in enumerate(cycles):
            logger.warning(f"DependencyResolver: Cycle {i + 1}: {' -> '.join(cycle)}")

            # Find the weakest link (dependency with lowest complexity difference)
            weakest_link = self._find_weakest_link_in_cycle(cycle, atomic_tasks)

            if weakest_link:
                from_task, to_task = weakest_link
                logger.info(f"DependencyResolver: Breaking dependency {from_task} -> {to_task}")

                # Remove the dependency
                self._dependency_graph[from_task].discard(to_task)
                self._reverse_graph[to_task].discard(from_task)

                # Update the atomic task object
                task_obj = next((t for t in atomic_tasks if t.name == from_task), None)
                if task_obj and to_task in task_obj.dependencies:
                    task_obj.dependencies.remove(to_task)
                    logger.info(f"DependencyResolver: Removed {to_task} from {from_task}'s dependencies")

    def _find_weakest_link_in_cycle(self, cycle: List[str], atomic_tasks: List[Any]) -> Optional[Tuple[str, str]]:
        """
        Find the weakest dependency link in a cycle.

        The weakest link is determined by:
        1. Smallest complexity difference between tasks
        2. Least critical dependency (e.g., utility functions)
        """
        task_complexity = {}
        for task in atomic_tasks:
            task_complexity[task.name] = getattr(task, 'estimated_complexity', 1)

        weakest_score = float('inf')
        weakest_link = None

        for i in range(len(cycle) - 1):
            from_task = cycle[i]
            to_task = cycle[i + 1]

            # Calculate weakness score (lower is weaker)
            complexity_diff = abs(task_complexity.get(from_task, 1) -
                                  task_complexity.get(to_task, 1))

            # Prefer breaking dependencies to utility functions
            utility_bonus = 0
            if 'util' in to_task.lower() or 'helper' in to_task.lower():
                utility_bonus = -2

            weakness_score = complexity_diff + utility_bonus

            if weakness_score < weakest_score:
                weakest_score = weakness_score
                weakest_link = (from_task, to_task)

        return weakest_link

    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort using Kahn's algorithm.

        Returns:
            List of task names in topological order
        """
        # Calculate in-degrees
        in_degree = {name: 0 for name in self._dependency_graph}
        for name in self._dependency_graph:
            for dep in self._dependency_graph[name]:
                in_degree[name] += 1

        # Initialize queue with nodes having no dependencies
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            # Process nodes with no remaining dependencies
            current_batch = []
            for _ in range(len(queue)):
                node = queue.popleft()
                current_batch.append(node)
                result.append(node)

                # Update in-degrees of dependent nodes
                for dependent in self._reverse_graph[node]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            logger.debug(f"DependencyResolver: Batch {len(result) // len(current_batch)}: {current_batch}")

        # Check if all nodes were processed (no remaining cycles)
        if len(result) != len(self._dependency_graph):
            unprocessed = [name for name in self._dependency_graph if name not in result]
            logger.error(f"DependencyResolver: Could not resolve all dependencies. Unprocessed: {unprocessed}")
            # Add unprocessed nodes to the end
            result.extend(unprocessed)

        return result

    def _order_tasks_by_execution_order(self,
                                        atomic_tasks: List[Any],
                                        execution_order: List[str]) -> List[Any]:
        """
        Reorder atomic tasks according to the execution order.

        Args:
            atomic_tasks: Original list of tasks
            execution_order: Desired order of task names

        Returns:
            Reordered list of tasks
        """
        task_map = {task.name: task for task in atomic_tasks}
        ordered_tasks = []

        # Add tasks in execution order
        for task_name in execution_order:
            if task_name in task_map:
                ordered_tasks.append(task_map[task_name])

        # Add any tasks not in execution order (shouldn't happen normally)
        for task in atomic_tasks:
            if task not in ordered_tasks:
                logger.warning(f"DependencyResolver: Task '{task.name}' not in execution order, appending")
                ordered_tasks.append(task)

        return ordered_tasks

    def analyze_dependency_levels(self, atomic_tasks: List[Any]) -> Dict[int, List[str]]:
        """
        Analyze dependency levels for parallel execution planning.

        Returns:
            Dictionary mapping level -> list of task names that can run at that level
        """
        self._build_dependency_graphs(atomic_tasks)

        levels = {}
        processed = set()
        current_level = 0

        while len(processed) < len(atomic_tasks):
            current_level_tasks = []

            for task in atomic_tasks:
                if (task.name not in processed and
                        all(dep in processed for dep in task.dependencies)):
                    current_level_tasks.append(task.name)

            if not current_level_tasks:
                # Should not happen if dependencies are properly resolved
                logger.error("DependencyResolver: No tasks available at current level - possible unresolved cycle")
                break

            levels[current_level] = current_level_tasks
            processed.update(current_level_tasks)
            current_level += 1

        logger.info(f"DependencyResolver: Identified {len(levels)} dependency levels")
        for level, tasks in levels.items():
            logger.debug(f"Level {level}: {tasks}")

        return levels

    def get_dependency_stats(self, atomic_tasks: List[Any]) -> Dict[str, Any]:
        """
        Get statistics about the dependency structure.

        Returns:
            Dictionary with dependency statistics
        """
        self._build_dependency_graphs(atomic_tasks)

        total_dependencies = sum(len(deps) for deps in self._dependency_graph.values())
        max_dependencies = max(len(deps) for deps in self._dependency_graph.values()) if self._dependency_graph else 0

        # Find critical path (longest dependency chain)
        critical_path_length = self._find_critical_path_length()

        # Count isolated tasks (no dependencies)
        isolated_tasks = sum(1 for deps in self._dependency_graph.values() if not deps)

        return {
            'total_tasks': len(atomic_tasks),
            'total_dependencies': total_dependencies,
            'max_dependencies_per_task': max_dependencies,
            'critical_path_length': critical_path_length,
            'isolated_tasks': isolated_tasks,
            'average_dependencies': total_dependencies / len(atomic_tasks) if atomic_tasks else 0
        }

    def _find_critical_path_length(self) -> int:
        """Find the length of the critical path (longest dependency chain)"""
        memo = {}

        def dfs(node: str) -> int:
            if node in memo:
                return memo[node]

            if not self._dependency_graph[node]:
                memo[node] = 1
                return 1

            max_depth = 0
            for dep in self._dependency_graph[node]:
                max_depth = max(max_depth, dfs(dep))

            memo[node] = max_depth + 1
            return memo[node]

        return max(dfs(node) for node in self._dependency_graph) if self._dependency_graph else 0