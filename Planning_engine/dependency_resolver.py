"""Dependency validation, topological ordering and graph queries for tasks."""
from __future__ import annotations

import heapq
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Union

from .planning_core import DependencyCycleError, DependencyError, Task

TaskInput = Union[Mapping[str, Task], Iterable[Task]]


class DependencyResolver:
    """Stateless helper operating on task graphs (``task.dependencies``)."""

    @staticmethod
    def as_mapping(tasks: TaskInput) -> Dict[str, Task]:
        """Normalise a task collection to ``{id: Task}``."""
        if isinstance(tasks, Mapping):
            return dict(tasks)
        result: Dict[str, Task] = {}
        for t in tasks:
            if t.id in result:
                raise DependencyError(f"Duplicate task id '{t.id}'")
            result[t.id] = t
        return result

    def validate(self, tasks: TaskInput) -> None:
        """Raise if a dependency is missing or the graph has a cycle."""
        mapping = self.as_mapping(tasks)
        for t in mapping.values():
            missing = sorted(d for d in t.dependencies if d not in mapping)
            if missing:
                raise DependencyError(f"Task '{t.id}' depends on unknown tasks {missing}")
        cycle = self.find_cycle(mapping)
        if cycle:
            raise DependencyCycleError(cycle)

    def find_cycle(self, tasks: TaskInput) -> Optional[List[str]]:
        """Return one dependency cycle as a path, or ``None``."""
        mapping = self.as_mapping(tasks)
        color = {tid: 0 for tid in mapping}  # 0 new, 1 on stack, 2 done
        for root in sorted(mapping):
            if color[root]:
                continue
            color[root] = 1
            path = [root]
            stack = [(root, iter(sorted(mapping[root].dependencies)))]
            while stack:
                node, it = stack[-1]
                for nxt in it:
                    if nxt not in mapping:
                        continue
                    if color[nxt] == 1:
                        return path[path.index(nxt):] + [nxt]
                    if color[nxt] == 0:
                        color[nxt] = 1
                        path.append(nxt)
                        stack.append((nxt, iter(sorted(mapping[nxt].dependencies))))
                        break
                else:
                    color[node] = 2
                    path.pop()
                    stack.pop()
        return None

    def dependents(self, tasks: TaskInput) -> Dict[str, Set[str]]:
        """Map each task id to the ids that directly depend on it."""
        mapping = self.as_mapping(tasks)
        out: Dict[str, Set[str]] = {tid: set() for tid in mapping}
        for t in mapping.values():
            for d in t.dependencies:
                if d in out:
                    out[d].add(t.id)
        return out

    def topological_order(
        self, tasks: TaskInput, key: Optional[Callable[[Task], Any]] = None
    ) -> List[str]:
        """Kahn's algorithm; ties broken by ``key`` (default: priority, then id)."""
        mapping = self.as_mapping(tasks)
        self.validate(mapping)
        key = key or (lambda t: -int(t.priority))
        deps = self.dependents(mapping)
        indeg = {tid: len(t.dependencies) for tid, t in mapping.items()}
        heap = [(key(mapping[tid]), tid) for tid, n in indeg.items() if n == 0]
        heapq.heapify(heap)
        order: List[str] = []
        while heap:
            _, tid = heapq.heappop(heap)
            order.append(tid)
            for nxt in deps[tid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    heapq.heappush(heap, (key(mapping[nxt]), nxt))
        return order

    def layers(self, tasks: TaskInput) -> List[List[str]]:
        """Group tasks by dependency depth; each layer can run in parallel."""
        mapping = self.as_mapping(tasks)
        depth: Dict[str, int] = {}
        for tid in self.topological_order(mapping):
            deps = mapping[tid].dependencies
            depth[tid] = 1 + max((depth[d] for d in deps), default=-1)
        out: List[List[str]] = [[] for _ in range(max(depth.values(), default=-1) + 1)]
        for tid, d in depth.items():
            out[d].append(tid)
        return [sorted(layer) for layer in out]

    def ancestors(self, tasks: TaskInput, task_id: str) -> Set[str]:
        """All transitive prerequisites of ``task_id``."""
        mapping = self.as_mapping(tasks)
        seen: Set[str] = set()
        todo = [task_id]
        while todo:
            for d in mapping[todo.pop()].dependencies:
                if d not in seen:
                    seen.add(d)
                    todo.append(d)
        return seen

    def descendants(self, tasks: TaskInput, task_id: str) -> Set[str]:
        """All tasks transitively blocked by ``task_id``."""
        deps = self.dependents(tasks)
        seen: Set[str] = set()
        todo = [task_id]
        while todo:
            for d in deps[todo.pop()]:
                if d not in seen:
                    seen.add(d)
                    todo.append(d)
        return seen
