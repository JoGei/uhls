"""Shared graph traversal helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def breadth_first_walk(
    roots: Iterable[T],
    neighbors: Callable[[T], Iterable[T]],
    *,
    key: Callable[[T], Hashable] | None = None,
) -> Iterator[T]:
    """Yield one breadth-first traversal order with stable de-duplication."""
    if key is None:
        seen: set[object] = set()
        queue: deque[T] = deque()
        append = queue.append
        popleft = queue.popleft

        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            append(root)

        while queue:
            node = popleft()
            yield node
            for child in neighbors(node):
                if child in seen:
                    continue
                seen.add(child)
                append(child)
        return

    seen_keys: set[Hashable] = set()
    queue: deque[T] = deque()
    append = queue.append
    popleft = queue.popleft

    for root in roots:
        root_key = key(root)
        if root_key in seen_keys:
            continue
        seen_keys.add(root_key)
        append(root)

    while queue:
        node = popleft()
        yield node
        for child in neighbors(node):
            child_key = key(child)
            if child_key in seen_keys:
                continue
            seen_keys.add(child_key)
            append(child)


def assert_acyclic(
    nodes: Iterable[T],
    neighbors: Callable[[T], Iterable[T]],
    *,
    key: Callable[[T], Hashable] | None = None,
    cycle_error: Callable[[T], Exception] | None = None,
) -> None:
    """Raise when one graph contains a cycle."""
    if key is None:
        visiting: set[object] = set()
        visited: set[object] = set()

        def visit(node: T) -> None:
            if node in visited:
                return
            if node in visiting:
                if cycle_error is not None:
                    raise cycle_error(node)
                raise ValueError(f"graph contains a cycle involving {node!r}")
            visiting.add(node)
            for child in neighbors(node):
                visit(child)
            visiting.remove(node)
            visited.add(node)

    else:
        visiting_keys: set[Hashable] = set()
        visited_keys: set[Hashable] = set()

        def visit(node: T) -> None:
            node_key = key(node)
            if node_key in visited_keys:
                return
            if node_key in visiting_keys:
                if cycle_error is not None:
                    raise cycle_error(node)
                raise ValueError(f"graph contains a cycle involving {node!r}")
            visiting_keys.add(node_key)
            for child in neighbors(node):
                visit(child)
            visiting_keys.remove(node_key)
            visited_keys.add(node_key)

    for node in nodes:
        visit(node)


def topological_sort(
    nodes: Iterable[T],
    neighbors: Callable[[T], Iterable[T]],
    *,
    key: Callable[[T], Hashable] | None = None,
    cycle_error: Callable[[T], Exception] | None = None,
) -> list[T]:
    """Return one stable topological order or raise when the graph is cyclic."""
    ordered_nodes = list(nodes)

    if key is None:
        node_keys = list(ordered_nodes)
    else:
        node_keys = [key(node) for node in ordered_nodes]

    key_to_node: dict[Hashable, T] = {}
    for node, node_key in zip(ordered_nodes, node_keys, strict=False):
        key_to_node.setdefault(node_key, node)

    adjacency: dict[Hashable, list[Hashable]] = {node_key: [] for node_key in key_to_node}
    indegree: dict[Hashable, int] = {node_key: 0 for node_key in key_to_node}
    seen_edges: set[tuple[Hashable, Hashable]] = set()

    for node, source_key in zip(ordered_nodes, node_keys, strict=False):
        for child in neighbors(node):
            target_key = child if key is None else key(child)
            if target_key not in key_to_node:
                continue
            edge_key = (source_key, target_key)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            adjacency[source_key].append(target_key)
            indegree[target_key] += 1

    ready: deque[Hashable] = deque(node_key for node_key in node_keys if indegree[node_key] == 0)
    topo_keys: list[Hashable] = []
    while ready:
        node_key = ready.popleft()
        topo_keys.append(node_key)
        for child_key in adjacency[node_key]:
            indegree[child_key] -= 1
            if indegree[child_key] == 0:
                ready.append(child_key)

    if len(topo_keys) != len(key_to_node):
        cyclic_key = next(node_key for node_key, degree in indegree.items() if degree > 0)
        cyclic_node = key_to_node[cyclic_key]
        if cycle_error is not None:
            raise cycle_error(cyclic_node)
        raise ValueError(f"graph contains a cycle involving {cyclic_node!r}")

    return [key_to_node[node_key] for node_key in topo_keys]
