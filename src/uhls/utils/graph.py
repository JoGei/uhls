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
