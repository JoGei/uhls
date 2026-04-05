"""Shared graph traversal helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator
from heapq import heappop, heappush
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


def intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Return whether two inclusive intervals overlap."""
    return not (left[1] < right[0] or right[1] < left[0])


def interval_conflicts(
    items: Iterable[T],
    interval: Callable[[T], tuple[int, int]],
    *,
    key: Callable[[T], Hashable] | None = None,
) -> dict[Hashable, set[Hashable]]:
    """Build one undirected interval-overlap conflict map."""
    ordered_items = list(items)
    item_keys: dict[Hashable, T] = {}
    conflicts: dict[Hashable, set[Hashable]] = {}

    for item in ordered_items:
        item_key = item if key is None else key(item)
        item_keys[item_key] = item
        conflicts.setdefault(item_key, set())

    for index, left_item in enumerate(ordered_items):
        left_key = left_item if key is None else key(left_item)
        left_interval = interval(left_item)
        for right_item in ordered_items[index + 1 :]:
            right_key = right_item if key is None else key(right_item)
            right_interval = interval(right_item)
            if not intervals_overlap(left_interval, right_interval):
                continue
            conflicts[left_key].add(right_key)
            conflicts[right_key].add(left_key)

    return conflicts


def left_edge_color_intervals(
    items: Iterable[T],
    interval: Callable[[T], tuple[int, int]],
    *,
    key: Callable[[T], Hashable] | None = None,
) -> dict[Hashable, int]:
    """Assign one optimal interval-partition color using a left-edge sweep."""
    ordered_items = list(items)
    if key is None:
        ordered_items.sort(key=interval)
    else:
        ordered_items.sort(key=lambda item: (*interval(item), key(item)))

    colors: dict[Hashable, int] = {}
    active_colors: list[tuple[int, int]] = []
    reusable_colors: list[int] = []
    next_color = 0

    for item in ordered_items:
        start, end = interval(item)
        if end < start:
            raise ValueError(f"invalid interval {start}..{end}")

        while active_colors and active_colors[0][0] < start:
            _, reusable_color = heappop(active_colors)
            heappush(reusable_colors, reusable_color)

        if reusable_colors:
            chosen_color = heappop(reusable_colors)
        else:
            chosen_color = next_color
            next_color += 1

        item_key = item if key is None else key(item)
        colors[item_key] = chosen_color
        heappush(active_colors, (end, chosen_color))

    return colors


def greedy_color_graph(
    nodes: Iterable[T],
    conflicts: Callable[[T], Iterable[T]] | dict[Hashable, set[Hashable]],
    *,
    key: Callable[[T], object] | None = None,
    node_key: Callable[[T], Hashable] | None = None,
) -> dict[Hashable, int]:
    """Assign one stable greedy graph coloring."""
    ordered_nodes = list(nodes)
    if key is not None:
        ordered_nodes.sort(key=key)

    if node_key is None:
        keys = list(ordered_nodes)
    else:
        keys = [node_key(node) for node in ordered_nodes]

    if callable(conflicts):
        adjacency: dict[Hashable, set[Hashable]] = {}
        for node, current_key in zip(ordered_nodes, keys, strict=False):
            adjacency[current_key] = {
                neighbor if node_key is None else node_key(neighbor)
                for neighbor in conflicts(node)
            }
    else:
        adjacency = {current_key: set(conflicts.get(current_key, set())) for current_key in keys}

    colors: dict[Hashable, int] = {}
    for current_key in keys:
        used = {colors[neighbor] for neighbor in adjacency.get(current_key, set()) if neighbor in colors}
        color = 0
        while color in used:
            color += 1
        colors[current_key] = color
    return colors


def greedy_color(
    items: Iterable[T],
    conflicts: Callable[[T], Iterable[T]],
    *,
    key: Callable[[T], Hashable] | None = None,
) -> dict[Hashable, int]:
    """Color one conflict graph greedily in the provided item order."""
    ordered_items = list(items)
    if key is None:
        item_keys = {item: item for item in ordered_items}
    else:
        item_keys = {item: key(item) for item in ordered_items}

    colors: dict[Hashable, int] = {}
    for item in ordered_items:
        used = {colors[item_keys[neighbor]] for neighbor in conflicts(item) if item_keys.get(neighbor) in colors}
        color = 0
        while color in used:
            color += 1
        colors[item_keys[item]] = color
    return colors
