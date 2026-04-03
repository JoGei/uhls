"""CFG-oriented analyses for canonical µhLS IR."""

from __future__ import annotations

from dataclasses import dataclass

from uhls.middleend.uir import CondBranchOp, Function
from uhls.middleend.passes.util.pass_manager import AnalysisPass, analysis_pass


@dataclass(frozen=True)
class ControlFlowInfo:
    """Predecessor/successor maps for one function CFG."""

    predecessors: dict[str, set[str]]
    successors: dict[str, set[str]]


@dataclass(frozen=True)
class ControlFlowGraph:
    """Graph view of one function's explicit CFG."""

    function: Function
    entry: str
    order: tuple[str, ...]
    predecessors: dict[str, set[str]]
    successors: dict[str, set[str]]


@dataclass(frozen=True)
class DominatorInfo:
    """Dominator facts for one function CFG."""

    cfg: ControlFlowGraph
    dominators: dict[str, set[str]]
    immediate_dominators: dict[str, str | None]
    tree: dict[str, list[str]]
    frontiers: dict[str, set[str]]

    def dominates(self, dominator: str, node: str) -> bool:
        """Return whether ``dominator`` dominates ``node``."""
        return dominator in self.dominators[node]


@dataclass(frozen=True)
class LoopInfo:
    """One natural loop discovered from a backedge."""

    header: str
    latches: tuple[str, ...]
    body: frozenset[str]
    exits: frozenset[str]


def control_flow(function: Function) -> ControlFlowInfo:
    """Compute predecessor and successor maps for one function."""
    predecessors = {block.label: set() for block in function.blocks}
    successors = {block.label: set() for block in function.blocks}

    for block in function.blocks:
        terminator = block.terminator
        if isinstance(terminator, CondBranchOp):
            successors[block.label].add(terminator.true_target)
            successors[block.label].add(terminator.false_target)
            predecessors[terminator.true_target].add(block.label)
            predecessors[terminator.false_target].add(block.label)
        elif hasattr(terminator, "target"):
            target = terminator.target
            successors[block.label].add(target)
            predecessors[target].add(block.label)

    return ControlFlowInfo(predecessors=predecessors, successors=successors)


def build_cfg(function: Function) -> ControlFlowGraph:
    """Build a CFG graph view for one function."""
    info = control_flow(function)
    return ControlFlowGraph(
        function=function,
        entry=function.entry,
        order=tuple(block.label for block in function.blocks),
        predecessors=info.predecessors,
        successors=info.successors,
    )


def compute_dominators(function: Function, cfg: ControlFlowGraph | None = None) -> DominatorInfo:
    """Compute dominators, immediate dominators, tree, and dominance frontiers."""
    active_cfg = cfg or build_cfg(function)
    dominators = _dominators(active_cfg)
    immediate = _immediate_dominators(active_cfg, dominators)
    tree = _dominator_tree(active_cfg, immediate)
    frontiers = _dominance_frontiers(active_cfg, immediate, tree)
    return DominatorInfo(active_cfg, dominators, immediate, tree, frontiers)


def detect_loops(
    function: Function,
    cfg: ControlFlowGraph | None = None,
    dominators: DominatorInfo | None = None,
) -> list[LoopInfo]:
    """Detect natural loops using CFG backedges and dominators."""
    active_cfg = cfg or build_cfg(function)
    active_dominators = dominators or compute_dominators(function, active_cfg)
    loops_by_header: dict[str, set[str]] = {}

    for source, targets in active_cfg.successors.items():
        for target in targets:
            if active_dominators.dominates(target, source):
                loops_by_header.setdefault(target, set()).add(source)

    results: list[LoopInfo] = []
    for header in active_cfg.order:
        latches = loops_by_header.get(header)
        if not latches:
            continue
        body = {header}
        worklist = list(latches)
        body.update(latches)
        while worklist:
            label = worklist.pop()
            for pred in active_cfg.predecessors[label]:
                if pred not in body:
                    body.add(pred)
                    worklist.append(pred)
        exits = {
            succ
            for member in body
            for succ in active_cfg.successors[member]
            if succ not in body
        }
        results.append(
            LoopInfo(
                header=header,
                latches=tuple(sorted(latches)),
                body=frozenset(body),
                exits=frozenset(exits),
            )
        )
    return results


def control_flow_pass(key: str = "control_flow") -> AnalysisPass:
    """Return a reusable control-flow analysis pass."""
    return analysis_pass("control_flow", control_flow, key=key)


def _dominators(cfg: ControlFlowGraph) -> dict[str, set[str]]:
    labels = list(cfg.order)
    all_blocks = set(labels)
    dominators = {label: set(all_blocks) for label in labels}
    dominators[cfg.entry] = {cfg.entry}

    changed = True
    while changed:
        changed = False
        for label in labels:
            if label == cfg.entry:
                continue
            predecessors = cfg.predecessors[label]
            if not predecessors:
                new_dom = {label}
            else:
                pred_iter = iter(predecessors)
                new_dom = set(dominators[next(pred_iter)])
                for pred in pred_iter:
                    new_dom &= dominators[pred]
                new_dom.add(label)
            if new_dom != dominators[label]:
                dominators[label] = new_dom
                changed = True
    return dominators


def _immediate_dominators(
    cfg: ControlFlowGraph,
    dominators: dict[str, set[str]],
) -> dict[str, str | None]:
    immediate = {cfg.entry: None}
    for label in cfg.order:
        if label == cfg.entry:
            continue
        strict = dominators[label] - {label}
        if not strict:
            immediate[label] = None
            continue
        immediate[label] = max(strict, key=lambda item: len(dominators[item]))
    return immediate


def _dominator_tree(
    cfg: ControlFlowGraph,
    immediate: dict[str, str | None],
) -> dict[str, list[str]]:
    tree = {label: [] for label in cfg.order}
    for label, parent in immediate.items():
        if parent is not None:
            tree[parent].append(label)
    return tree


def _dominance_frontiers(
    cfg: ControlFlowGraph,
    immediate: dict[str, str | None],
    tree: dict[str, list[str]],
) -> dict[str, set[str]]:
    frontier = {label: set() for label in cfg.order}

    def visit(label: str) -> None:
        for child in tree[label]:
            visit(child)

        for succ in cfg.successors[label]:
            if immediate.get(succ) != label:
                frontier[label].add(succ)

        for child in tree[label]:
            for member in frontier[child]:
                if immediate.get(member) != label:
                    frontier[label].add(member)

    visit(cfg.entry)
    return frontier
