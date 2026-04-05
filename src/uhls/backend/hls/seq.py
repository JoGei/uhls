"""Sequencing-graph construction for HLS lowering."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from uhls.backend.uhir.model import (
    AttributeValue,
    UHIREdge,
    UHIRDesign,
    UHIRNode,
    UHIRPort,
    UHIRRegion,
    UHIRRegionRef,
    UHIRSourceMap,
)
from uhls.middleend.passes.analyze import detect_loops
from uhls.middleend.uir import (
    ArrayType,
    BinaryOp,
    BranchOp,
    CallOp,
    CompareOp,
    CondBranchOp,
    ConstOp,
    Function,
    LoadOp,
    Module,
    ParamOp,
    PhiOp,
    PrintOp,
    ReturnOp,
    StoreOp,
    UnaryOp,
    type_name,
)
from uhls.utils.graph import breadth_first_walk


@dataclass(slots=True)
class SGNode:
    """One node inside a sequencing graph unit."""

    id: str
    opcode: str
    operands: tuple[str, ...] = ()
    result_type: str | None = None
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    children: tuple[str, ...] = ()


@dataclass(slots=True)
class SGEdge:
    """One edge inside a sequencing graph unit."""

    kind: str
    source: str
    target: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass(slots=True)
class SGUnit:
    """One sequencing graph unit."""

    id: str
    kind: str
    parent: str | None = None
    region_refs: list[str] = field(default_factory=list)
    nodes: list[SGNode] = field(default_factory=list)
    edges: list[SGEdge] = field(default_factory=list)
    mappings: list[UHIRSourceMap] = field(default_factory=list)


@dataclass(slots=True)
class SGDesign:
    """One hierarchical sequencing graph."""

    name: str
    inputs: list[UHIRPort] = field(default_factory=list)
    outputs: list[UHIRPort] = field(default_factory=list)
    units: list[SGUnit] = field(default_factory=list)

    def get_unit(self, unit_id: str) -> SGUnit | None:
        """Return one SGU by id."""
        for unit in self.units:
            if unit.id == unit_id:
                return unit
        return None


def build_sequencing_graph(
    module: Module,
    top: str | None = None,
) -> SGDesign:
    """Lower one canonical µIR module to an internal sequencing graph."""
    root = _select_top_function(module, top)
    reachable = _reachable_functions(module, root)
    external_callees = _reachable_external_callees(module, reachable)
    lowerer = _SeqLowerer(module=module, reachable=reachable, external_callees=external_callees, top=root)
    return lowerer.lower()


def lower_module_to_seq(
    module: Module,
    top: str | None = None,
) -> UHIRDesign:
    """Lower one canonical µIR module to hierarchical `seq` µhIR."""
    return _sequencing_graph_to_uhir(build_sequencing_graph(module, top))


def _sequencing_graph_to_uhir(graph: SGDesign) -> UHIRDesign:
    design = UHIRDesign(name=graph.name, stage="seq")
    design.inputs = list(graph.inputs)
    design.outputs = list(graph.outputs)
    design.regions = [_sg_unit_to_uhir(unit) for unit in graph.units]
    return design


def _sg_unit_to_uhir(unit: SGUnit) -> UHIRRegion:
    region = UHIRRegion(id=unit.id, kind=unit.kind, parent=unit.parent)
    seen_refs: set[str] = set(unit.region_refs)
    for node in unit.nodes:
        seen_refs.update(node.children)
        region.nodes.append(_sg_node_to_uhir(node))
    for edge in unit.edges:
        if _looks_like_region_ref(edge.source):
            seen_refs.add(edge.source)
        if _looks_like_region_ref(edge.target):
            seen_refs.add(edge.target)
        region.edges.append(UHIREdge(edge.kind, edge.source, edge.target, dict(edge.attributes)))
    region.region_refs.extend(UHIRRegionRef(target) for target in sorted(seen_refs))
    region.mappings = list(unit.mappings)
    return region


def _sg_node_to_uhir(node: SGNode) -> UHIRNode:
    attributes = dict(node.attributes)
    if node.opcode in {"call", "loop"} and node.children:
        attributes["child"] = node.children[0]
    if node.opcode == "branch":
        if len(node.children) >= 1:
            attributes["true_child"] = node.children[0]
        if len(node.children) >= 2:
            attributes["false_child"] = node.children[1]
    return UHIRNode(node.id, node.opcode, node.operands, node.result_type, attributes)


def _looks_like_region_ref(name: str) -> bool:
    return name.startswith(("proc_", "bb_", "loop_"))


def _node_id_by_role(unit: SGUnit, role: str) -> str | None:
    for node in unit.nodes:
        if node.opcode == "nop" and node.attributes.get("role") == role:
            return node.id
    return None


def _select_top_function(module: Module, requested: str | None) -> Function:
    if requested is not None:
        function = module.get_function(requested)
        if function is None:
            raise ValueError(f"unknown top function '{requested}'")
        return function
    if len(module.functions) == 1:
        return module.functions[0]
    main_function = module.get_function("main")
    if main_function is not None:
        return main_function
    raise ValueError("module contains multiple functions; pass --top")


def _reachable_functions(module: Module, root: Function) -> list[Function]:
    discovered: dict[str, Function] = {}
    worklist = [root]
    while worklist:
        function = worklist.pop()
        if function.name in discovered:
            continue
        discovered[function.name] = function
        for block in function.blocks:
            for instruction in block.instructions:
                if isinstance(instruction, CallOp):
                    callee = module.get_function(instruction.callee)
                    if callee is not None:
                        worklist.append(callee)
    return [discovered[root.name], *sorted((f for name, f in discovered.items() if name != root.name), key=lambda item: item.name)]


def _reachable_external_callees(module: Module, reachable: list[Function]) -> list[str]:
    locals_by_name = {function.name for function in module.functions}
    externals: set[str] = set()
    for function in reachable:
        for block in function.blocks:
            for instruction in block.instructions:
                if isinstance(instruction, CallOp) and instruction.callee not in locals_by_name:
                    externals.add(instruction.callee)
    return sorted(externals)


@dataclass
class _LoopSummary:
    header: str
    region_id: str
    helper_id: str
    body_region_id: str
    empty_region_id: str
    body: frozenset[str]


@dataclass
class _BranchSummary:
    header: str
    true_target: str
    false_target: str
    true_region_id: str
    false_region_id: str
    child_blocks: frozenset[str]
    join_target: str | None = None
    empty_region_id: str | None = None


@dataclass
class _SeqLowerer:
    module: Module
    reachable: list[Function]
    external_callees: list[str]
    top: Function
    value_counter: int = 0

    def lower(self) -> SGDesign:
        design = SGDesign(name=self.module.name or self.top.name)
        design.inputs.extend(self._top_inputs())
        if type_name(self.top.return_type) is not None:
            design.outputs.append(UHIRPort("output", "result", type_name(self.top.return_type) or "i32"))

        units: list[SGUnit] = []
        for function in self.reachable:
            units.extend(self._lower_function(function))
        for callee in self.external_callees:
            units.append(self._external_stub_unit(callee))
        for unit in units:
            self._close_dangling_vertices(unit)
            self._prune_redundant_boundary_edges(unit)
            self._prune_transitive_data_edges(unit)
            self._dedupe_edges(unit)
        design.units = units
        self._relabel_nodes_breadth_first(design, _function_region_id(self.top.name))
        return design

    def _top_inputs(self) -> list[UHIRPort]:
        ports: list[UHIRPort] = []
        for parameter in self.top.params:
            parameter_type = parameter.type
            if isinstance(parameter_type, ArrayType):
                ports.append(UHIRPort("input", parameter.name, f"memref<{parameter_type.element_type}>"))
            else:
                ports.append(UHIRPort("input", parameter.name, type_name(parameter_type) or "i32"))
        return ports

    def _lower_function(self, function: Function) -> list[SGUnit]:
        proc_id = _function_region_id(function.name)
        loop_infos = detect_loops(function)
        top_level_loops = self._top_level_loops(function, loop_infos)
        loop_by_header = {loop.header: loop for loop in top_level_loops}
        branch_layouts = self._branch_layouts(function, loop_by_header)
        branch_child_blocks = {label for layout in branch_layouts.values() for label in layout.child_blocks}
        block_units = [self._lower_block(function, label, proc_id) for label in sorted(branch_child_blocks)]
        block_units.extend(
            self._lower_branch_empty_unit(layout.empty_region_id, proc_id)
            for layout in branch_layouts.values()
            if layout.empty_region_id is not None
        )
        loop_units: list[SGUnit] = []
        for loop in top_level_loops:
            loop_units.append(self._lower_loop_unit(function, loop, proc_id))
            loop_units.append(self._lower_loop_body_unit(function, loop))
            loop_units.append(self._lower_empty_unit(loop))

        proc_unit = SGUnit(id=proc_id, kind="procedure")
        source = self._nop_node("source")
        sink = self._nop_node("sink")
        proc_unit.nodes.extend([source, sink])

        proc_unit.region_refs.extend(
            sorted(
                {_block_region_id(function.name, label) for label in branch_child_blocks}
                | {layout.empty_region_id for layout in branch_layouts.values() if layout.empty_region_id is not None}
            )
        )
        proc_unit.region_refs.extend(sorted(loop.region_id for loop in top_level_loops))
        self._lower_top_level_proc(
            function,
            proc_unit,
            source.id,
            sink.id,
            loop_by_header,
            branch_child_blocks,
            branch_layouts,
        )

        return [proc_unit, *loop_units, *block_units]

    def _branch_layouts(self, function: Function, loop_by_header: dict[str, _LoopSummary]) -> dict[str, _BranchSummary]:
        layouts: dict[str, _BranchSummary] = {}
        for block in function.blocks:
            terminator = block.terminator
            if not isinstance(terminator, CondBranchOp):
                continue
            if block.label in loop_by_header:
                continue
            join_target = self._branch_join_target(function, terminator.true_target, terminator.false_target)
            empty_region_id: str | None = None
            true_region_id = _block_region_id(function.name, terminator.true_target)
            false_region_id = _block_region_id(function.name, terminator.false_target)
            child_blocks = {terminator.true_target, terminator.false_target}
            if join_target is None:
                if _target_reaches_target(function, terminator.true_target, terminator.false_target):
                    join_target = terminator.false_target
                elif _target_reaches_target(function, terminator.false_target, terminator.true_target):
                    join_target = terminator.true_target
            if join_target == terminator.true_target:
                empty_region_id = _branch_empty_region_id(function.name, block.label, "true")
                true_region_id = empty_region_id
                child_blocks.discard(terminator.true_target)
            elif join_target == terminator.false_target:
                empty_region_id = _branch_empty_region_id(function.name, block.label, "false")
                false_region_id = empty_region_id
                child_blocks.discard(terminator.false_target)
            layouts[block.label] = _BranchSummary(
                header=block.label,
                true_target=terminator.true_target,
                false_target=terminator.false_target,
                true_region_id=true_region_id,
                false_region_id=false_region_id,
                child_blocks=frozenset(child_blocks),
                join_target=join_target,
                empty_region_id=empty_region_id,
            )
        return layouts

    def _lower_top_level_proc(
        self,
        function: Function,
        proc_unit: SGUnit,
        source_id: str,
        sink_id: str,
        loop_by_header: dict[str, _LoopSummary],
        branch_child_blocks: set[str],
        branch_layouts: dict[str, _BranchSummary],
    ) -> None:
        block_map = function.block_map()
        node_defs: dict[str, str] = {}
        last_memory: dict[str, str] = {}
        cursor = source_id
        current = function.entry
        visited: set[str] = set()

        while current not in visited:
            if current in branch_child_blocks:
                proc_unit.edges.append(SGEdge("data", cursor, sink_id))
                return
            if current in loop_by_header:
                loop = loop_by_header[current]
                helper = SGNode(
                    loop.helper_id,
                    "loop",
                    (),
                    children=(loop.region_id,),
                )
                proc_unit.nodes.append(helper)
                proc_unit.edges.append(SGEdge("data", cursor, helper.id))
                proc_unit.edges.append(_hier_edge(helper.id, loop.region_id))
                proc_unit.edges.append(_hier_edge(loop.region_id, helper.id))
                for defined_name in _loop_defined_names(function, loop):
                    node_defs[defined_name] = helper.id
                cursor = helper.id
                current = _loop_exit_target(block_map[current].terminator)
                if current is None:
                    proc_unit.edges.append(SGEdge("data", cursor, sink_id))
                    return
                continue

            visited.add(current)
            block = block_map[current]
            cursor, _ = self._append_instructions(
                proc_unit,
                block.instructions,
                cursor=cursor,
                node_defs=node_defs,
                last_memory=last_memory,
            )
            terminator = block.terminator

            if isinstance(terminator, BranchOp):
                current = terminator.target
                continue

            if isinstance(terminator, CondBranchOp):
                layout = branch_layouts[current]
                helper_id = self._new_node_id()
                helper = SGNode(
                    helper_id,
                    "branch",
                    (_format_operand_name(terminator.cond),),
                    children=(layout.true_region_id, layout.false_region_id),
                )
                proc_unit.nodes.append(helper)
                proc_unit.edges.append(SGEdge("data", cursor, helper.id))
                proc_unit.edges.append(_hier_edge(helper.id, layout.true_region_id, {"when": True}))
                proc_unit.edges.append(_hier_edge(helper.id, layout.false_region_id, {"when": False}))
                proc_unit.edges.append(_hier_edge(layout.true_region_id, helper.id, {"when": True}))
                proc_unit.edges.append(_hier_edge(layout.false_region_id, helper.id, {"when": False}))
                cursor = helper.id
                join_target = layout.join_target
                if join_target is None or join_target in branch_child_blocks:
                    proc_unit.edges.append(SGEdge("data", cursor, sink_id))
                    return
                current = join_target
                continue

            if isinstance(terminator, ReturnOp):
                cursor = self._append_return_node(proc_unit, terminator, cursor, node_defs)
            else:
                proc_unit.edges.append(SGEdge("data", cursor, sink_id))
            return

        proc_unit.edges.append(SGEdge("data", cursor, sink_id))

    def _append_instructions(
        self,
        unit: SGUnit,
        instructions: list[object],
        *,
        cursor: str,
        node_defs: dict[str, str],
        last_memory: dict[str, str],
    ) -> tuple[str, list[str]]:
        source_id = _node_id_by_role(unit, "source")
        assert source_id is not None

        produced_nodes: list[str] = []
        last_node_id = cursor
        for instruction in instructions:
            node = self._instruction_node(instruction)
            unit.nodes.append(node)
            produced_nodes.append(node.id)
            unit.mappings.extend(_instruction_maps(node.id, instruction))

            if len(produced_nodes) == 1 and cursor != source_id:
                unit.edges.append(SGEdge("data", cursor, node.id))

            for operand_name in _instruction_uses(instruction):
                producer = node_defs.get(operand_name, source_id)
                unit.edges.append(SGEdge("data", producer, node.id))

            memory_name = _instruction_memory_name(instruction)
            if memory_name is not None:
                producer = last_memory.get(memory_name, source_id)
                unit.edges.append(SGEdge("mem", producer, node.id))
                last_memory[memory_name] = node.id

            if isinstance(instruction, CallOp):
                self._connect_call_hierarchy(unit, node.id, instruction.callee)

            last_node_id = node.id
            dest_name = _instruction_dest(instruction)
            if dest_name is not None:
                node_defs[dest_name] = node.id

        return last_node_id, produced_nodes

    def _connect_call_hierarchy(self, unit: SGUnit, node_id: str, callee: str) -> None:
        callee_region = _function_region_id(callee)
        if callee_region not in unit.region_refs:
            unit.region_refs.append(callee_region)
        unit.edges.append(_hier_edge(node_id, callee_region))
        unit.edges.append(_hier_edge(callee_region, node_id))

    def _connect_produced_nodes_to_sink(self, unit: SGUnit, produced_nodes: list[str]) -> None:
        sink_id = _node_id_by_role(unit, "sink")
        assert sink_id is not None
        used_as_source = {edge.source for edge in unit.edges if edge.kind == "data"}
        for node_id in produced_nodes:
            if node_id not in used_as_source:
                unit.edges.append(SGEdge("data", node_id, sink_id))

    def _append_return_node(
        self,
        unit: SGUnit,
        terminator: ReturnOp,
        cursor: str,
        node_defs: dict[str, str],
    ) -> str:
        operands = () if terminator.value is None else (_format_operand_name(terminator.value),)
        node = SGNode(self._new_node_id(), "ret", operands)
        unit.nodes.append(node)
        added_predecessor = False
        if terminator.value is not None:
            operand_name = _format_operand_name(terminator.value)
            if _looks_like_symbol_operand(operand_name):
                producer = node_defs.get(operand_name, cursor)
                unit.edges.append(SGEdge("data", producer, node.id))
                added_predecessor = True
        if not added_predecessor:
            unit.edges.append(SGEdge("data", cursor, node.id))
        unit.edges.append(SGEdge("data", node.id, unit.nodes[1].id))
        return node.id

    def _top_level_loops(self, function: Function, loop_infos: list[object]) -> list[_LoopSummary]:
        summaries: list[_LoopSummary] = []
        for info in loop_infos:
            if any(info.body < other.body for other in loop_infos):
                continue
            summaries.append(
                _LoopSummary(
                    header=info.header,
                    region_id=_loop_region_id(function.name, info.header),
                    helper_id=self._new_node_id(),
                    body_region_id=_loop_body_region_id(function.name, info.header),
                    empty_region_id=_loop_empty_region_id(function.name, info.header),
                    body=info.body,
                )
            )
        return summaries

    def _branch_join_target(
        self,
        function: Function,
        true_target: str,
        false_target: str,
    ) -> str | None:
        block_map = function.block_map()

        def successors(label: str) -> Iterator[str]:
            successor = _direct_successor_label(block_map[label].terminator)
            if successor is not None:
                yield successor

        false_reachable = set(breadth_first_walk([false_target], successors))
        for label in breadth_first_walk([true_target], successors):
            if label in false_reachable and label not in {true_target, false_target}:
                return label
        return None

    def _lower_loop_unit(self, function: Function, loop: _LoopSummary, parent_id: str) -> SGUnit:
        unit = SGUnit(id=loop.region_id, kind="loop", parent=parent_id)
        source = self._nop_node("source")
        sink = self._nop_node("sink")
        unit.nodes.extend([source, sink])
        unit.region_refs.extend([loop.body_region_id, loop.empty_region_id])

        block_map = function.block_map()
        header_block = block_map[loop.header]
        node_defs: dict[str, str] = {}
        last_memory: dict[str, str] = {}
        compare_node_id: str | None = None

        last_node_id, produced_nodes = self._append_instructions(
            unit,
            header_block.instructions,
            cursor=source.id,
            node_defs=node_defs,
            last_memory=last_memory,
        )
        for instruction, node in zip(header_block.instructions, unit.nodes[2: 2 + len(header_block.instructions)], strict=False):
            if isinstance(instruction, CompareOp):
                compare_node_id = node.id

        branch_id = self._new_node_id()
        branch_operands = () if compare_node_id is None else (_format_operand_name(header_block.terminator.cond),)
        unit.nodes.append(
            SGNode(
                branch_id,
                "branch",
                branch_operands,
                children=(loop.body_region_id, loop.empty_region_id),
            )
        )
        unit.edges.append(SGEdge("data", compare_node_id or last_node_id, branch_id))
        unit.edges.append(_hier_edge(branch_id, loop.body_region_id, {"when": True}))
        unit.edges.append(_hier_edge(branch_id, loop.empty_region_id, {"when": False}))
        unit.edges.append(_hier_edge(loop.body_region_id, branch_id, {"when": True}))
        unit.edges.append(_hier_edge(loop.empty_region_id, branch_id, {"when": False}))
        unit.edges.append(SGEdge("data", branch_id, sink.id))

        self._connect_produced_nodes_to_sink(unit, produced_nodes)
        return unit

    def _lower_loop_body_unit(self, function: Function, loop: _LoopSummary) -> SGUnit:
        unit = SGUnit(id=loop.body_region_id, kind="body", parent=loop.region_id)
        source = self._nop_node("source")
        sink = self._nop_node("sink")
        unit.nodes.extend([source, sink])
        block_map = function.block_map()
        ordered_blocks = [
            block_map[block.label]
            for block in function.blocks
            if block.label in loop.body and block.label != loop.header
        ]
        node_defs: dict[str, str] = {}
        last_memory: dict[str, str] = {}
        produced_nodes: list[str] = []

        for block in ordered_blocks:
            _, block_nodes = self._append_instructions(
                unit,
                block.instructions,
                cursor=source.id,
                node_defs=node_defs,
                last_memory=last_memory,
            )
            produced_nodes.extend(block_nodes)

        self._connect_produced_nodes_to_sink(unit, produced_nodes)
        return unit

    def _lower_empty_unit(self, loop: _LoopSummary) -> SGUnit:
        unit = SGUnit(id=loop.empty_region_id, kind="empty", parent=loop.region_id)
        source = self._nop_node("source")
        sink = self._nop_node("sink")
        unit.nodes.extend([source, sink])
        unit.edges.append(SGEdge("data", source.id, sink.id))
        return unit

    def _lower_branch_empty_unit(self, region_id: str, parent_id: str) -> SGUnit:
        unit = SGUnit(id=region_id, kind="empty", parent=parent_id)
        source = self._nop_node("source")
        sink = self._nop_node("sink")
        unit.nodes.extend([source, sink])
        unit.edges.append(SGEdge("data", source.id, sink.id))
        return unit

    def _lower_block(self, function: Function, block_label: str, parent_id: str) -> SGUnit:
        block = function.block_map()[block_label]
        unit = SGUnit(id=_block_region_id(function.name, block_label), kind="basicblock", parent=parent_id)
        source = self._nop_node("source")
        sink = self._nop_node("sink")
        unit.nodes.extend([source, sink])

        node_defs: dict[str, str] = {}
        last_memory: dict[str, str] = {}
        last_node_id, produced_nodes = self._append_instructions(
            unit,
            block.instructions,
            cursor=source.id,
            node_defs=node_defs,
            last_memory=last_memory,
        )

        if isinstance(block.terminator, ReturnOp):
            last_node_id = self._append_return_node(unit, block.terminator, last_node_id, node_defs)
        elif not produced_nodes:
            unit.edges.append(SGEdge("data", source.id, sink.id))
        self._connect_produced_nodes_to_sink(unit, produced_nodes)
        return unit

    def _instruction_node(self, instruction: object) -> SGNode:
        node_id = self._new_node_id()
        opcode = getattr(instruction, "opcode", instruction.__class__.__name__)
        operands = tuple(_instruction_operands(instruction))
        result_type = type_name(getattr(instruction, "type", None))
        if isinstance(instruction, CallOp):
            return SGNode(
                node_id,
                "call",
                operands,
                result_type,
                children=(_function_region_id(instruction.callee),),
            )
        return SGNode(node_id, opcode, operands, result_type)

    def _external_stub_unit(self, callee: str) -> SGUnit:
        unit = SGUnit(id=_function_region_id(callee), kind="procedure")
        source = self._nop_node("source")
        sink = self._nop_node("sink")
        unit.nodes.extend([source, sink])
        return unit

    def _close_dangling_vertices(self, unit: SGUnit) -> None:
        source_id = _node_id_by_role(unit, "source")
        sink_id = _node_id_by_role(unit, "sink")
        if source_id is None or sink_id is None:
            return

        predecessor_kinds = {"data", "mem"}
        successor_kinds = {"data", "mem"}
        existing = {(edge.kind, edge.source, edge.target) for edge in unit.edges}

        for node in unit.nodes:
            if node.id in {source_id, sink_id}:
                continue

            has_predecessor = any(
                edge.kind in predecessor_kinds and edge.target == node.id and edge.source != node.id for edge in unit.edges
            )
            if not has_predecessor and ("data", source_id, node.id) not in existing:
                unit.edges.append(SGEdge("data", source_id, node.id))
                existing.add(("data", source_id, node.id))

            has_successor = any(
                edge.kind in successor_kinds and edge.source == node.id and edge.target != node.id for edge in unit.edges
            )
            if not has_successor and ("data", node.id, sink_id) not in existing:
                unit.edges.append(SGEdge("data", node.id, sink_id))
                existing.add(("data", node.id, sink_id))

    def _dedupe_edges(self, unit: SGUnit) -> None:
        seen: set[tuple[object, ...]] = set()
        unique: list[SGEdge] = []
        for edge in unit.edges:
            key = (
                edge.kind,
                edge.source,
                edge.target,
                tuple(sorted(edge.attributes.items())),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(edge)
        unit.edges = unique

    def _prune_redundant_boundary_edges(self, unit: SGUnit) -> None:
        source_id = _node_id_by_role(unit, "source")
        sink_id = _node_id_by_role(unit, "sink")
        if source_id is None or sink_id is None:
            return

        incoming_by_target: dict[str, list[SGEdge]] = {}
        outgoing_by_source: dict[str, list[SGEdge]] = {}
        for edge in unit.edges:
            if edge.kind not in {"data", "mem"}:
                continue
            incoming_by_target.setdefault(edge.target, []).append(edge)
            outgoing_by_source.setdefault(edge.source, []).append(edge)

        kept: list[SGEdge] = []
        for edge in unit.edges:
            if edge.kind not in {"data", "mem"}:
                kept.append(edge)
                continue

            if edge.source == source_id and edge.target not in {source_id, sink_id}:
                has_other_incoming = any(other.source != source_id for other in incoming_by_target.get(edge.target, []))
                if has_other_incoming:
                    continue

            if edge.target == sink_id and edge.source not in {source_id, sink_id}:
                has_other_outgoing = any(other.target != sink_id for other in outgoing_by_source.get(edge.source, []))
                if has_other_outgoing:
                    continue

            kept.append(edge)
        unit.edges = kept

    def _prune_transitive_data_edges(self, unit: SGUnit) -> None:
        data_edges = [edge for edge in unit.edges if edge.kind == "data"]
        if len(data_edges) < 2:
            return

        adjacency: dict[str, list[str]] = {}
        for edge in data_edges:
            adjacency.setdefault(edge.source, []).append(edge.target)

        def reaches(source: str, target: str, skipped_edge: SGEdge) -> bool:
            visited: set[str] = set()

            def successors(node_id: str) -> Iterator[str]:
                for succ_id in adjacency.get(node_id, []):
                    if node_id == skipped_edge.source and succ_id == skipped_edge.target:
                        continue
                    if succ_id not in visited:
                        yield succ_id

            for node_id in breadth_first_walk([source], successors):
                if node_id == source:
                    continue
                if node_id == target:
                    return True
                visited.add(node_id)
            return False

        kept: list[SGEdge] = []
        for edge in unit.edges:
            if edge.kind != "data":
                kept.append(edge)
                continue
            if reaches(edge.source, edge.target, edge):
                continue
            kept.append(edge)
        unit.edges = kept

    def _nop_node(self, role: str) -> SGNode:
        return SGNode(self._new_node_id(), "nop", attributes={"role": role})

    def _new_node_id(self) -> str:
        self.value_counter += 1
        return f"v{self.value_counter}"

    def _relabel_nodes_breadth_first(self, design: SGDesign, root_unit_id: str) -> None:
        unit_by_id = {unit.id: unit for unit in design.units}
        ordered_units = [
            unit_by_id[unit_id]
            for unit_id in breadth_first_walk(
                [root_unit_id] if root_unit_id in unit_by_id else [],
                lambda unit_id: self._child_unit_ids(unit_by_id[unit_id], unit_by_id),
            )
        ]

        for unit in design.units:
            if unit not in ordered_units:
                ordered_units.append(unit)

        rename: dict[str, str] = {}
        next_id = 0
        for unit in ordered_units:
            for node in self._breadth_first_nodes(unit):
                rename[node.id] = f"v{next_id}"
                next_id += 1

        for unit in design.units:
            for node in unit.nodes:
                node.id = rename[node.id]
            for edge in unit.edges:
                edge.source = rename.get(edge.source, edge.source)
                edge.target = rename.get(edge.target, edge.target)
            for mapping in unit.mappings:
                object.__setattr__(mapping, "node_id", rename[mapping.node_id])

    def _breadth_first_nodes(self, unit: SGUnit) -> list[SGNode]:
        node_by_id = {node.id: node for node in unit.nodes}
        node_order = {node.id: index for index, node in enumerate(unit.nodes)}
        source = next(
            (node for node in unit.nodes if node.opcode == "nop" and node.attributes.get("role") == "source"),
            None,
        )
        sink = next((node for node in unit.nodes if node.opcode == "nop" and node.attributes.get("role") == "sink"), None)
        if source is None:
            return list(unit.nodes)

        adjacency: dict[str, list[str]] = {node.id: [] for node in unit.nodes}
        seen_pairs: set[tuple[str, str]] = set()
        for edge in unit.edges:
            if edge.kind not in {"data", "mem"}:
                continue
            if edge.source not in node_by_id or edge.target not in node_by_id:
                continue
            pair = (edge.source, edge.target)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            adjacency[edge.source].append(edge.target)
        for targets in adjacency.values():
            targets.sort(key=lambda target_id: node_order[target_id])

        sink_id = None if sink is None else sink.id
        ordered = [
            node_by_id[node_id]
            for node_id in breadth_first_walk(
                [source.id],
                lambda node_id: (target_id for target_id in adjacency[node_id] if target_id != sink_id),
            )
        ]
        visited = {node.id for node in ordered}

        for node in unit.nodes:
            if node.id in visited or node.id == sink_id:
                continue
            ordered.append(node)
            visited.add(node.id)

        if sink is not None:
            ordered.append(sink)
        return ordered

    def _child_unit_ids(self, unit: SGUnit, unit_by_id: dict[str, SGUnit]) -> Iterator[str]:
        for node in self._breadth_first_nodes(unit):
            for child_id in node.children:
                if child_id in unit_by_id:
                    yield child_id


def _function_region_id(name: str) -> str:
    return f"proc_{_sanitize(name)}"


def _block_region_id(function_name: str, block_label: str) -> str:
    return f"bb_{_sanitize(function_name)}_{_sanitize(block_label)}"


def _loop_region_id(function_name: str, header: str) -> str:
    return f"loop_{_sanitize(function_name)}_{_sanitize(header)}"


def _loop_body_region_id(function_name: str, header: str) -> str:
    return f"loop_body_{_sanitize(function_name)}_{_sanitize(header)}"


def _loop_empty_region_id(function_name: str, header: str) -> str:
    return f"loop_empty_{_sanitize(function_name)}_{_sanitize(header)}"


def _branch_empty_region_id(function_name: str, header: str, arm: str) -> str:
    return f"branch_empty_{_sanitize(function_name)}_{_sanitize(header)}_{_sanitize(arm)}"


def _sanitize(text: str) -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in text)
    if not safe:
        return "anon"
    if safe[0].isdigit():
        return f"_{safe}"
    return safe


def _instruction_dest(instruction: object) -> str | None:
    return getattr(instruction, "dest", None)


def _instruction_uses(instruction: object) -> list[str]:
    operands = _instruction_operands(instruction)
    return [operand for operand in operands if _looks_like_symbol_operand(operand)]


def _instruction_operands(instruction: object) -> list[str]:
    if isinstance(instruction, ConstOp):
        return [str(instruction.value)]
    if isinstance(instruction, UnaryOp):
        return [_format_operand_name(instruction.value)]
    if isinstance(instruction, (BinaryOp, CompareOp)):
        return [_format_operand_name(instruction.lhs), _format_operand_name(instruction.rhs)]
    if isinstance(instruction, LoadOp):
        return [instruction.array, _format_operand_name(instruction.index)]
    if isinstance(instruction, StoreOp):
        return [instruction.array, _format_operand_name(instruction.index), _format_operand_name(instruction.value)]
    if isinstance(instruction, PhiOp):
        return [_format_operand_name(item.value) for item in instruction.incoming]
    if isinstance(instruction, CallOp):
        return [instruction.callee, *[_format_operand_name(operand) for operand in instruction.operands]]
    if isinstance(instruction, PrintOp):
        return [_format_operand_name(operand) for operand in instruction.operands]
    if isinstance(instruction, ParamOp):
        return [str(instruction.index), _format_operand_name(instruction.value)]
    return []


def _instruction_memory_name(instruction: object) -> str | None:
    if isinstance(instruction, (LoadOp, StoreOp)):
        return instruction.array
    return None


def _instruction_maps(node_id: str, instruction: object) -> list[UHIRSourceMap]:
    dest = _instruction_dest(instruction)
    if dest is not None:
        return [UHIRSourceMap(node_id=node_id, source_id=dest)]
    if isinstance(instruction, CallOp):
        return [UHIRSourceMap(node_id=node_id, source_id=f"call:{instruction.callee}")]
    return []


def _looks_like_symbol_operand(text: str) -> bool:
    if not text:
        return False
    if text.startswith(("i", "u")) and ":" in text:
        return False
    return text[0].isalpha() or text[0] in {"_", "%"}


def _direct_successor_label(terminator: object) -> str | None:
    if isinstance(terminator, BranchOp):
        return terminator.target
    return None


def _target_reaches_target(function: Function, start: str, target: str) -> bool:
    block_map = function.block_map()

    def successors(label: str) -> Iterator[str]:
        successor = _direct_successor_label(block_map[label].terminator)
        if successor is not None:
            yield successor

    return target in set(breadth_first_walk([start], successors))


def _loop_exit_target(terminator: object) -> str | None:
    if isinstance(terminator, CondBranchOp):
        return terminator.false_target
    return None


def _loop_defined_names(function: Function, loop: _LoopSummary) -> set[str]:
    block_map = function.block_map()
    names: set[str] = set()
    for label in loop.body:
        block = block_map.get(label)
        if block is None:
            continue
        for instruction in getattr(block, "instructions", []):
            dest = _instruction_dest(instruction)
            if dest is not None:
                names.add(dest)
    return names


def _hier_edge(source: str, target: str, attributes: dict[str, AttributeValue] | None = None) -> SGEdge:
    merged = {"hierarchy": True}
    if attributes is not None:
        merged.update(attributes)
    return SGEdge("seq", source, target, merged)


def _format_operand_name(value: object) -> str:
    if hasattr(value, "name"):
        return str(getattr(value, "name"))
    return str(value)
