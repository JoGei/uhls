from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from uhls.backend.hls.uhir import create_builtin_gopt_pass, lower_module_to_seq, parse_uhir, run_gopt_passes
from uhls.interpreter import run_uhir, run_uir
from uhls.middleend.uir import BinaryOp, Block as UIRBlock, CallOp, Function as UIRFunction, Module, Parameter, ReturnOp, parse_module


@dataclass
class Param:
    """Minimal duck-typed parameter fixture used by the interpreter tests."""

    name: str
    type: str


@dataclass
class Inst:
    """Minimal duck-typed instruction fixture.

    The real IR classes do not exist yet, so the tests use this compact
    structure to exercise the interpreter's flexible field lookup logic.
    """

    opcode: str
    dest: str | None = None
    type: str | None = None
    operands: list[object] = field(default_factory=list)
    value: object | None = None
    array: str | None = None
    index: object | None = None
    incoming: list[tuple[str, object]] | None = None
    cond: object | None = None
    true_target: str | None = None
    false_target: str | None = None
    target: str | None = None
    callee: str | None = None
    arg_count: int | None = None


@dataclass
class Block:
    """Minimal basic-block fixture with explicit body and terminator."""

    label: str
    instructions: list[Inst]
    terminator: Inst


@dataclass
class Function:
    """Minimal function fixture matching the interpreter's expected shape."""

    name: str
    params: list[Param]
    blocks: list[Block]
    return_type: str
    entry: str = "entry"


class InterpreterTests(unittest.TestCase):
    """Behavioral tests for the shared µIR interpreter runtime."""

    def _build_select_function(self) -> Function:
        """Build a small CFG with branch-specific loads and a merge."""

        return Function(
            name="uir_select",
            params=[Param("sel", "i1"), Param("idx", "u8")],
            return_type="i32",
            blocks=[
                Block(
                    label="entry",
                    instructions=[],
                    terminator=Inst(
                        opcode="cbr",
                        cond="sel",
                        true_target="then_blk",
                        false_target="else_blk",
                    ),
                ),
                Block(
                    label="then_blk",
                    instructions=[
                        Inst(opcode="load", dest="tmp", type="i32", array="A", index="idx"),
                        Inst(opcode="add", dest="x", type="i32", operands=["tmp", 1]),
                    ],
                    terminator=Inst(opcode="br", target="merge_blk"),
                ),
                Block(
                    label="else_blk",
                    instructions=[
                        Inst(opcode="load", dest="tmp", type="i32", array="B", index="idx"),
                        Inst(opcode="sub", dest="x", type="i32", operands=["tmp", 1]),
                    ],
                    terminator=Inst(opcode="br", target="merge_blk"),
                ),
                Block(
                    label="merge_blk",
                    instructions=[
                        Inst(opcode="store", array="OUT", index=0, value="x"),
                    ],
                    terminator=Inst(opcode="ret", value="x"),
                ),
            ],
        )

    def test_uir_executes_then_branch_and_memory(self) -> None:
        """UIR should load from the then-branch array, store, and return that value."""

        # A truthy selector should enter then_blk, load A[1] == 20, and store
        # the incremented result to OUT before returning it.
        result = run_uir(
            self._build_select_function(),
            {"sel": 1, "idx": 1},
            arrays={
                "A": {"data": [10, 20], "element_type": "i32"},
                "B": {"data": [30, 40], "element_type": "i32"},
                "OUT": {"size": 1, "element_type": "i32"},
            },
        )

        self.assertEqual(result.return_value, 21)
        self.assertEqual(result.state.memory.snapshot()["OUT"], [21])

    def test_uir_executes_else_branch_and_memory(self) -> None:
        """UIR should load from the else-branch array, store, and return that value."""

        # A falsey selector should enter else_blk, load B[0] == 30, and store
        # the decremented result to OUT before returning it.
        result = run_uir(
            self._build_select_function(),
            {"sel": 0, "idx": 0},
            arrays={
                "A": {"data": [10, 20], "element_type": "i32"},
                "B": {"data": [30, 40], "element_type": "i32"},
                "OUT": {"size": 1, "element_type": "i32"},
            },
        )

        self.assertEqual(result.return_value, 29)
        self.assertEqual(result.state.memory.snapshot()["OUT"], [29])

    def test_uir_emits_branch_trace(self) -> None:
        """UIR should emit branch trace information when tracing is enabled."""

        # Tracing should record the conditional branch decision when the CFG is
        # executed with branch tracing enabled.
        result = run_uir(
            self._build_select_function(),
            {"sel": 1, "idx": 1},
            arrays={
                "A": {"data": [10, 20], "element_type": "i32"},
                "B": {"data": [30, 40], "element_type": "i32"},
                "OUT": {"size": 1, "element_type": "i32"},
            },
            trace=True,
        )

        self.assertTrue(any(event.kind == "branch" for event in result.state.trace))

    def test_uir_resolves_phi_on_block_entry(self) -> None:
        """UIR should resolve leading phi nodes using the predecessor block on entry."""

        # This SSA-shaped CFG computes a different definition of x on each side
        # of the branch and joins them with a phi in the merge block.
        function = Function(
            name="uir_phi_select",
            params=[Param("sel_0", "i1"), Param("a_0", "i32"), Param("b_0", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    label="entry",
                    instructions=[],
                    terminator=Inst(
                        opcode="cbr",
                        cond="sel_0",
                        true_target="then_blk",
                        false_target="else_blk",
                    ),
                ),
                Block(
                    label="then_blk",
                    instructions=[
                        Inst(opcode="add", dest="x_1", type="i32", operands=["a_0", 3]),
                    ],
                    terminator=Inst(opcode="br", target="merge_blk"),
                ),
                Block(
                    label="else_blk",
                    instructions=[
                        Inst(opcode="sub", dest="x_2", type="i32", operands=["b_0", 5]),
                    ],
                    terminator=Inst(opcode="br", target="merge_blk"),
                ),
                Block(
                    label="merge_blk",
                    instructions=[
                        Inst(
                            opcode="phi",
                            dest="x_3",
                            type="i32",
                            incoming=[("then_blk", "x_1"), ("else_blk", "x_2")],
                        ),
                        Inst(opcode="mul", dest="y_0", type="i32", operands=["x_3", 2]),
                    ],
                    terminator=Inst(opcode="ret", value="y_0"),
                ),
            ],
        )

        # Run both control-flow paths so the test checks phi selection from each
        # predecessor edge rather than only one happy path.
        true_result = run_uir(function, {"sel_0": 1, "a_0": 7, "b_0": 100}, trace=True)
        false_result = run_uir(function, {"sel_0": 0, "a_0": 7, "b_0": 100})

        # then_blk: (7 + 3) * 2 == 20
        # else_blk: (100 - 5) * 2 == 190
        self.assertEqual(true_result.return_value, 20)
        self.assertEqual(false_result.return_value, 190)
        self.assertTrue(any(event.kind == "phi" for event in true_result.state.trace))

    def test_width_normalization_wraps_integer_results(self) -> None:
        """Arithmetic results should wrap to the destination integer type."""

        # i8 max value plus one should wrap into the negative range when the
        # interpreter normalizes the result to signed 8-bit semantics.
        function = Function(
            name="wrap",
            params=[],
            return_type="i8",
            blocks=[
                Block(
                    label="entry",
                    instructions=[
                        Inst(opcode="const", dest="x", type="i8", value=127),
                        Inst(opcode="add", dest="y", type="i8", operands=["x", 1]),
                    ],
                    terminator=Inst(opcode="ret", value="y"),
                )
            ],
        )

        result = run_uir(function, [])

        # 127 + 1 in i8 becomes 128, which is represented as -128 in signed i8.
        # In the interpreter's fixed-width i8 semantics, 127 + 1 wraps to -128.
        self.assertEqual(result.return_value, -128)

    def test_store_uses_array_element_type_for_normalization(self) -> None:
        """Stores should normalize values to the destination array's element type."""

        # The scalar is intentionally wider than the destination array element.
        # The store should truncate to u8, and the following load should observe
        # the truncated value from memory.
        function = Function(
            name="store_wrap",
            params=[],
            return_type="u8",
            blocks=[
                Block(
                    label="entry",
                    instructions=[
                        Inst(opcode="const", dest="x", type="u16", value=(0x100 + 2)),
                        Inst(opcode="store", array="A", index=0, value="x"),
                        Inst(opcode="load", dest="y", type="u8", array="A", index=0),
                    ],
                    terminator=Inst(opcode="ret", value="y"),
                )
            ],
        )

        result = run_uir(
            function,
            [],
            arrays={"A": {"size": 1, "element_type": "u8"}},
        )

        # 258 mod 256 == 2, so both memory and the loaded return value should be 2.
        self.assertEqual(result.return_value, 2)
        self.assertEqual(result.state.memory.snapshot()["A"], [2])

    def test_signed_division_and_modulo_truncate_toward_zero(self) -> None:
        """Signed div/mod should follow C-style truncation toward zero."""

        function = Function(
            name="divmod",
            params=[],
            return_type="i32",
            blocks=[
                Block(
                    label="entry",
                    instructions=[
                        Inst(opcode="const", dest="x", type="i32", value=-7),
                        Inst(opcode="div", dest="q", type="i32", operands=["x", 3]),
                        Inst(opcode="mod", dest="r", type="i32", operands=["x", 3]),
                        Inst(opcode="add", dest="y", type="i32", operands=["q", "r"]),
                    ],
                    terminator=Inst(opcode="ret", value="y"),
                )
            ],
        )

        result = run_uir(function, [])

        self.assertEqual(result.return_value, -3)

    def test_direct_call_executes_module_local_callee(self) -> None:
        """Canonical direct calls should execute a sibling function."""

        callee = Function(
            name="mac",
            params=[Param("a", "i32"), Param("b", "i32"), Param("c", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    label="entry",
                    instructions=[
                        Inst(opcode="mul", dest="prod", type="i32", operands=["a", "b"]),
                        Inst(opcode="add", dest="sum", type="i32", operands=["prod", "c"]),
                    ],
                    terminator=Inst(opcode="ret", value="sum"),
                )
            ],
        )
        caller = Function(
            name="caller",
            params=[Param("x", "i32"), Param("y", "i32"), Param("z", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    label="entry",
                    instructions=[
                        Inst(
                            opcode="call",
                            callee="mac",
                            operands=["x", "y", "z"],
                            dest="out",
                            type="i32",
                        )
                    ],
                    terminator=Inst(opcode="ret", value="out"),
                )
            ],
        )
        module = Module(functions=[callee, caller])

        result = run_uir(caller, {"x": 2, "y": 3, "z": 4}, module=module)

        self.assertEqual(result.return_value, 10)

    def test_lowered_param_call_executes_module_local_callee(self) -> None:
        """Lowered ``param`` plus ``call foo, N`` should execute in argument order."""

        callee = Function(
            name="mix",
            params=[Param("a", "i32"), Param("b", "i32"), Param("c", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    label="entry",
                    instructions=[
                        Inst(opcode="sub", dest="d", type="i32", operands=["a", "b"]),
                        Inst(opcode="add", dest="e", type="i32", operands=["d", "c"]),
                    ],
                    terminator=Inst(opcode="ret", value="e"),
                )
            ],
        )
        caller = Function(
            name="caller",
            params=[Param("x", "i32"), Param("y", "i32"), Param("z", "i32")],
            return_type="i32",
            blocks=[
                Block(
                    label="entry",
                    instructions=[
                        Inst(opcode="param", index=1, value="x"),
                        Inst(opcode="param", index=2, value="y"),
                        Inst(opcode="param", index=3, value="z"),
                        Inst(opcode="call", callee="mix", arg_count=3, dest="out", type="i32"),
                    ],
                    terminator=Inst(opcode="ret", value="out"),
                )
            ],
        )
        module = Module(functions=[callee, caller])

        result = run_uir(caller, {"x": 9, "y": 4, "z": 2}, module=module)

        self.assertEqual(result.return_value, 7)

    def test_uhir_executes_branch_based_seq_loop(self) -> None:
        """Seq-stage µhIR should execute lowered branch-based loops directly."""

        module = parse_module(
            """
            func dot4(A:i32[]) -> i32

            block entry:
                br for_header_1


            block for_header_1:
                i_1:i32 = phi(entry: 0:i32, for_body_2: t2_0)
                sum_1:i32 = phi(entry: 0:i32, for_body_2: t1_0)
                t0_0:i1 = lt i_1, 4:i32
                cbr t0_0, for_body_2, for_exit_4


            block for_body_2:
                t0_1:i32 = load A[i_1]
                t1_0:i32 = add sum_1, t0_1
                t2_0:i32 = add i_1, 1:i32
                br for_header_1


            block for_exit_4:
                ret sum_1
            """
        )

        design = lower_module_to_seq(module, top="dot4")
        result = run_uhir(
            design,
            arrays={"A": {"data": [1, 2, 3, 4], "element_type": "i32"}},
        )

        self.assertEqual(result.return_value, 10)

    def test_uhir_executes_parsed_seq_arithmetic_and_memory(self) -> None:
        """Seq-stage µhIR text should run directly through the µhIR interpreter."""

        design = parse_uhir(
            """
            design direct_seq
            stage seq
            input  A : memref<i32, 2>
            input  x : i32
            output result : i32
            const  K = 3 : i32

            region proc_direct_seq kind=procedure {
              node v0 = nop role=source
              node v1 = load A[1:i32] : i32
              node v2 = add v1, x : i32
              node v3 = add v2, K : i32
              node v4 = store A[0:i32], v3
              node v5 = ret v3
              node v6 = nop role=sink

              edge data v0 -> v1
              edge data v1 -> v2
              edge data v2 -> v3
              edge data v3 -> v4
              edge data v4 -> v5
              edge data v5 -> v6
            }
            """
        )

        result = run_uhir(
            design,
            arguments={"x": 5},
            arrays={"A": {"data": [0, 7], "element_type": "i32"}},
        )

        self.assertEqual(result.return_value, 15)
        self.assertEqual(result.state.memory.snapshot()["A"], [15, 7])

    def test_uhir_executes_explicit_static_loop_after_gopt(self) -> None:
        """Seq-stage µhIR should execute explicit/static loop dialect forms too."""

        module = parse_module(
            """
            func dot4(A:i32[]) -> i32

            block entry:
                br for_header_1


            block for_header_1:
                i_1:i32 = phi(entry: 0:i32, for_body_2: t2_0)
                sum_1:i32 = phi(entry: 0:i32, for_body_2: t1_0)
                t0_0:i1 = lt i_1, 4:i32
                cbr t0_0, for_body_2, for_exit_4


            block for_body_2:
                t0_1:i32 = load A[i_1]
                t1_0:i32 = add sum_1, t0_1
                t2_0:i32 = add i_1, 1:i32
                br for_header_1


            block for_exit_4:
                ret sum_1
            """
        )

        design = run_gopt_passes(
            lower_module_to_seq(module, top="dot4"),
            [
                create_builtin_gopt_pass("infer_loops"),
                create_builtin_gopt_pass("translate_loop_dialect"),
                create_builtin_gopt_pass("infer_static"),
                create_builtin_gopt_pass("simplify_static_control"),
            ],
        )
        result = run_uhir(
            design,
            arrays={"A": {"data": [1, 2, 3, 4], "element_type": "i32"}},
        )

        self.assertEqual(result.return_value, 10)

    def test_uhir_executes_interprocedural_call_with_inferred_live_ins(self) -> None:
        """Seq-stage µhIR should infer callee live-ins for procedure calls."""

        callee = UIRFunction(
            name="callee",
            params=[Parameter("x", "i32")],
            return_type="i32",
            blocks=[UIRBlock("entry", instructions=[BinaryOp("add", "y", "i32", "x", 1)], terminator=ReturnOp("y"))],
        )
        caller = UIRFunction(
            name="caller",
            params=[Parameter("z", "i32")],
            return_type="i32",
            blocks=[UIRBlock("entry", instructions=[CallOp("callee", ["z"], dest="r", type="i32")], terminator=ReturnOp("r"))],
        )

        design = lower_module_to_seq(Module(functions=[callee, caller]), top="caller")
        result = run_uhir(design, arguments={"z": 7})

        self.assertEqual(result.return_value, 8)


if __name__ == "__main__":
    unittest.main()
