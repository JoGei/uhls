from __future__ import annotations

import unittest
from pathlib import Path

from uhls.frontend import analyze_program, lower_source_to_uir, parse_program, tokenize
from uhls.interpreter import run_uir
from uhls.middleend.uir import Function, pretty, verify_module
from uhls.middleend.passes.opt import inline_calls


DOT4_SOURCE = """
int32_t dot4(int32_t A[4], int32_t B[4]) {
    int32_t i;
    int32_t sum = 0;

    for (i = 0; i < 4; i = i + 1) {
        sum = sum + A[i] * B[i];
    }

    return sum;
}
"""

DOT4_RELU_EXAMPLE = Path("examples/dot4_relu/dot4_relu.c")
DOT4_I8_PACKED_EXAMPLE = Path("examples/dot4_i8_i32_relu_packed/dot4_i8_i32_relu_packed.c")


class FrontendTests(unittest.TestCase):
    """End-to-end tests for the µC frontend package."""

    def test_frontend_parses_analyzes_and_lowers_dot4(self) -> None:
        program = parse_program(DOT4_SOURCE)
        info = analyze_program(program)
        module = lower_source_to_uir(DOT4_SOURCE)

        self.assertEqual(len(program.functions), 1)
        self.assertEqual(program.functions[0].name, "dot4")
        self.assertIn("dot4", info.functions)
        verify_module(module)
        self.assertIn("func dot4(A:i32[4], B:i32[4]) -> i32", pretty(module))

    def test_frontend_lowered_dot4_runs_in_interpreter(self) -> None:
        module = lower_source_to_uir(DOT4_SOURCE)
        function = module.functions[0]

        result = run_uir(
            function,
            arrays={
                "A": {"data": [1, 2, 3, 4], "element_type": "i32"},
                "B": {"data": [5, 6, 7, 8], "element_type": "i32"},
            },
        )

        self.assertEqual(result.return_value, 70)

    def test_frontend_contextually_types_uint_literals_in_arithmetic(self) -> None:
        module = lower_source_to_uir(
            """
            uint32_t bump(uint32_t x) {
                return x + 1;
            }
            """
        )

        verify_module(module)
        self.assertIn("1:u32", pretty(module))

    def test_frontend_parses_and_lowers_explicit_scalar_casts(self) -> None:
        module = lower_source_to_uir(
            """
            int32_t widen(int8_t x) {
                return (int32_t)x + 1;
            }
            """
        )

        verify_module(module)
        rendered = pretty(module)
        self.assertIn(" = mov x_0", rendered)
        self.assertEqual(run_uir(module.functions[0], [7]).return_value, 8)

    def test_frontend_tokenizes_and_lowers_hex_integer_literals(self) -> None:
        tokens = tokenize("int32_t f(void) { return 0x2a + 0X1; }")
        self.assertIn("0x2a", [token.text for token in tokens if token.kind == "INT"])
        self.assertIn("0X1", [token.text for token in tokens if token.kind == "INT"])

        module = lower_source_to_uir(
            """
            int32_t hex_add(void) {
                return 0x2a + 0X1;
            }
            """
        )

        verify_module(module)
        self.assertEqual(run_uir(module.functions[0]).return_value, 43)

    def test_frontend_accepts_hex_array_extents(self) -> None:
        module = lower_source_to_uir(
            """
            int32_t first(const int32_t A[0x4]) {
                return A[0x0];
            }
            """
        )

        verify_module(module)
        self.assertEqual(
            run_uir(module.functions[0], arrays={"A": {"data": [7, 8, 9, 10], "element_type": "i32"}}).return_value,
            7,
        )

    def test_frontend_lowers_and_executes_division_and_modulo(self) -> None:
        module = lower_source_to_uir(
            """
            int32_t divmod_accum(int32_t x) {
                return (x / 3) + (x % 3);
            }
            """
        )

        verify_module(module)
        self.assertIn(" = div ", pretty(module))
        self.assertIn(" = mod ", pretty(module))
        self.assertEqual(run_uir(module.functions[0], [10]).return_value, 4)

    def test_frontend_accepts_const_array_params_and_post_increment_in_for_loops(self) -> None:
        module = lower_source_to_uir(
            """
            int32_t sum4(const int32_t A[4]) {
                int32_t i;
                int32_t sum = 0;
                for (i = 0; i < 4; i++) {
                    sum = sum + A[i];
                }
                return sum;
            }
            """
        )

        verify_module(module)
        self.assertEqual(
            run_uir(module.functions[0], arrays={"A": {"data": [1, 2, 3, 4], "element_type": "i32"}}).return_value,
            10,
        )

    def test_frontend_lowers_direct_calls_in_hierarchical_example(self) -> None:
        source = DOT4_RELU_EXAMPLE.read_text(encoding="utf-8")

        program = parse_program(source)
        info = analyze_program(program)
        module = lower_source_to_uir(source)

        self.assertIn("mac", info.functions["dot4_relu"].called_functions)
        verify_module(module)
        self.assertIn("call mac(", pretty(module))

    def test_frontend_accepts_mixed_width_shift_amounts_in_packed_example(self) -> None:
        source = DOT4_I8_PACKED_EXAMPLE.read_text(encoding="utf-8")

        module = lower_source_to_uir(source)

        verify_module(module)
        rendered = pretty(module)
        self.assertIn(" = shr ", rendered)
        self.assertIn(" = mul ", rendered)

    def test_frontend_hierarchical_example_runs_after_inlining(self) -> None:
        source = DOT4_RELU_EXAMPLE.read_text(encoding="utf-8")
        module = lower_source_to_uir(source)
        inlined = inline_calls(module)
        function = inlined.get_function("dot4_relu")
        assert function is not None

        result = run_uir(
            function,
            arrays={
                "A": {"data": [1, 2, 3, 4], "element_type": "i32"},
                "B": {"data": [5, 6, 7, 8], "element_type": "i32"},
            },
        )

        self.assertEqual(result.return_value, 70)

    def test_frontend_inlines_calls_to_callees_with_phi_nodes(self) -> None:
        module = lower_source_to_uir(
            """
            int32_t accum(int32_t n) {
                int32_t i;
                int32_t sum = 0;

                for (i = 0; i < n; i = i + 1) {
                    sum = sum + i;
                }

                return sum;
            }

            int32_t caller(int32_t n) {
                return accum(n);
            }
            """
        )
        inlined = inline_calls(module)
        function = inlined.get_function("caller")
        assert function is not None

        verify_module(inlined)
        self.assertNotIn("call accum(", pretty(inlined))
        self.assertEqual(run_uir(function, [5]).return_value, 10)

    def test_frontend_does_not_inline_calls_made_directly_from_main(self) -> None:
        source = DOT4_RELU_EXAMPLE.read_text(encoding="utf-8")
        module = lower_source_to_uir(source)
        inlined = inline_calls(module)
        main_function = inlined.get_function("main")
        dot4_function = inlined.get_function("dot4_relu")
        assert main_function is not None
        assert dot4_function is not None

        rendered = pretty(inlined)
        verify_module(inlined)
        self.assertIn("call dot4_relu(", rendered)
        self.assertIn("func dot4_relu(", rendered)

    def test_frontend_runs_integrated_testbench_main(self) -> None:
        source = DOT4_RELU_EXAMPLE.read_text(encoding="utf-8")
        module = lower_source_to_uir(source)
        verify_module(module)
        main_function = module.get_function("main")
        assert main_function is not None

        result = run_uir(main_function, module=module)

        self.assertEqual(result.return_value, 0)
        self.assertEqual(result.state.stdout, ["Success!"])

    def test_frontend_renames_print_operands_in_ssa(self) -> None:
        module = lower_source_to_uir(
            """
            int32_t id(int32_t x) {
                return x;
            }

            int32_t main(void) {
                int32_t expected = 9;
                int32_t res;
                res = id(7);
                if (res != expected) {
                    uhls_printf("Unexpected return value: %d", res);
                    return 1;
                }
                return 0;
            }
            """
        )
        verify_module(module)
        main_function = module.get_function("main")
        assert main_function is not None

        result = run_uir(main_function, module=module)

        self.assertEqual(result.return_value, 1)
        self.assertEqual(result.state.stdout, ["Unexpected return value: 7"])

    def test_frontend_uir_construction_inserts_phi_for_branch_merge(self) -> None:
        source = """
        int32_t select(int32_t sel, int32_t a, int32_t b) {
            int32_t x = 0;
            if (sel != 0) {
                x = a + 1;
            } else {
                x = b - 1;
            }
            return x;
        }
        """
        function = lower_source_to_uir(source).functions[0]

        merge_block = next(
            block
            for block in function.blocks
            if block.instructions and block.instructions[0].opcode == "phi"
        )

        self.assertEqual(merge_block.instructions[0].opcode, "phi")
        self.assertEqual(run_uir(function, {"sel_0": 1, "a_0": 7, "b_0": 100}).return_value, 8)
        self.assertEqual(run_uir(function, {"sel_0": 0, "a_0": 7, "b_0": 100}).return_value, 99)
