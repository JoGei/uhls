from __future__ import annotations

import unittest

from uhls.backend.hls.drv import emit_uglir_driver
from uhls.backend.hls.uglir import parse_uglir


class DriverEmissionTests(unittest.TestCase):
    """Coverage for wrapped-µglIR driver emission."""

    def test_emit_c_driver_from_wrapped_uglir(self) -> None:
        design = parse_uglir(
            """design wrapped
stage uglir
input  clk : clock
input  rst : i1
input  obi_req_i : i1
input  obi_addr_i : u32
output obi_gnt_o : i1
output obi_rvalid_o : i1
output obi_rdata_o : u32
address_map obi {
  register control_status offset=32'h0000_0000 access=rw symbol=OBI_REG_CONTROL_STATUS
  register x offset=32'h0000_0100 access=rw symbol=OBI_REG_IN_X type=i32
  register result offset=32'h0000_0200 access=ro symbol=OBI_REG_OUT_RESULT type=i32
  memory A offset=32'h0000_1000 span=32'h0000_0010 access=rw symbol=OBI_MEM_A_BASE word_t=i32 depth=4
}
resources {
  reg x_q : i32
  reg result_q : i32
  mem A_mem_q : i32[4]
}
assign obi_gnt_o = true
assign obi_rvalid_o = false
assign obi_rdata_o = 0:u32
seq clk {
  if rst {
    x_q <= 0:i32
  } else {
    x_q <= x_q
  }
}
"""
        )

        rendered = emit_uglir_driver(design, lang="c")

        self.assertIn("typedef struct {", rendered)
        self.assertIn("uintptr_t base_addr;", rendered)
        self.assertIn("wrapped_drv_t;", rendered)
        self.assertIn("#define WRAPPED_DRV_REG_CONTROL_STATUS_RANGE_START ((uintptr_t)0x0u)", rendered)
        self.assertIn("#define WRAPPED_DRV_REG_CONTROL_STATUS_RANGE_END ((uintptr_t)0x3u)", rendered)
        self.assertIn("#define WRAPPED_DRV_MEM_A_RANGE_END ((uintptr_t)0x100fu)", rendered)
        self.assertIn("#define WRAPPED_DRV_CONTROL_STATUS_DONE_MASK", rendered)
        self.assertIn("static inline void wrapped_drv_init(wrapped_drv_t *inst, uintptr_t base_addr)", rendered)
        self.assertIn("static inline void wrapped_drv_set_x(wrapped_drv_t *inst, int32_t value)", rendered)
        self.assertIn("static inline int32_t wrapped_drv_get_result(const wrapped_drv_t *inst)", rendered)
        self.assertIn("static inline void wrapped_drv_write_a(wrapped_drv_t *inst, size_t index, int32_t value)", rendered)
        self.assertIn("static inline int32_t wrapped_drv_read_a(const wrapped_drv_t *inst, size_t index)", rendered)
        self.assertIn("static inline void wrapped_drv_start_nonblocking(wrapped_drv_t *inst)", rendered)
        self.assertIn("static inline int32_t wrapped_drv_start_blocking(wrapped_drv_t *inst, int32_t value_x)", rendered)

