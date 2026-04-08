"""Quick Verilator-backed execution helpers for wrapped µglIR designs."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from uhls.backend.hls.lib import parse_component_spec, resolve_component_definition
from uhls.backend.hls.uglir import UGLIRAddressMap, UGLIRAddressMapEntry, UGLIRDesign
from uhls.interpreter import CallHookResult

from .lower import lower_uglir_to_rtl


@dataclass(frozen=True)
class _ScalarRegister:
    name: str
    offset: int
    access: str
    type: str


@dataclass(frozen=True)
class _MemoryWindow:
    name: str
    offset: int
    span: int
    depth: int
    word_bytes: int


@dataclass(frozen=True)
class _VerilogSupportAssets:
    sources: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    defines: tuple[str, ...]


class VerilatorWrappedRunner:
    """One compile-once runner for one wrapped µglIR design."""

    def __init__(
        self,
        design: UGLIRDesign,
        *,
        component_library: dict[str, dict[str, object]] | None = None,
        library_root: Path | None = None,
        reset: str = "sync+active_hi",
    ) -> None:
        if shutil.which("verilator") is None:
            raise ValueError("verilator is not available on PATH")
        if not design.address_maps:
            raise ValueError("Verilator run backend currently requires wrapped µglIR with one address_map")
        self.design = design
        self.component_library = component_library
        self.library_root = Path.cwd() if library_root is None else library_root
        self.address_map = design.address_maps[0]
        self.protocol = self.address_map.name.strip().lower()
        if self.protocol not in {"wishbone", "obi"}:
            raise ValueError(f"Verilator run backend currently supports wrapped wishbone/obi µglIR, got '{self.address_map.name}'")
        self.reset = reset
        self.scalar_inputs, self.scalar_outputs, self.memory_windows = _collect_mmio(self.address_map)
        self.result_register = next((register for register in self.scalar_outputs if register.name == "result"), None)
        self.support_assets = _collect_verilog_support_assets(design, component_library, self.library_root)
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._binary_path: Path | None = None

    def close(self) -> None:
        """Release any build artifacts owned by this runner."""
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
            self._binary_path = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def invoke(self, scalar_arguments: Mapping[str, int], arrays_by_alias: Mapping[str, Sequence[int]]) -> CallHookResult:
        self._ensure_built()
        assert self._tempdir is not None
        assert self._binary_path is not None
        workdir = Path(self._tempdir.name)
        request_path = workdir / "request.txt"
        response_path = workdir / "response.txt"
        request_path.write_text(_encode_request(scalar_arguments, arrays_by_alias), encoding="utf-8")
        try:
            completed = subprocess.run(
                [str(self._binary_path), str(request_path), str(response_path)],
                cwd=workdir,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            stdout = exc.stdout.strip()
            detail = stderr or stdout or str(exc)
            raise ValueError(f"Verilator execution failed: {detail}") from exc
        response = _decode_response(response_path.read_text(encoding="utf-8"))
        return CallHookResult(
            return_value=response["return_value"],
            updated_arrays=response["arrays"],
            metadata={"backend": "verilog", "cycles": response["cycles"]},
        )

    def _ensure_built(self) -> None:
        if self._binary_path is not None:
            return
        self._tempdir = tempfile.TemporaryDirectory(prefix="uhls_verilator_")
        workdir = Path(self._tempdir.name)
        verilog_path = workdir / f"{self.design.name}.v"
        harness_path = workdir / f"{self.design.name}_sim.cpp"
        verilog_path.write_text(lower_uglir_to_rtl(self.design, hdl="verilog", reset=self.reset), encoding="utf-8")
        harness_path.write_text(_emit_verilator_harness(self.design, self.address_map, self.scalar_inputs, self.scalar_outputs, self.memory_windows), encoding="utf-8")
        try:
            subprocess.run(
                [
                    "verilator",
                    "--cc",
                    "-Wno-WIDTHTRUNC",
                    "-Wno-WIDTHEXPAND",
                    str(verilog_path),
                    *[f"-I{path}" for path in self.support_assets.include_dirs],
                    *[f"-D{define}" for define in self.support_assets.defines],
                    *[str(path) for path in self.support_assets.sources],
                    "--exe",
                    str(harness_path),
                    "--build",
                    "--top-module",
                    self.design.name,
                    "-CFLAGS",
                    "-std=c++17",
                ],
                cwd=workdir,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            stdout = exc.stdout.strip()
            detail = stderr or stdout or str(exc)
            raise ValueError(f"Verilator build failed: {detail}") from exc
        binary = workdir / "obj_dir" / f"V{self.design.name}"
        if not binary.is_file():
            raise ValueError(f"Verilator build did not produce expected binary '{binary}'")
        self._binary_path = binary


def _collect_mmio(address_map: UGLIRAddressMap) -> tuple[list[_ScalarRegister], list[_ScalarRegister], list[_MemoryWindow]]:
    scalar_inputs: list[_ScalarRegister] = []
    scalar_outputs: list[_ScalarRegister] = []
    memory_windows: list[_MemoryWindow] = []
    for entry in address_map.entries:
        if entry.kind == "register":
            register = _ScalarRegister(
                name=entry.name,
                offset=_parse_u32_like(entry.attributes["offset"]),
                access=str(entry.attributes.get("access", "rw")),
                type=str(entry.attributes.get("type", "u32")),
            )
            if register.name == "control_status":
                continue
            if register.access.startswith("rw"):
                scalar_inputs.append(register)
            if register.access.startswith("ro") or register.access.startswith("rw"):
                scalar_outputs.append(register)
        elif entry.kind == "memory":
            word_t = str(entry.attributes.get("word_t", "u32"))
            memory_windows.append(
                _MemoryWindow(
                    name=entry.name,
                    offset=_parse_u32_like(entry.attributes["offset"]),
                    span=_parse_u32_like(entry.attributes["span"]),
                    depth=int(entry.attributes.get("depth", 0)),
                    word_bytes=_type_size_bytes(word_t),
                )
            )
    return scalar_inputs, scalar_outputs, memory_windows


def _parse_u32_like(value: object) -> int:
    text = str(value).strip().lower()
    if text.startswith("32'h") and "_" in text:
        head, tail = text[4:].split("_", 1)
        return (int(head, 16) << 16) | int(tail, 16)
    raise ValueError(f"unsupported u32 literal '{value}'")


def _type_size_bytes(type_hint: str) -> int:
    if len(type_hint) >= 2 and type_hint[0] in {"i", "u"} and type_hint[1:].isdigit():
        width = int(type_hint[1:])
        return max((width + 7) // 8, 1)
    return 4


def _encode_request(scalars: Mapping[str, int], arrays: Mapping[str, Sequence[int]]) -> str:
    lines: list[str] = []
    for name in sorted(scalars):
        lines.append(f"scalar {name} {int(scalars[name])}")
    for name in sorted(arrays):
        values = [int(value) for value in arrays[name]]
        payload = " ".join(str(value) for value in values)
        lines.append(f"array {name} {len(values)}{(' ' + payload) if payload else ''}")
    return "\n".join(lines) + ("\n" if lines else "")


def _decode_response(text: str) -> dict[str, object]:
    result: dict[str, object] = {"return_value": None, "cycles": 0, "arrays": {}}
    arrays: dict[str, list[int]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "return":
            result["return_value"] = int(parts[1])
        elif parts[0] == "cycles":
            result["cycles"] = int(parts[1])
        elif parts[0] == "array":
            name = parts[1]
            count = int(parts[2])
            arrays[name] = [int(value) for value in parts[3 : 3 + count]]
    result["arrays"] = arrays
    return result


def _emit_verilator_harness(
    design: UGLIRDesign,
    address_map: UGLIRAddressMap,
    scalar_inputs: list[_ScalarRegister],
    scalar_outputs: list[_ScalarRegister],
    memory_windows: list[_MemoryWindow],
) -> str:
    top = design.name
    protocol = address_map.name.strip().lower()
    has_wb_err = any(port.name == "wb_err_o" for port in design.outputs)
    result_register = next((register for register in scalar_outputs if register.name == "result"), None)
    scalar_write_lines = [
        f'  if (scalars.count("{register.name}")) sim.write32({_cpp_hex(register.offset)}, (uint32_t)scalars.at("{register.name}"));'
        for register in scalar_inputs
    ]
    array_write_lines = []
    array_read_lines = []
    for memory in memory_windows:
        array_write_lines.extend(
            [
                f'  if (arrays.count("{memory.name}")) {{',
                f'    const auto &values = arrays.at("{memory.name}");',
                f"    for (size_t index = 0; index < values.size() && index < {memory.depth}u; ++index) {{",
                f"      sim.write32({_cpp_hex(memory.offset)} + (uint32_t)(index * {memory.word_bytes}u), values[index]);",
                "    }",
                "  }",
            ]
        )
        array_read_lines.extend(
            [
                f'  out << "array {memory.name} {memory.depth}";',
                f"  for (size_t index = 0; index < {memory.depth}u; ++index) {{",
                f"    out << \" \" << sim.read32({_cpp_hex(memory.offset)} + (uint32_t)(index * {memory.word_bytes}u));",
                "  }",
                '  out << "\\n";',
            ]
        )
    if protocol == "wishbone":
        bus_helpers = _emit_wishbone_helpers(has_wb_err)
    else:
        bus_helpers = _emit_obi_helpers()
    result_lines = []
    if result_register is not None:
        result_lines.append(f'  out << "return " << sim.read32({_cpp_hex(result_register.offset)}) << "\\n";')
    result_lines.extend(array_read_lines)
    return f'''#include "V{top}.h"
#include "verilated.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Request {{
  std::unordered_map<std::string, uint32_t> scalars;
  std::unordered_map<std::string, std::vector<uint32_t>> arrays;
}};

static Request parse_request(const char *path) {{
  Request request;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open request file");
  std::string line;
  while (std::getline(in, line)) {{
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string kind;
    ss >> kind;
    if (kind == "scalar") {{
      std::string name;
      long long value = 0;
      ss >> name >> value;
      request.scalars[name] = (uint32_t)value;
    }} else if (kind == "array") {{
      std::string name;
      size_t count = 0;
      ss >> name >> count;
      auto &values = request.arrays[name];
      values.reserve(count);
      for (size_t i = 0; i < count; ++i) {{
        long long value = 0;
        ss >> value;
        values.push_back((uint32_t)value);
      }}
    }}
  }}
  return request;
}}

struct Sim {{
  V{top} dut;
  uint64_t cycles = 0;

  void eval() {{
    dut.eval();
  }}

  void tick() {{
    dut.clk = 0;
    dut.eval();
    dut.clk = 1;
    dut.eval();
    ++cycles;
    dut.clk = 0;
    dut.eval();
  }}

  void reset() {{
    dut.clk = 0;
    dut.rst = 1;
    zero_bus();
    tick();
    tick();
    dut.rst = 0;
    tick();
  }}

  void zero_bus() {{
{_emit_zero_bus(protocol)}
  }}

{bus_helpers}
}};

int main(int argc, char **argv) {{
  Verilated::commandArgs(argc, argv);
  if (argc != 3) {{
    return 2;
  }}
  const Request request = parse_request(argv[1]);
  Sim sim;
  sim.reset();
  const auto &scalars = request.scalars;
  const auto &arrays = request.arrays;
{chr(10).join(scalar_write_lines) if scalar_write_lines else ""}
{chr(10).join(array_write_lines) if array_write_lines else ""}
  sim.write32(0x0u, 1u);
  while ((sim.read32(0x0u) & 0x1u) == 0u) {{
  }}
  std::ofstream out(argv[2]);
  if (!out) throw std::runtime_error("failed to open response file");
  out << "cycles " << sim.cycles << "\\n";
{chr(10).join(result_lines) if result_lines else ""}
  return 0;
}}
'''


def _emit_zero_bus(protocol: str) -> str:
    if protocol == "wishbone":
        return "\n".join(
            [
                "    dut.wb_cyc_i = 0;",
                "    dut.wb_stb_i = 0;",
                "    dut.wb_we_i = 0;",
                "    dut.wb_adr_i = 0;",
                "    dut.wb_dat_i = 0;",
                "    dut.wb_sel_i = 0;",
            ]
        )
    return "\n".join(
        [
            "    dut.obi_req_i = 0;",
            "    dut.obi_addr_i = 0;",
            "    dut.obi_we_i = 0;",
            "    dut.obi_be_i = 0;",
            "    dut.obi_wdata_i = 0;",
            "    dut.obi_rready_i = 1;",
        ]
    )


def _emit_wishbone_helpers(has_err: bool) -> str:
    err_check = '      if (dut.wb_err_o) throw std::runtime_error("wishbone error response");\n' if has_err else ""
    return f'''  void write32(uint32_t addr, uint32_t value) {{
    dut.wb_adr_i = addr;
    dut.wb_dat_i = value;
    dut.wb_sel_i = 0xf;
    dut.wb_we_i = 1;
    dut.wb_cyc_i = 1;
    dut.wb_stb_i = 1;
    dut.eval();
{err_check}    tick();
    zero_bus();
    dut.eval();
  }}

  uint32_t read32(uint32_t addr) {{
    dut.wb_adr_i = addr;
    dut.wb_sel_i = 0xf;
    dut.wb_we_i = 0;
    dut.wb_cyc_i = 1;
    dut.wb_stb_i = 1;
    dut.eval();
{err_check}    const uint32_t value = dut.wb_dat_o;
    tick();
    zero_bus();
    dut.eval();
    return value;
  }}
'''


def _emit_obi_helpers() -> str:
    return '''  void write32(uint32_t addr, uint32_t value) {
    dut.obi_addr_i = addr;
    dut.obi_wdata_i = value;
    dut.obi_be_i = 0xf;
    dut.obi_we_i = 1;
    dut.obi_req_i = 1;
    dut.obi_rready_i = 1;
    while (true) {
      dut.eval();
      if (dut.obi_gnt_o) {
        tick();
        break;
      }
      tick();
    }
    dut.obi_req_i = 0;
    dut.obi_we_i = 0;
    dut.eval();
    while (!dut.obi_rvalid_o) {
      tick();
    }
    tick();
    zero_bus();
    dut.eval();
  }

  uint32_t read32(uint32_t addr) {
    dut.obi_addr_i = addr;
    dut.obi_be_i = 0xf;
    dut.obi_we_i = 0;
    dut.obi_req_i = 1;
    dut.obi_rready_i = 1;
    while (true) {
      dut.eval();
      if (dut.obi_gnt_o) {
        tick();
        break;
      }
      tick();
    }
    dut.obi_req_i = 0;
    dut.eval();
    while (!dut.obi_rvalid_o) {
      tick();
    }
    dut.eval();
    const uint32_t value = dut.obi_rdata_o;
    tick();
    zero_bus();
    dut.eval();
    return value;
  }
'''


def _cpp_hex(value: int) -> str:
    return f"0x{value:x}u"


def _collect_verilog_support_assets(
    design: UGLIRDesign,
    component_library: dict[str, dict[str, object]] | None,
    library_root: Path,
) -> _VerilogSupportAssets:
    sources: list[Path] = []
    include_dirs: list[Path] = []
    defines: list[str] = []
    seen_sources: set[Path] = set()
    seen_include_dirs: set[Path] = set()
    seen_defines: set[str] = set()
    for resource in design.resources:
        if resource.kind != "inst":
            continue
        if component_library is None:
            raise ValueError(
                f"Verilator run backend requires --resources for instance '{resource.id}: {resource.value}'"
            )
        base_name, component = _resolve_instance_component(component_library, resource.value)
        hdl = component.get("hdl")
        if not isinstance(hdl, dict):
            raise ValueError(
                f"component '{base_name}' used by instance '{resource.id}' is missing required 'hdl' linkage"
            )
        language = str(hdl.get("language", "")).strip().lower()
        if language != "verilog":
            raise ValueError(
                f"Verilator run backend requires verilog-linked components; '{base_name}' uses hdl.language={language!r}"
            )
        source_texts = _collect_hdl_source_texts(hdl)
        if not source_texts:
            raise ValueError(
                f"component '{base_name}' used by instance '{resource.id}' must define hdl.source or hdl.sources in the component library"
            )
        for source_text in source_texts:
            source_path = Path(source_text)
            if not source_path.is_absolute():
                source_path = (library_root / source_path).resolve()
            if not source_path.is_file():
                raise ValueError(
                    f"component '{base_name}' HDL source '{source_path}' for instance '{resource.id}' does not exist"
                )
            if source_path not in seen_sources:
                seen_sources.add(source_path)
                sources.append(source_path)
        for include_dir_text in _collect_hdl_include_dirs(hdl):
            include_dir = Path(include_dir_text)
            if not include_dir.is_absolute():
                include_dir = (library_root / include_dir).resolve()
            if not include_dir.is_dir():
                raise ValueError(
                    f"component '{base_name}' HDL include dir '{include_dir}' for instance '{resource.id}' does not exist"
                )
            if include_dir not in seen_include_dirs:
                seen_include_dirs.add(include_dir)
                include_dirs.append(include_dir)
        for define in _collect_hdl_defines(hdl):
            if define not in seen_defines:
                seen_defines.add(define)
                defines.append(define)
    return _VerilogSupportAssets(tuple(sources), tuple(include_dirs), tuple(defines))


def _collect_hdl_source_texts(hdl: Mapping[str, object]) -> list[str]:
    source_texts: list[str] = []
    source = hdl.get("source")
    if isinstance(source, str) and source.strip():
        source_texts.append(source.strip())
    sources = hdl.get("sources")
    if isinstance(sources, list):
        for entry in sources:
            if isinstance(entry, str) and entry.strip():
                source_texts.append(entry.strip())
    return source_texts


def _collect_hdl_include_dirs(hdl: Mapping[str, object]) -> list[str]:
    include_dirs = hdl.get("include_dirs")
    if not isinstance(include_dirs, list):
        return []
    return [entry.strip() for entry in include_dirs if isinstance(entry, str) and entry.strip()]


def _collect_hdl_defines(hdl: Mapping[str, object]) -> list[str]:
    defines = hdl.get("defines")
    if not isinstance(defines, list):
        return []
    return [entry.strip() for entry in defines if isinstance(entry, str) and entry.strip()]


def _resolve_instance_component(
    component_library: dict[str, dict[str, object]],
    instance_spec: str,
) -> tuple[str, dict[str, object]]:
    try:
        base_name, _params, component = resolve_component_definition(component_library, instance_spec)
        return base_name, component
    except ValueError:
        module_name, _params = parse_component_spec(instance_spec)
        for component_name, component in component_library.items():
            if not isinstance(component, dict):
                continue
            hdl = component.get("hdl")
            if not isinstance(hdl, dict):
                continue
            if str(hdl.get("module", "")) == module_name:
                return str(component_name), component
        raise


__all__ = ["VerilatorWrappedRunner"]
