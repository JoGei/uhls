"""Generic slave-wrapper planning shared across RTL backends."""

from __future__ import annotations

from dataclasses import dataclass

from uhls.backend.hls.uhir.model import UHIRDesign, UHIRPort


@dataclass(frozen=True)
class WrapperMemoryInterface:
    """One raw core memory bundle exposed through a wrapper."""

    base: str
    data_type: str
    addr_type: str | None
    write_type: str | None
    has_write: bool


@dataclass(frozen=True)
class SlaveWrapperPlan:
    """Generic wrapper plan independent of the emitted HDL syntax."""

    protocol: str
    scalar_inputs: tuple[UHIRPort, ...]
    scalar_outputs: tuple[UHIRPort, ...]
    memory_interfaces: tuple[WrapperMemoryInterface, ...]


def plan_slave_wrapper(design: UHIRDesign, protocol: str) -> SlaveWrapperPlan:
    """Build one generic slave-wrapper plan for one uglir core."""
    normalized = protocol.strip().lower()
    if normalized not in {"wishbone", "obi"}:
        raise ValueError(f"unsupported slave wrapper protocol '{protocol}'")
    scalar_inputs, scalar_outputs, memory_interfaces = classify_core_ports(design)
    return SlaveWrapperPlan(
        protocol=normalized,
        scalar_inputs=tuple(scalar_inputs),
        scalar_outputs=tuple(scalar_outputs),
        memory_interfaces=tuple(memory_interfaces),
    )


def classify_core_ports(
    design: UHIRDesign,
) -> tuple[list[UHIRPort], list[UHIRPort], list[WrapperMemoryInterface]]:
    """Split raw core ports into scalar and memory-facing groups."""
    memory_by_base: dict[str, dict[str, object]] = {}
    for port in design.inputs:
        if port.name.endswith("_rdata"):
            base = port.name[:-6]
            memory_by_base.setdefault(
                base,
                {
                    "base": base,
                    "data_type": port.type,
                    "addr_type": None,
                    "write_type": None,
                    "has_write": False,
                },
            )["data_type"] = port.type
    for port in design.outputs:
        if port.name.endswith("_addr"):
            base = port.name[:-5]
            memory_by_base.setdefault(
                base,
                {
                    "base": base,
                    "data_type": "i32",
                    "addr_type": None,
                    "write_type": None,
                    "has_write": False,
                },
            )["addr_type"] = port.type
        elif port.name.endswith("_wdata"):
            base = port.name[:-6]
            info = memory_by_base.setdefault(
                base,
                {
                    "base": base,
                    "data_type": port.type,
                    "addr_type": None,
                    "write_type": None,
                    "has_write": False,
                },
            )
            info["write_type"] = port.type
            info["data_type"] = port.type
            info["has_write"] = True
        elif port.name.endswith("_we"):
            base = port.name[:-3]
            info = memory_by_base.setdefault(
                base,
                {
                    "base": base,
                    "data_type": "i32",
                    "addr_type": None,
                    "write_type": None,
                    "has_write": True,
                },
            )
            info["has_write"] = True

    memory_bases = set(memory_by_base)
    scalar_inputs = [
        port
        for port in design.inputs
        if port.name not in {"clk", "rst", "req_valid", "resp_ready"}
        and not any(port.name == f"{base}_rdata" for base in memory_bases)
    ]
    scalar_outputs = [
        port
        for port in design.outputs
        if port.name not in {"req_ready", "resp_valid"}
        and not any(port.name in {f"{base}_addr", f"{base}_wdata", f"{base}_we"} for base in memory_bases)
    ]
    memory_interfaces = [
        WrapperMemoryInterface(
            base=str(memory_by_base[key]["base"]),
            data_type=str(memory_by_base[key]["data_type"]),
            addr_type=None if memory_by_base[key]["addr_type"] is None else str(memory_by_base[key]["addr_type"]),
            write_type=None if memory_by_base[key]["write_type"] is None else str(memory_by_base[key]["write_type"]),
            has_write=bool(memory_by_base[key]["has_write"]),
        )
        for key in sorted(memory_by_base)
    ]
    return scalar_inputs, scalar_outputs, memory_interfaces


def wrapper_core_signal(port_name: str, plan: SlaveWrapperPlan) -> str:
    """Resolve one wrapper-side signal name for one raw core port."""
    if port_name == "clk" or port_name == "rst":
        return port_name
    if port_name == "req_valid":
        return "core_req_valid_n"
    if port_name == "resp_ready":
        return "core_resp_ready_n"
    if port_name == "req_ready":
        return "core_req_ready_n"
    if port_name == "resp_valid":
        return "core_resp_valid_n"
    for interface in plan.memory_interfaces:
        if port_name == f"{interface.base}_rdata":
            return f"core_{interface.base}_rdata_n"
        if port_name == f"{interface.base}_addr":
            return f"core_{interface.base}_addr_n"
        if port_name == f"{interface.base}_wdata":
            return f"core_{interface.base}_wdata_n"
        if port_name == f"{interface.base}_we":
            return f"core_{interface.base}_we_n"
    if any(port.name == port_name for port in plan.scalar_inputs):
        return f"core_{port_name}_n"
    if any(port.name == port_name for port in plan.scalar_outputs):
        return f"core_{port_name}_n"
    return f"core_{port_name}_n"
