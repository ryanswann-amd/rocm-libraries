"""
Communication primitives — composable operations that map to functional unit work.

Each primitive's resolve() traces the full data path through the cache hierarchy
and returns FunctionalUnitWork for one loop iteration (cl_per_iter cache lines).
"""

from dataclasses import dataclass
from .types import FunctionalUnitWork, CommConfig, CACHELINE_BYTES


@dataclass
class Load:
    """Read from local HBM into registers."""

    def resolve(self, cl_per_iter: int, instrs_per_cl: int, elements_per_iter: int) -> FunctionalUnitWork:
        return FunctionalUnitWork(
            vmem_read_instrs=cl_per_iter * instrs_per_cl,
            tcp_read_cl=cl_per_iter,
            l2_read_cl=cl_per_iter,
            mall_read_cl=cl_per_iter,
            hbm_read_cl=cl_per_iter,
        )


@dataclass
class Store:
    """Write from registers to local HBM."""
    write_through: bool = False

    def resolve(self, cl_per_iter: int, instrs_per_cl: int, elements_per_iter: int) -> FunctionalUnitWork:
        return FunctionalUnitWork(
            vmem_write_instrs=cl_per_iter * instrs_per_cl,
            tcp_write_cl=cl_per_iter,
            l2_write_cl=0 if self.write_through else cl_per_iter,
            mall_write_cl=cl_per_iter,
            hbm_write_cl=cl_per_iter,
        )


@dataclass
class Pull:
    """Read from a remote GPU's HBM via xGMI (ingress)."""
    peer: int = 0

    def resolve(self, cl_per_iter: int, instrs_per_cl: int, elements_per_iter: int) -> FunctionalUnitWork:
        return FunctionalUnitWork(
            vmem_read_instrs=cl_per_iter * instrs_per_cl,
            tcp_read_cl=cl_per_iter,
            l2_read_cl=cl_per_iter,
            xgmi_read_cl=cl_per_iter,
        )


@dataclass
class Push:
    """Write to a remote GPU's HBM via xGMI (egress)."""
    peer: int = 0

    def resolve(self, cl_per_iter: int, instrs_per_cl: int, elements_per_iter: int) -> FunctionalUnitWork:
        return FunctionalUnitWork(
            vmem_read_instrs=cl_per_iter * instrs_per_cl,
            tcp_read_cl=cl_per_iter,
            l2_read_cl=cl_per_iter,
            mall_read_cl=cl_per_iter,
            hbm_read_cl=cl_per_iter,
            xgmi_write_cl=cl_per_iter,
        )


@dataclass
class Reduce:
    """Element-wise reduction on data in registers."""
    op: str = "sum"

    def resolve(self, cl_per_iter: int, instrs_per_cl: int, elements_per_iter: int) -> FunctionalUnitWork:
        return FunctionalUnitWork(valu_ops=elements_per_iter)


@dataclass
class Signal:
    """Notify a peer that data is ready."""
    peer: int = 0

    def resolve(self, cl_per_iter: int, instrs_per_cl: int, elements_per_iter: int) -> FunctionalUnitWork:
        return FunctionalUnitWork(atomic_count=1, xgmi_write_cl=1)


@dataclass
class Wait:
    """Spin-wait for a peer's signal."""
    peer: int = 0

    def resolve(self, cl_per_iter: int, instrs_per_cl: int, elements_per_iter: int) -> FunctionalUnitWork:
        return FunctionalUnitWork(atomic_count=1, l2_read_cl=1)


Op = Load | Store | Pull | Push | Reduce | Signal | Wait
