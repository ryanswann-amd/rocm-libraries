#!/usr/bin/env python3
"""Dump reference cost-model-priced schedules for the C++ port to match.

Runs the roofline, event-driven and XCD runtimes over a GEMM-plus-collective
graph and prints, for every atom, its start, duration and lane. Those three
numbers are the whole output of a priced schedule, so comparing them node by
node leaves nowhere for a discrepancy to hide.

The cost model is the reference ``Roofline`` over MI300X parameters, which the
C++ dumper reimplements arithmetic-for-arithmetic. Both sides are IEEE doubles
doing the same operations, so the printed values are expected to agree well
beyond the six decimals shown. The reference computes seconds throughout while
the port computes cycles and converts once at the end, which is a difference in
where a clock is applied rather than in what is computed.

``XcdRuntime`` lives only on the wg-graphs-figures-update branch, so point
PYTHONPATH at a checkout of it:
    PYTHONPATH=/path/to/wg_graphs_figures python3 priced_oracle.py
"""

from wg_graphs.core import Allocation, Graph, Operation, Read, Write
from wg_graphs.cost import MI300X, OpWork, Roofline, comm_step, gemm_tile
from wg_graphs.runtime import EventDrivenRuntime, RooflineRuntime, XcdRuntime
from wg_graphs.symbolic import contiguous, streamed


def gemm_then_reduce():
    """8 GEMM tiles feeding 4 collective steps: the motivating overlap case."""
    c = Allocation("c", (64,))
    out = Allocation("out", (64,))
    gemm = Operation("gemm", 8, (Write(c, contiguous(8)),))
    ar = Operation("all_reduce", 4, (Read(c, contiguous(16)), Write(out, contiguous(16))))
    specs = {
        "gemm": gemm_tile(bm=128, bn=128, k=4096),
        "all_reduce": comm_step(elems=1 << 20, fan=2),
    }
    return "gemm_reduce", Graph([gemm, ar]), specs


def streamed_pair():
    """One workgroup each, four iterations: exercises serialize_wg_iters."""
    buf = Allocation("buf", (32,))
    p = Operation("produce", 1, (Write(buf, streamed(32, 4)),), num_iters=4)
    q = Operation("consume", 1, (Read(buf, streamed(32, 4)),), num_iters=4)
    specs = {
        "produce": gemm_tile(bm=64, bn=64, k=512),
        "consume": comm_step(elems=1 << 16),
    }
    return "streamed_pair", Graph([p, q]), specs


def with_barrier():
    """A zero-work 'sync' stage, which the XCD runtime skips over."""
    x = Allocation("x", (16,))
    y = Allocation("y", (16,))
    a = Operation("a", 4, (Write(x, contiguous(4)),))
    s = Operation("sync_all", 1, (Read(x, contiguous(16)), Write(y, contiguous(16))))
    b = Operation("b", 4, (Read(y, contiguous(4)),))
    specs = {
        "a": gemm_tile(bm=64, bn=64, k=1024),
        "sync_all": OpWork(),
        "b": comm_step(elems=1 << 14),
    }
    return "with_barrier", Graph([a, s, b]), specs


CASES = [gemm_then_reduce, streamed_pair, with_barrier]


def render(graph, sched) -> str:
    """One field per atom: label@start+duration/lane, in dispatch order."""
    parts = []
    for n in graph.nodes():
        parts.append(f"{n!r}@{sched.start[n]:.6f}+{sched.duration[n]:.6f}/L{sched.lane[n]}")
    return " ".join(parts)


def main() -> None:
    for case in CASES:
        name, g, specs = case()
        cost = Roofline(MI300X, specs)

        configs = []
        for lanes in (None, 2, 8):
            for ser in (False, True):
                configs.append((f"roofline/lanes={lanes}/ser={ser}",
                                RooflineRuntime(cost, lanes=lanes, serialize_wg_iters=ser)))
                configs.append((f"event/lanes={lanes}/ser={ser}",
                                EventDrivenRuntime(cost, lanes=lanes, serialize_wg_iters=ser)))
        # A lane pool that splits the machine between the two operations.
        configs.append(("roofline/pool", RooflineRuntime(
            cost, lanes=8, lane_pool={"gemm": [0, 1, 2, 3, 4, 5], "all_reduce": [6, 7],
                                      "produce": [0, 1], "consume": [2, 3],
                                      "a": [0, 1, 2], "b": [3]})))
        configs.append(("xcd/2x2", XcdRuntime(cost, num_xcds=2, cus_per_xcd=2)))
        configs.append(("xcd/8x38", XcdRuntime(cost, num_xcds=8, cus_per_xcd=38)))

        for tag, rt in configs:
            s = rt.schedule(g)
            head = f"{name}/{tag}"
            print(f"{head} makespan={s.makespan:.6f} lanes={s.num_lanes} "
                  f"util={s.utilization():.6f}")
            print(f"{head} atoms={render(g, s)}")
        print()


if __name__ == "__main__":
    main()
