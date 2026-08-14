#!/usr/bin/env python3
"""Dump reference integer-timestep schedules for the C++ port to match.

Runs breadth-first, ASAP and depth-first over a range of graphs and lane counts
and prints the makespan plus the full timestep-by-timestep dispatch. The
timesteps matter as much as the makespan: two policies often agree on when the
last node finishes while disagreeing completely about the order in between, and
the order is the whole point of having three policies.

``name_order`` and ``streamed`` exist to probe one specific divergence. The
reference sorts depth-first successors by operation *name*; a C++ node carries
the operation *index*. Those agree only when names happen to sort in declaration
order, and these two cases are built so they do not.

Usage:
    PYTHONPATH=/path/to/wg_graphs python3 runtime_oracle.py
"""

from wg_graphs.core import Allocation, Graph, Operation, Read, Write
from wg_graphs.runtime import AsapRuntime, BreadthFirstRuntime, DepthFirstRuntime
from wg_graphs.symbolic import contiguous, streamed


def case_one_to_one():
    x = Allocation("x", (16,))
    a = Operation("a", 4, (Write(x, contiguous(4)),))
    b = Operation("b", 4, (Read(x, contiguous(4)),))
    return "one_to_one", Graph([a, b])


def case_fan_in():
    x = Allocation("x", (16,))
    a = Operation("a", 4, (Write(x, contiguous(4)),))
    b = Operation("b", 2, (Read(x, contiguous(8)),))
    return "fan_in", Graph([a, b])


def case_chain3():
    x = Allocation("x", (16,))
    y = Allocation("y", (16,))
    a = Operation("a", 4, (Write(x, contiguous(4)),))
    b = Operation("b", 4, (Read(x, contiguous(4)), Write(y, contiguous(4))))
    c = Operation("c", 4, (Read(y, contiguous(4)),))
    return "chain3", Graph([a, b, c])


def case_diamond():
    x = Allocation("x", (16,))
    y = Allocation("y", (16,))
    z = Allocation("z", (16,))
    a = Operation("a", 4, (Write(x, contiguous(4)),))
    b = Operation("b", 4, (Read(x, contiguous(4)), Write(y, contiguous(4))))
    c = Operation("c", 4, (Read(x, contiguous(4)), Write(z, contiguous(4))))
    d = Operation("d", 4, (Read(y, contiguous(4)), Read(z, contiguous(4))))
    return "diamond", Graph([a, b, c, d])


def case_streamed():
    buf = Allocation("buf", (32,))
    p = Operation("produce", 1, (Write(buf, streamed(32, 4)),), num_iters=4)
    c = Operation("consume", 1, (Read(buf, streamed(32, 4)),), num_iters=4)
    return "streamed", Graph([p, c])


def case_name_order():
    """Declaration order and alphabetical order disagree on every operation."""
    x = Allocation("x", (16,))
    y = Allocation("y", (16,))
    z = Operation("z", 4, (Write(x, contiguous(4)),))
    m = Operation("m", 4, (Read(x, contiguous(4)), Write(y, contiguous(4))))
    a = Operation("a", 4, (Read(y, contiguous(4)),))
    return "name_order", Graph([z, m, a])


def case_split_named():
    """A node whose two consumers sit in operations that sort against their order.

    ``a`` feeds both ``z`` (declared second) and ``b`` (declared third). Sorting
    the successors of ``a#i`` by name puts ``b#i`` first; sorting by operation
    index puts ``z#i`` first. Every other case in this file has at most one
    consumer operation per node, so this is the only place the two rules can be
    told apart.
    """
    x = Allocation("x", (16,))
    a = Operation("a", 4, (Write(x, contiguous(4)),))
    z = Operation("z", 4, (Read(x, contiguous(4)),))
    b = Operation("b", 4, (Read(x, contiguous(4)),))
    return "split_named", Graph([a, z, b])


def case_two_sources():
    """Two independent producers, declared against alphabetical order."""
    x = Allocation("x", (16,))
    y = Allocation("y", (16,))
    z = Operation("z", 4, (Write(x, contiguous(4)),))
    b = Operation("b", 4, (Write(y, contiguous(4)),))
    c = Operation("c", 4, (Read(x, contiguous(4)), Read(y, contiguous(4))))
    return "two_sources", Graph([z, b, c])


CASES = [case_one_to_one, case_fan_in, case_chain3, case_diamond, case_streamed,
         case_name_order, case_split_named, case_two_sources]
LANES = [None, 1, 2, 3]


def render(sched) -> str:
    """One line per timestep: "t: node node node"."""
    parts = []
    for t, group in sched.timesteps().items():
        parts.append(f"{t}:" + "".join(f" {n!r}" for n in group))
    return " | ".join(parts)


def main() -> None:
    for case in CASES:
        name, g = case()
        for lanes in LANES:
            for rt in (BreadthFirstRuntime(lanes), AsapRuntime(lanes), DepthFirstRuntime(lanes)):
                s = rt.schedule(g)
                tag = f"{name}/lanes={lanes}/{rt.name}"
                print(f"{tag} makespan={s.makespan}")
                print(f"{tag} steps={render(s)}")
        print()


if __name__ == "__main__":
    main()
