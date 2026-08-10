#!/usr/bin/env python3
"""Dump reference critical-path results for the C++ analysis port to match.

Builds a handful of graphs with the Python ``wg_graphs`` library, runs
``critical_path`` over each, and prints the makespan and the critical chain.
The chain matters as much as the makespan: several chains usually tie, and which
one gets reported depends on the topological order and on Python's ``max``
returning the first maximal key. The C++ port reproduces both tie-breaks, and
this script is how that claim is checked.

Usage:
    PYTHONPATH=/path/to/wg_graphs python3 analysis_oracle.py
"""

from wg_graphs.analysis import critical_path, overlap_report
from wg_graphs.core import Allocation, Graph, Operation, Read, Write
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


CASES = [case_one_to_one, case_fan_in, case_chain3, case_diamond, case_streamed]

EDGE_COST = lambda e: e.weight / 8.0


def node_cost_for(graph):
    """Deliberately lopsided, so a weighted run cannot agree with a unit one.

    The reference keys nodes by operation *name*; the C++ port interns the name
    to its index in the operator list. Looking the index up here keeps the two
    cost functions numerically identical.
    """
    order = {op.name: i for i, op in enumerate(graph.operations)}
    return lambda n: 1.0 + order[n.op] * 2.0 + n.wg * 0.5


def main() -> None:
    for case in CASES:
        name, g = case()
        for tag, nc, ec in (("unit", None, None), ("weighted", node_cost_for(g), EDGE_COST)):
            cp = critical_path(g, nc, ec)
            chain = " -> ".join(repr(n) for n in cp.path)
            print(f"{name}/{tag} edges={len(g.edges)} makespan={cp.makespan:.4f}")
            print(f"{name}/{tag} chain={chain}")
        print(f"{name}/report {overlap_report(g, 8.0)!r}")
        print()


if __name__ == "__main__":
    main()
