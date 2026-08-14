# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Graphs for kernels worth naming, built from a schedule the caller already has.

``origami.graphs`` gives the vocabulary — operations, access patterns, edges — and
this module gives the handful of shapes that keep coming back, so a caller
modelling a known kernel does not re-derive its dependency structure by hand.

What these builders take is the *materialised* schedule: which flags each
workgroup-iteration writes, which ones each waits on. They do not take the
kernel's decode parameters and re-implement the decode. That is deliberate. A
kernel already has a host-side mirror of its own pid decode — it needs one to
size the launch and to test itself — and a second copy here would be a second
thing to keep in step with the kernel, silently wrong the first time the kernel
changed. Reading the tables the caller already computed means the graph is right
whenever the caller's decode is, which is a property the caller can test on its
own terms.

So the division is: this module owns the *shape* of a kernel's dependency — that
an all-gather producer raises flags and a matmul consumer waits on them — and the
caller owns which flags those are.
"""

from typing import Optional, Sequence

from origami import graphs as _graphs

__all__ = ["all_gather_matmul"]

# The buffer the handoff flows through, and the operation names the graph
# reports. Named here rather than inline so a caller can index a schedule or a
# trace by the same strings the graph uses.
FLAGS = "flags"
FETCH = "fetch"
GEMM = "gemm"


def _pattern(table: Sequence[Sequence[Sequence[int]]]):
    """An ``index_offsets`` map over ``table[wg][it]``, empty outside the table.

    Workgroups walk different numbers of iterations — the last one in a strided
    loop runs fewer — but an operation has one iteration count, so the short
    walks are padded. An empty index set touches nothing and therefore produces
    no edges, which is exactly what a workgroup that has already finished does.
    """

    def indices(wg: int, it: int, ctx) -> list:
        if wg >= len(table):
            return []
        walk = table[wg]
        if it >= len(walk):
            return []
        return list(walk[it])

    return _graphs.index_offsets(indices)


def _grid(table: Sequence[Sequence[Sequence[int]]], what: str):
    if len(table) == 0:
        raise ValueError(f"all_gather_matmul: {what} table is empty; expected one entry per workgroup")
    return len(table), max(1, max(len(walk) for walk in table))


def _derived(fetch_writes, gemm_reads, num_flags, name):
    """Let origami find the edges, by intersecting the two access patterns."""
    fetch_wgs, fetch_iters = _grid(fetch_writes, FETCH)
    gemm_wgs, gemm_iters = _grid(gemm_reads, GEMM)

    flags = _graphs.allocation_t(FLAGS, [num_flags])

    fetch = _graphs.operation_t(FETCH, fetch_wgs)
    fetch.num_iters = fetch_iters
    fetch.access_patterns = [_graphs.write(flags, _pattern(fetch_writes))]

    gemm = _graphs.operation_t(GEMM, gemm_wgs)
    gemm.num_iters = gemm_iters
    gemm.access_patterns = [_graphs.read(flags, _pattern(gemm_reads))]

    return _graphs.graph_t([fetch, gemm], name=name)


def _inverted(fetch_writes, gemm_reads, num_flags, name):
    """State the edges outright, from an index of who raised each flag.

    Same edges as :func:`_derived`, reached without the quadratic scan. The
    derivation compares every fetch atom against every gemm atom, which is the
    right thing to do when a pattern is a formula and the answer is not known in
    advance. Here the answer is known: a flag has one producer, so one pass over
    the writes builds the index and one pass over the reads uses it, and the work
    is proportional to the edges rather than to the pairs. On a tall shape that
    is the difference between a minute and no time at all.

    Only correct because a flag is raised once. That is a property of the kernel
    — a second writer would be a race, not a schedule — and it is checked here
    rather than assumed.
    """
    producer = {}
    for wg, walk in enumerate(fetch_writes):
        for it, flags in enumerate(walk):
            for f in flags:
                if f in producer:
                    raise ValueError(
                        f"all_gather_matmul: flag {f} is raised by fetch workgroup "
                        f"{producer[f][0]} and again by {wg}; a flag has one producer"
                    )
                producer[f] = (wg, it)

    graph = _graphs.free_graph_t(name)
    graph.add_op(FETCH, *_grid(fetch_writes, FETCH))
    graph.add_op(GEMM, *_grid(gemm_reads, GEMM))

    for wg, walk in enumerate(gemm_reads):
        for it, flags in enumerate(walk):
            for f in flags:
                src = producer.get(f)
                if src is None:
                    continue  # nothing raises it, so nothing feeds this wait
                graph.add_edge(
                    _graphs.wg_node_t(0, src[0], src[1]),
                    _graphs.wg_node_t(1, wg, it),
                    FLAGS,
                    1,
                )
    return graph


def all_gather_matmul(
    fetch_writes: Sequence[Sequence[Sequence[int]]],
    gemm_reads: Sequence[Sequence[Sequence[int]]],
    num_flags: int,
    name: str = "all_gather_matmul",
    cost: Optional[object] = None,
    derive: bool = True,
):
    """The dependency graph of a fused all-gather and matmul.

    Producers stage tiles of A and raise a flag per tile; consumers spin on the
    flags covering the A they need and then multiply. Every dependency in the
    kernel flows through that flag array and nothing else, so a fetch
    workgroup-iteration feeds a gemm workgroup-iteration exactly when the two
    name a common flag, and the number they share is the edge's weight.

    Both tables are indexed ``[workgroup][iteration]`` and hold flag indices.
    Ragged walks are fine; short ones are padded with iterations that touch
    nothing.

    :param fetch_writes: Flags each fetch workgroup-iteration raises.
    :param gemm_reads: Flags each gemm workgroup-iteration waits on.
    :param num_flags: Size of the flag array; every index must be below it.
    :param name: Graph label, surfaced in ranking results.
    :param cost: Cost model to attach, if the caller has one.
    :param derive: Intersect the access patterns and let origami find the edges.
        Set false to invert the flag array instead, which is the same answer in
        time proportional to the edges rather than to the pairs of atoms. The
        two are checked against each other in origami's own tests, so the fast
        path is worth taking once a caller's tables are trusted.
    :returns: A graph over operations ``fetch`` and ``gemm``. Derived, it is a
        ``graphs.graph_t``; inverted, a ``graphs.free_graph_t``. Both satisfy
        ``wg_graph_t``, which is all a runtime needs.
    :raises ValueError: If a table is empty, an index is out of range, or a flag
        has more than one producer.
    """
    for what, table in ((FETCH, fetch_writes), (GEMM, gemm_reads)):
        for wg, walk in enumerate(table):
            for it, flags in enumerate(walk):
                for f in flags:
                    if not 0 <= f < num_flags:
                        raise ValueError(
                            f"all_gather_matmul: {what} workgroup {wg} iteration {it} "
                            f"names flag {f}, outside [0, {num_flags})"
                        )

    build = _derived if derive else _inverted
    graph = build(fetch_writes, gemm_reads, num_flags, name)
    if cost is not None:
        graph.set_cost(cost)
    return graph
