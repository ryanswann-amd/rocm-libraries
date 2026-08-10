# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the ``origami.graphs`` bindings.

The C++ suite already covers the model itself, so these check the *binding*: that
each layer is reachable from Python, that the types round-trip, and that the
whole pipeline composes — build a graph, schedule it, price it, rank the
alternatives, emit a trace.
"""

import json
import os
import sys

import pytest


def _import_origami():
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if src not in sys.path:
        sys.path.insert(0, src)
    import origami  # noqa: E402

    if not hasattr(origami, "graphs"):
        pytest.skip("origami extension built without the graphs submodule")
    return origami


origami = _import_origami()
G = origami.graphs


@pytest.fixture
def fan_in():
    """a(4 workgroups) writes x; b(2 workgroups) reads it. Two edges per consumer."""
    x = G.allocation_t("x", [16])
    a = G.operation_t("a", 4)
    a.access_patterns = [G.write(x, G.contiguous(4))]
    b = G.operation_t("b", 2)
    b.access_patterns = [G.read(x, G.contiguous(8))]
    return G.graph_t([a, b], name="fan_in")


def flat_cost(graph, seconds=1.0):
    table = G.cost_table_t(graph)
    for op in range(graph.num_operations()):
        table.set(graph.op_name(op), G.op_cost_t.from_custom(lambda node, active: seconds))
    return table


# --- derivation ---------------------------------------------------------


def test_edges_are_derived_from_access_patterns(fan_in):
    assert fan_in.name() == "fan_in"
    assert fan_in.num_operations() == 2
    assert fan_in.num_nodes() == 6

    # Each consumer reads the blocks two producers wrote, so four edges.
    assert len(fan_in.edges()) == 4
    for e in fan_in.edges():
        assert e.allocation == "x"
        assert e.src.op == 0 and e.dst.op == 1


def test_nodes_and_labels_round_trip(fan_in):
    nodes = fan_in.nodes()
    assert len(nodes) == 6
    assert nodes[0] == G.wg_node_t(0, 0, 0)
    assert fan_in.label(nodes[0]) == "a#0"
    # Hashable, so a Python dict keyed by node works.
    assert len({n: fan_in.label(n) for n in nodes}) == 6


def test_a_free_graph_takes_explicit_edges():
    g = G.free_graph_t("hand_built")
    g.add_op("produce", 2)
    g.add_op("consume", 2)
    g.add_edge(G.wg_node_t(0, 0), G.wg_node_t(1, 1), "buf", 4)

    assert g.num_nodes() == 4
    assert len(g.edges()) == 1
    assert g.edges()[0].weight == 4


def test_a_snapshot_matches_its_origin(fan_in):
    snap = fan_in.to_free()
    assert snap.num_nodes() == fan_in.num_nodes()
    assert len(snap.edges()) == len(fan_in.edges())


# --- analysis -----------------------------------------------------------


def test_topological_order_and_critical_path(fan_in):
    order = G.topological_order(fan_in)
    assert len(order) == 6
    assert G.is_acyclic(fan_in)

    # Unit cost, unbounded resources: one producer then one consumer.
    assert G.critical_path(fan_in).makespan == 2.0
    assert len(G.critical_path(fan_in).path) == 2


def test_critical_path_takes_a_python_cost_function(fan_in):
    path = G.critical_path(fan_in, lambda node: 1.0 + node.op)
    assert path.makespan == pytest.approx(3.0)  # a costs 1, b costs 2


# --- integer runtimes ---------------------------------------------------


def test_the_three_integer_runtimes_are_reachable(fan_in):
    for runtime in (G.breadth_first_runtime_t, G.asap_runtime_t, G.depth_first_runtime_t):
        s = runtime(lanes=2).schedule(fan_in)
        assert len(s.order) == 6
        assert s.makespan() > 0
        assert "makespan" in s.summary()


def test_more_lanes_never_lengthen_an_integer_schedule(fan_in):
    previous = 10**9
    for lanes in (1, 2, 4, 8):
        now = G.asap_runtime_t(lanes=lanes).schedule(fan_in).makespan()
        assert now <= previous
        previous = now


def test_no_overlap_is_never_faster_than_producer_greedy(fan_in):
    assert (
        G.breadth_first_runtime_t(lanes=2).schedule(fan_in).makespan()
        >= G.asap_runtime_t(lanes=2).schedule(fan_in).makespan()
    )


# --- cost ---------------------------------------------------------------


def test_the_roofline_arm_picks_the_binding_roof():
    tile = G.roofline_spec_t.gemm_tile(128, 128, 4096)
    step = G.roofline_spec_t.comm_step(1 << 20, fan=2.0)

    assert G.roofline_seconds(tile) > 0
    assert G.roofline_bound(tile) == "hbm"  # deep K with no reuse modelled
    assert G.roofline_bound(step) == "link"
    assert G.roofline_bound(G.roofline_spec_t()) == "none"


def test_a_cost_table_dispatches_per_operation(fan_in):
    table = G.cost_table_t(fan_in)
    table.set("a", G.op_cost_t.from_roofline(G.roofline_spec_t.gemm_tile(128, 128, 4096)))
    table.set("b", G.op_cost_t.from_custom(lambda node, active: 1e-6 * active))

    assert table.kind_of(0) == G.cost_kind_t.roofline
    assert table.kind_of(1) == G.cost_kind_t.custom
    assert table.node_cost(G.wg_node_t(0, 0)) > 0

    # The custom arm receives the contention level, so a Python callable can
    # model it the same way a C++ one does.
    assert table.node_cost_at(G.wg_node_t(1, 0), 4) == pytest.approx(4e-6)


def test_an_unpriced_operation_falls_back_to_the_default(fan_in):
    assert G.cost_table_t(fan_in).node_cost(G.wg_node_t(0, 0)) == 0.0

    settings = G.cost_settings_t()
    settings.default_seconds = 2e-6
    assert G.cost_table_t(fan_in, settings).node_cost(G.wg_node_t(0, 0)) == pytest.approx(2e-6)


def test_a_misspelled_operation_name_raises(fan_in):
    with pytest.raises(IndexError):
        G.cost_table_t(fan_in).set("nope", G.op_cost_t.from_roofline(G.roofline_spec_t()))


def test_the_comm_arm_reaches_the_calibrated_model(fan_in):
    # The comm arm is bound beside the HIP-linked GEMM surface, so a graphs-only
    # build (python/tests/build_graphs_only.sh) does not have it.
    comm = getattr(origami, "comm", None)
    if comm is None:
        pytest.skip("origami built without the comm submodule")
    topology = comm.gpu_topology_t(
        arch=origami.architecture_t.gfx942,
        num_cu=304,
        num_xcd=8,
        cu_per_xcd=38,
        l2_capacity_bytes=4 * 1024 * 1024,
    )
    system = comm.make_system(
        comm.get_arch_ceilings(origami.architecture_t.gfx942), topology, 2.0
    )

    spec = G.comm_spec_t(
        work_graph=["load", "store", "push"],
        wg_tile_bytes=1 << 16,
        num_wgs=8,
        bw_per_wg=8.0,
        primitive=comm.primitive_t.all_gather,
    )
    assert G.comm_seconds(spec, system) > 0
    # Contention only bites once the machine is genuinely crowded.
    assert G.comm_seconds(spec, system, 304) > G.comm_seconds(spec, system, 1)

    settings = G.cost_settings_t()
    settings.comm_system = system
    table = G.cost_table_t(fan_in, settings)
    table.set("b", G.op_cost_t.from_comm(spec))
    assert table.kind_of(1) == G.cost_kind_t.comm
    assert table.node_cost(G.wg_node_t(1, 0)) > 0


def test_an_unknown_comm_primitive_raises():
    with pytest.raises(ValueError):
        G.comm_spec_t(work_graph=["teleport"], wg_tile_bytes=1024)


# --- continuous runtimes ------------------------------------------------


def test_the_three_continuous_runtimes_are_reachable(fan_in):
    table = flat_cost(fan_in, 2.0)
    options = G.cost_runtime_options_t()
    options.lanes = 3
    options.scale = 1.0

    for s in (
        G.roofline_runtime_t(table, options).schedule(fan_in),
        G.event_driven_runtime_t(table, options).schedule(fan_in),
        G.xcd_runtime_t(table, G.xcd_options_t()).schedule(fan_in),
    ):
        assert len(s.order) == 6
        assert s.makespan() > 0
        assert 0.0 < s.utilization() <= 1.0
        for n in s.order:
            assert s.finish(n) == pytest.approx(s.start(n) + s.duration(n))


def test_a_lane_pool_partitions_the_machine(fan_in):
    table = flat_cost(fan_in, 2.0)
    options = G.cost_runtime_options_t()
    options.lanes = 4
    options.scale = 1.0
    options.lane_pool = {"a": [0, 1], "b": [2, 3]}

    s = G.roofline_runtime_t(table, options).schedule(fan_in)
    for n in s.order:
        assert s.lane(n) <= 1 if n.op == 0 else s.lane(n) >= 2


def test_serialising_iterations_exposes_a_pipeline():
    buf = G.allocation_t("buf", [32])
    p = G.operation_t("produce", 1)
    p.num_iters = 4
    p.access_patterns = [G.write(buf, G.streamed(32, 4))]
    c = G.operation_t("consume", 1)
    c.num_iters = 4
    c.access_patterns = [G.read(buf, G.streamed(32, 4))]
    g = G.graph_t([p, c], name="streamed")

    table = flat_cost(g, 1.0)
    loose = G.cost_runtime_options_t()
    loose.scale = 1.0
    tight = G.cost_runtime_options_t()
    tight.scale = 1.0
    tight.serialize_wg_iters = True

    # Nothing in the graph says a workgroup runs its own iterations in sequence.
    assert G.roofline_runtime_t(table, loose).schedule(g).makespan() == pytest.approx(2.0)
    assert G.roofline_runtime_t(table, tight).schedule(g).makespan() == pytest.approx(5.0)


# --- extending the model from Python ------------------------------------


class PyCost(G.cost_model_t):
    """A cost model written in Python: seconds per operation, free hops."""

    def __init__(self, per_op):
        super().__init__()
        self.per_op = per_op

    def node_cost(self, node):
        return self.per_op[node.op]

    def edge_cost(self, edge):
        return 0.0


class SerialRuntime(G.cost_runtime_t):
    """A scheduling policy written in Python: one lane, topological order.

    Deliberately not one of the shipped policies, so the numbers it produces
    could not have come from anywhere else.
    """

    def name(self):
        return "serial"

    def schedule(self, graph):
        s = G.cost_schedule_t()
        s.runtime = self.name()
        s.lanes = 1
        s.units = "us"
        cost = graph.cost()
        t = 0.0
        for node in G.topological_order(graph):
            d = cost.node_cost(node) if cost is not None else 1.0
            s.place(node, t, d, 0)
            t += d
        return s


def test_a_graph_carries_its_own_cost_model(fan_in):
    assert fan_in.cost() is None

    fan_in.set_cost(PyCost([2.0, 5.0]))
    assert isinstance(fan_in.cost(), PyCost)

    options = G.cost_runtime_options_t()
    options.scale = 1.0
    # Unlimited lanes: four producers at 2.0, then two consumers at 5.0.
    assert G.roofline_runtime_t(options).schedule(fan_in).makespan() == pytest.approx(7.0)


def test_a_runtimes_own_cost_model_overrides_the_graphs(fan_in):
    fan_in.set_cost(PyCost([20.0, 50.0]))
    options = G.cost_runtime_options_t()
    options.scale = 1.0

    cheap = G.roofline_runtime_t(PyCost([2.0, 5.0]), options).schedule(fan_in)
    assert cheap.makespan() == pytest.approx(7.0)


def test_scheduling_a_graph_with_no_cost_raises(fan_in):
    with pytest.raises(Exception):
        G.roofline_runtime_t().schedule(fan_in)


def test_a_python_runtime_schedules_and_ranks(fan_in):
    fan_in.set_cost(PyCost([2.0, 5.0]))
    runtime = SerialRuntime()

    s = runtime.schedule(fan_in)
    assert s.runtime == "serial"
    assert s.num_lanes() == 1
    # One lane, nothing overlapping: 4 * 2.0 + 2 * 5.0.
    assert s.makespan() == pytest.approx(18.0)
    assert len(s.order) == fan_in.num_nodes()
    for n in s.order:
        assert s.finish(n) == pytest.approx(s.start(n) + s.duration(n))

    c = G.graph_config_t()
    c.runtime_override = runtime
    c.name = "serial"
    assert G.predict_latency(fan_in, c) == pytest.approx(18.0)

    ranked = G.rank_configs(fan_in, [c])
    assert ranked[0].config.name == "serial"
    # The override survives being copied into the result, which a raw pointer
    # to a Python-owned object would not.
    assert ranked[0].config.runtime_override.name() == "serial"


def test_a_python_runtime_can_be_traced(fan_in):
    fan_in.set_cost(PyCost([2.0, 5.0]))
    s = SerialRuntime().schedule(fan_in)
    doc = json.loads(G.chrome_trace(fan_in, s))
    assert len(doc["traceEvents"]) > 0


def test_callable_set_indexes_a_table(fan_in):
    # The escape hatch: indices a kernel model already computed, rather than a
    # closed form re-derived here.
    touched = {0: [0, 1], 1: [2, 3]}
    x = G.allocation_t("x", [8])
    p = G.operation_t("produce", 2)
    p.access_patterns = [G.write(x, G.callable_set(lambda wg, it, env: touched[wg]))]
    c = G.operation_t("consume", 2)
    c.access_patterns = [G.read(x, G.contiguous(4))]
    g = G.graph_t([p, c], name="table")

    # Both producers land inside consumer 0's block [0, 4), and neither in 1's.
    assert {(g.label(e.src), g.label(e.dst)) for e in g.edges()} == {
        ("produce#0", "consume#0"),
        ("produce#1", "consume#0"),
    }


def test_symbolic_dimensions_are_bound_at_construction():
    x = G.allocation_t("x", [16])
    a = G.operation_t("a", 4)
    a.access_patterns = [G.write(x, G.contiguous(G.sym("block")))]
    b = G.operation_t("b", 2)
    b.access_patterns = [G.read(x, G.contiguous(8))]
    g = G.graph_t([a, b], {"block": 4}, "symbolic")

    assert g.dims()["block"] == 4
    # Resolved to 4, the graph is the fan-in fixture, edges and all.
    assert len(g.edges()) == 4


# --- prebuilt graphs ----------------------------------------------------


def ag_matmul_tables():
    """Four fetchers staging eight flags, four gemm tiles waiting on two each."""
    fetch = [[[2 * w], [2 * w + 1]] for w in range(4)]
    gemm = [[[2 * w], [2 * w + 1]] for w in range(4)]
    return fetch, gemm, 8


def test_all_gather_matmul_derives_the_flag_handoff():
    prebuilt = pytest.importorskip("origami.prebuilt")
    fetch, gemm, num_flags = ag_matmul_tables()
    g = prebuilt.all_gather_matmul(fetch, gemm, num_flags, name="agmm")

    assert g.name() == "agmm"
    assert [g.op_name(i) for i in range(g.num_operations())] == ["fetch", "gemm"]
    # Fetcher w raises flags 2w and 2w+1 on its two iterations, and only tile w
    # waits on them, on the matching iteration.
    assert {(e.src.wg, e.src.it, e.dst.wg, e.dst.it) for e in g.edges()} == {
        (w, i, w, i) for w in range(4) for i in range(2)
    }


def test_inverting_the_flag_array_finds_the_same_edges_as_deriving_them():
    # The fast path exists because the derivation is quadratic in atoms while the
    # inversion is linear in edges, which matters on a tall shape. It is only
    # worth having if it is the same answer, so this is what says so.
    prebuilt = pytest.importorskip("origami.prebuilt")
    fetch, gemm, num_flags = ag_matmul_tables()

    derived = prebuilt.all_gather_matmul(fetch, gemm, num_flags, derive=True)
    inverted = prebuilt.all_gather_matmul(fetch, gemm, num_flags, derive=False)

    def edges(g):
        return sorted(
            (e.src.op, e.src.wg, e.src.it, e.dst.op, e.dst.wg, e.dst.it, e.allocation, e.weight)
            for e in g.edges()
        )

    assert edges(derived) == edges(inverted)
    assert derived.num_nodes() == inverted.num_nodes()
    for op in range(derived.num_operations()):
        assert derived.op_name(op) == inverted.op_name(op)
        assert derived.wg_count(op) == inverted.wg_count(op)
        assert derived.iter_count(op) == inverted.iter_count(op)


def test_a_flag_with_two_producers_is_a_race_not_a_schedule():
    prebuilt = pytest.importorskip("origami.prebuilt")
    with pytest.raises(ValueError, match="one producer"):
        prebuilt.all_gather_matmul([[[0]], [[0]]], [[[0]]], num_flags=1, derive=False)


def test_a_flag_outside_the_array_is_rejected():
    prebuilt = pytest.importorskip("origami.prebuilt")
    with pytest.raises(ValueError, match="outside"):
        prebuilt.all_gather_matmul([[[7]]], [[[7]]], num_flags=4)


# --- ranking ------------------------------------------------------------


def config(runtime, lanes, name=""):
    c = G.graph_config_t()
    c.runtime = runtime
    c.lanes = lanes
    c.scale = 1.0
    c.name = name
    return c


def test_ranking_returns_best_first(fan_in):
    table = flat_cost(fan_in)
    configs = [
        config(G.runtime_kind_t.roofline, 1, "one"),
        config(G.runtime_kind_t.roofline, 4, "four"),
        config(G.runtime_kind_t.roofline, 2, "two"),
    ]
    ranked = G.rank_configs(fan_in, configs, table)

    assert [r.config.name for r in ranked][0] == "four"
    assert all(ranked[i - 1].latency <= ranked[i].latency for i in range(1, len(ranked)))
    assert G.select_config(fan_in, configs, table).config.name == "four"
    assert len(G.select_topk_configs(fan_in, configs, 2, table)) == 2


def test_integer_runtimes_rank_without_a_cost_model(fan_in):
    configs = [config(G.runtime_kind_t.asap, 1), config(G.runtime_kind_t.asap, 4)]
    ranked = G.rank_configs(fan_in, configs)
    assert ranked[0].latency < ranked[1].latency


def test_infeasible_candidates_are_reported_not_raised(fan_in):
    bad = config(G.runtime_kind_t.asap, 0, "broken")
    assert G.rejection_reason(fan_in, bad) != ""
    assert G.predict_latency(fan_in, bad) == G.REJECTED_LATENCY

    ranked = G.rank_configs(fan_in, [bad])
    assert len(ranked) == 1 and ranked[0].rejected and ranked[0].rejection


def test_an_empty_candidate_list_raises(fan_in):
    with pytest.raises(RuntimeError):
        G.rank_configs(fan_in, [])


def test_ranking_across_graphs_prices_the_cross_product(fan_in):
    loose = G.free_graph_t("loose")
    loose.add_op("a", 4)
    loose.add_op("b", 2)

    table = flat_cost(fan_in)
    configs = [
        config(G.runtime_kind_t.roofline, 2, "two"),
        config(G.runtime_kind_t.roofline, 8, "eight"),
    ]
    ranked = G.rank_graphs([fan_in, loose], configs, table)

    assert len(ranked) == 4
    # The winner is a pair: the unconstrained graph at the widest machine.
    assert ranked[0].graph_name == "loose"
    assert ranked[0].config.name == "eight"
    assert ranked[0].graph_index == 1


def test_runtime_kind_names_round_trip():
    for name in ("breadth_first", "asap", "depth_first", "roofline", "event_driven", "xcd"):
        kind = getattr(G.runtime_kind_t, name)
        assert G.is_continuous(kind) == (name in ("roofline", "event_driven", "xcd"))


# --- trace --------------------------------------------------------------


def test_a_trace_is_valid_json_a_viewer_can_load(fan_in):
    table = flat_cost(fan_in, 2.0)
    options = G.cost_runtime_options_t()
    options.lanes = 3
    options.scale = 1.0
    s = G.roofline_runtime_t(table, options).schedule(fan_in)

    doc = json.loads(G.chrome_trace(fan_in, s))
    events = doc["traceEvents"]

    slices = [e for e in events if e.get("ph") == "X"]
    assert len(slices) == 6
    assert {e["name"] for e in slices} == {"a", "b"}
    assert all(e["dur"] == 2 for e in slices)

    # Dependency arrows, one pair per edge.
    assert len([e for e in events if e.get("ph") == "s"]) == len(fan_in.edges())
    assert len([e for e in events if e.get("ph") == "f"]) == len(fan_in.edges())


def test_an_integer_schedule_also_traces(fan_in):
    s = G.asap_runtime_t(lanes=2).schedule(fan_in)
    doc = json.loads(G.chrome_trace(fan_in, s))
    assert len([e for e in doc["traceEvents"] if e.get("ph") == "X"]) == 6


def test_a_trace_can_be_written_to_disk(fan_in, tmp_path):
    table = flat_cost(fan_in, 2.0)
    options = G.cost_runtime_options_t()
    options.scale = 1.0
    s = G.roofline_runtime_t(table, options).schedule(fan_in)

    path = tmp_path / "trace.json"
    G.write_chrome_trace(fan_in, s, str(path))
    assert json.loads(path.read_text())["traceEvents"]


def test_flow_events_can_be_turned_off(fan_in):
    table = flat_cost(fan_in, 2.0)
    options = G.cost_runtime_options_t()
    options.scale = 1.0
    s = G.roofline_runtime_t(table, options).schedule(fan_in)

    quiet = G.trace_options_t()
    quiet.flow_edges = False
    quiet.process_name = "no arrows"
    doc = json.loads(G.chrome_trace(fan_in, s, quiet))
    assert not [e for e in doc["traceEvents"] if e.get("ph") == "s"]


# --- the whole pipeline -------------------------------------------------


def test_the_pipeline_composes_end_to_end(fan_in, tmp_path):
    """Derive, price, rank, schedule the winner, emit it. The motivating flow."""
    table = G.cost_table_t(fan_in)
    table.set("a", G.op_cost_t.from_roofline(G.roofline_spec_t.gemm_tile(128, 128, 4096)))
    table.set("b", G.op_cost_t.from_roofline(G.roofline_spec_t.comm_step(1 << 20, fan=2.0)))

    configs = [config(G.runtime_kind_t.roofline, n, f"lanes={n}") for n in (1, 2, 4, 8)]
    ranked = G.rank_configs(fan_in, configs, table)
    best = ranked[0]
    assert not best.rejected

    # Four producers is the widest this graph ever gets, so the eighth lane buys
    # nothing and the two candidates tie. The stable sort then keeps the one
    # offered first, which is the cheaper machine — the useful answer.
    assert best.config.name == "lanes=4"
    assert ranked[1].config.name == "lanes=8"
    assert ranked[0].latency == ranked[1].latency

    options = G.cost_runtime_options_t()
    options.lanes = best.config.lanes
    options.scale = best.config.scale
    s = G.roofline_runtime_t(table, options).schedule(fan_in)
    assert s.makespan() == pytest.approx(best.latency)

    # The collective dominates, which is the asymmetry a uniform timestep hides.
    busy = s.busy_by_op()
    assert busy[1] > busy[0]

    path = tmp_path / "pipeline.json"
    G.write_chrome_trace(fan_in, s, str(path))
    assert len(json.loads(path.read_text())["traceEvents"]) > 6
