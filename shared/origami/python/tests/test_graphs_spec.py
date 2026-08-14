"""The proposed symbolic graph API, exercised as the design document writes it.

Two examples are transcribed here rather than paraphrased. The first is the
authoring layer: one specification, symbolic in both namespaces, with a
node-level cost expression. The second is the consumer: bind a problem, sweep
configurations, and rank the candidates against a machine.

They are acceptance tests for the shape of the API, not for any particular
number. What they pin down is that a specification can be written without
mentioning a device, that instantiating it many times is one call per
candidate, and that hardware enters exactly once, at ranking.
"""

import pytest

origami = pytest.importorskip("origami")
G = origami.graphs


def ceil_div(value, divisor):
    """Return value divided by divisor, rounded upward."""
    return (value + divisor - 1) // divisor


def gemm_node_cost(node):
    """Build the symbolic runtime cost, in cycles, for any node of the GEMM operation."""
    M, N, K = G.problem("M"), G.problem("N"), G.problem("K")
    BM, BN = G.config("BM"), G.config("BN")

    # peak_flops is a spec-sheet MFMA rate: the MFMA shape and dtype it
    # assumes are a configuration property, not one hardware_params can
    # derive for us, so it can't be read straight off hardware_t the way
    # hbm_read_bw below is. It is still bound per cycle, though -- see
    # machine()'s docstring -- so no clock enters this expression at all.
    peak_flops = G.hardware("peak_flops_per_cu")
    # hbm_read_bw, like peak_flops, is already bytes/cycle, so no clock is
    # needed to price memory traffic either.
    hbm_bandwidth = G.hardware("hbm_read_bw")
    active_cus = G.runtime("active_cus")

    # Decode this node's workgroup so boundary workgroups are charged for their
    # actual tile rather than a full BM x BN tile.
    n_tiles = G.ceil_div(N, BN)
    tile_m = G.floor_div(node.wg, n_tiles)
    tile_n = G.mod(node.wg, n_tiles)

    m0 = tile_m * BM
    n0 = tile_n * BN
    tile_m_size = G.minimum(BM, M - m0)
    tile_n_size = G.minimum(BN, N - n0)

    tile_flops = 2 * tile_m_size * tile_n_size * K
    tile_bytes = (
        tile_m_size * K * G.problem("a_element_bytes")
        + K * tile_n_size * G.problem("b_element_bytes")
        + tile_m_size * tile_n_size * G.problem("d_element_bytes")
    )

    compute_cycles = tile_flops / (peak_flops * G.config("compute_efficiency"))

    # Each active CU receives an equal share of aggregate HBM bandwidth in this
    # simple model.
    memory_cycles = (
        tile_bytes * active_cus / (hbm_bandwidth * G.config("memory_efficiency"))
    )

    return G.maximum(compute_cycles, memory_cycles)


def tiled_gemm_spec():
    """Describe a tiled GEMM independently of concrete dimensions and parameters."""
    M, N, K = G.problem("M"), G.problem("N"), G.problem("K")
    BM, BN = G.config("BM"), G.config("BN")

    A = G.allocation_t("A", [M, K])
    B = G.allocation_t("B", [K, N])
    C = G.allocation_t("C", [M, N])

    # One graph node represents one K-complete output tile.
    gemm = G.operation_t("gemm", G.ceil_div(M, BM) * G.ceil_div(N, BN))

    def decode(wg, ctx):
        """Decode a linear workgroup ID into an output-tile coordinate."""
        n_tiles = ceil_div(ctx.problem["N"], ctx.config["BN"])
        return divmod(wg, n_tiles)

    def a_indices(wg, iteration, ctx):
        """Return flattened A indices read by one output-tile workgroup."""
        p, c = ctx.problem, ctx.config
        mt, _ = decode(wg, ctx)
        m0 = mt * c["BM"]

        return [
            m * p["K"] + k
            for m in range(m0, min(m0 + c["BM"], p["M"]))
            for k in range(p["K"])
        ]

    def b_indices(wg, iteration, ctx):
        """Return flattened B indices read by one output-tile workgroup."""
        p, c = ctx.problem, ctx.config
        _, nt = decode(wg, ctx)
        n0 = nt * c["BN"]

        return [
            k * p["N"] + n
            for k in range(p["K"])
            for n in range(n0, min(n0 + c["BN"], p["N"]))
        ]

    def c_indices(wg, iteration, ctx):
        """Return flattened C indices produced by one output-tile workgroup."""
        p, c = ctx.problem, ctx.config
        mt, nt = decode(wg, ctx)
        m0 = mt * c["BM"]
        n0 = nt * c["BN"]

        return [
            m * p["N"] + n
            for m in range(m0, min(m0 + c["BM"], p["M"]))
            for n in range(n0, min(n0 + c["BN"], p["N"]))
        ]

    gemm.access_patterns = [
        G.read(A, G.index_offsets(a_indices)),
        G.read(B, G.index_offsets(b_indices)),
        G.write(C, G.index_offsets(c_indices)),
    ]

    # Every concrete node derived from this operation uses the same symbolic
    # function, evaluated with that node and the current runtime state.
    gemm.node_cost = gemm_node_cost

    return G.graph_spec_t([gemm])


def small_problem():
    """The document's problem, scaled down so the quadratic derivation is quick."""
    return G.problem_t(
        {
            "M": 512,
            "N": 512,
            "K": 256,
            "a_element_bytes": 2,
            "b_element_bytes": 2,
            "d_element_bytes": 2,
        }
    )


def candidate_configs():
    """The document's three tile shapes."""
    return [
        G.config_t(
            {
                "BM": 128,
                "BN": 128,
                "BK": 32,
                "occupancy": 2,
                "compute_efficiency": 0.80,
                "memory_efficiency": 0.75,
            }
        ),
        G.config_t(
            {
                "BM": 256,
                "BN": 128,
                "BK": 32,
                "occupancy": 2,
                "compute_efficiency": 0.85,
                "memory_efficiency": 0.78,
            }
        ),
        G.config_t(
            {
                "BM": 256,
                "BN": 256,
                "BK": 32,
                "occupancy": 1,
                "compute_efficiency": 0.90,
                "memory_efficiency": 0.80,
            }
        ),
    ]


# The clock the spec sheet's 1.3e15 FLOP/s and 5.3e12 B/s numbers were quoted
# at. Used only to convert those two per-second figures into per-cycle ones,
# once, here -- not the clock of whatever part `machine()` is asked to model,
# which is what makes the results below invariant to that part's own clock.
_SPEC_SHEET_CLOCK_HZ = 2.1e9


def machine(clock_ghz=2.1):
    """Hardware bindings for an MI300X-shaped part, optionally clocked differently.

    `hardware_params` publishes what the device unambiguously reports. A peak
    matrix-core rate is not among those: it depends on the MFMA shape and data
    type a kernel selected, which is a configuration property, so the model
    binds it here rather than inheriting a number that would be wrong for its
    instruction mix. It is bound *per cycle*, from the spec sheet's own clock
    rather than `clock_ghz`, so this is a hardware constant like
    `peak_flops_per_cycle` in cost.hpp -- not a per-second figure re-diluted
    by whatever clock this particular part happens to run at. See
    gemm_node_cost's comment for why that distinction matters.
    """
    hardware = origami.comm.hardware_t()
    hardware.num_cu = 304
    hardware.num_xcd = 8
    hardware.cu_per_xcd = 38
    hardware.clock_ghz = clock_ghz
    # bytes per cycle, aggregate; not scaled by clock_ghz, since HBM bandwidth
    # is not the compute engine's DVFS domain.
    hardware.hbm_read_bw = 5.3e12 / _SPEC_SHEET_CLOCK_HZ

    params = G.hardware_params(hardware)
    # dense bf16 MFMA, per CU, per cycle.
    params["peak_flops_per_cu"] = (1.3e15 / 304) / _SPEC_SHEET_CLOCK_HZ
    return params


# ─── the authoring layer ──────────────────────────────────────────────


def test_one_spec_instantiates_to_many_graphs():
    spec = tiled_gemm_spec()
    problem = small_problem()

    graphs = [
        spec.instantiate(
            problem=config_and_name[0],
            config=config_and_name[1],
            name=config_and_name[2],
        )
        for config_and_name in [
            (problem, c, f"BM{c['BM']}_BN{c['BN']}_BK{c['BK']}")
            for c in candidate_configs()
        ]
    ]

    assert len(graphs) == 3
    # 512/128 x 512/128 = 16 tiles, then 2x4 = 8, then 2x2 = 4.
    assert [g.wg_count(0) for g in graphs] == [16, 8, 4]
    assert graphs[0].name() == "BM128_BN128_BK32"


def test_a_graph_remembers_the_namespaces_it_was_built_from():
    spec = tiled_gemm_spec()
    graph = spec.instantiate(problem=small_problem(), config=candidate_configs()[0])

    assert graph.problem()["M"] == 512
    assert graph.config()["BM"] == 128
    assert graph.config()["compute_efficiency"] == pytest.approx(0.80)


def test_a_config_reads_as_a_mapping():
    config = candidate_configs()[0]

    assert config["BM"] == 128
    assert "BK" in config
    assert dict(config)["occupancy"] == 2


def test_the_two_namespaces_stay_distinct():
    problem = G.problem_t({"size": 1024})
    config = G.config_t({"size": 64})

    op = G.operation_t("op", G.ceil_div(G.problem("size"), G.config("size")))
    graph = G.graph_spec_t([op]).instantiate(problem=problem, config=config)

    assert graph.wg_count(0) == 16


# ─── the consumer ─────────────────────────────────────────────────────


def test_ranking_binds_hardware_once_and_prices_the_spec():
    spec = tiled_gemm_spec()
    problem = small_problem()
    configs = candidate_configs()

    graphs = [
        spec.instantiate(
            problem=problem,
            config=config,
            name=f"BM{config['BM']}_BN{config['BN']}_BK{config['BK']}",
        )
        for config in configs
    ]

    schedule = G.graph_config_t()
    schedule.runtime = G.runtime_kind_t.event_driven
    schedule.name = "hardware scheduler"

    ranked = G.rank_graphs(graphs, [schedule], hardware=machine())

    assert len(ranked) == len(configs)
    # Every candidate was priced, which only happens if hardware reached the
    # specification's cost expressions.
    assert all(result.latency > 0.0 for result in ranked)
    # Best first.
    assert [r.latency for r in ranked] == sorted(r.latency for r in ranked)
    # And the winner names a real candidate.
    assert ranked[0].graph_name in {g.name() for g in graphs}
    assert dict(configs[ranked[0].graph_index])


def test_lanes_are_filled_from_the_device():
    graph = tiled_gemm_spec().instantiate(
        problem=small_problem(), config=candidate_configs()[0]
    )

    schedule = G.graph_config_t()
    schedule.runtime = G.runtime_kind_t.event_driven

    ranked = G.rank_graphs([graph], [schedule], hardware=machine())

    assert ranked[0].config.lanes == 304


def test_a_node_costs_more_under_contention():
    graph = tiled_gemm_spec().instantiate(
        problem=small_problem(), config=candidate_configs()[0]
    )
    assert G.attach_expr_cost(graph, machine())

    node = G.wg_node_t()
    node.op, node.wg, node.it = 0, 0, 0

    alone = graph.cost().node_cycles_at(node, 1)
    crowded = graph.cost().node_cycles_at(node, 304)
    assert crowded > alone


def test_compute_cycles_are_invariant_to_the_bound_clock():
    """The property the single-clock fixtures elsewhere in this file cannot
    see: pricing the same spec against two parts differing only in clock must
    yield identical cycles. `peak_flops_per_cu` is bound per cycle in
    machine(), from the spec sheet's own fixed clock rather than the part's,
    which is what makes this hold; if the bind ever regressed to multiplying
    a frozen per-second number by each part's own `clock_hz`, a 1.7 GHz part
    would silently understate cycles relative to a 2.1 GHz one, and this
    would catch it.
    """
    problem = small_problem()
    config = candidate_configs()[0]

    def priced_alone(clock_ghz):
        graph = tiled_gemm_spec().instantiate(problem=problem, config=config)
        assert G.attach_expr_cost(graph, machine(clock_ghz=clock_ghz))

        node = G.wg_node_t()
        node.op, node.wg, node.it = 0, 0, 0
        return graph.cost().node_cycles(node)

    assert priced_alone(2.1) == pytest.approx(priced_alone(1.7))


def test_a_boundary_tile_is_cheaper_than_a_full_one():
    # 300 is two 128-tiles and a 44-wide remainder, so the last tile in a row
    # owns less work than the first.
    problem = G.problem_t(
        {
            "M": 300,
            "N": 300,
            "K": 128,
            "a_element_bytes": 2,
            "b_element_bytes": 2,
            "d_element_bytes": 2,
        }
    )
    graph = tiled_gemm_spec().instantiate(
        problem=problem, config=candidate_configs()[0]
    )
    assert G.attach_expr_cost(graph, machine())

    assert graph.wg_count(0) == 9  # 3x3 tiles

    def cost_of(wg):
        node = G.wg_node_t()
        node.op, node.wg, node.it = 0, wg, 0
        return graph.cost().node_cycles(node)

    assert cost_of(8) < cost_of(0)
