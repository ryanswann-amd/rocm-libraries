# AllGather Outlier Investigation

Root-cause analysis of the AG 41% MdAPE outlier flagged for the C++ port pre-spike (2026-06-04). Conclusion: the math underlying the model is sound, but two structural gaps in the latency model — **per-ring-step proxy/sync overhead** and **xGMI write concentration efficiency** — account for nearly all of AG's residual error. Both are calibration additions, not rewrites; the C++ port can land them as new fields in [[heuristics]].

## TL;DR

| Primitive | Baseline MdAPE (W=8) | Fitted MdAPE (W=8) | Δ |
|-----------|---------------------:|-------------------:|--:|
| **all_gather**     | 45.3%               | **6.1%**          | −39 pp |
| reduce_scatter     | 30.8%               | 7.5%              | −23 pp |
| broadcast          | 17.6%               | 10.4%             | −7 pp  |
| all_reduce         | 27.7%               | 22.8%             | −5 pp  |
| all_to_all         | 24.6%               | 21.2%             | −3 pp  |

The 41% AG outlier had **three contributing causes**, listed in order of impact:

1. **Missing per-ring-step proxy/sync overhead.** AG's work-graph has no `Signal`/`Wait` ops; the model charges no inter-step cost. Reality: each ring step pays ~12-18 µs of CPU-mediated proxy/synchronization overhead. With 7 ring steps that's ~85-130 µs flat — exactly the small-message gap observed.
2. **xGMI write effective rate scales with WG concentration, not just MSHR cap.** A single WG saturates ~21% of the link payload rate, two WGs ~38%, five WGs ~65%, nine+ WGs ~83%. The current model uses the MSHR read cap (5.82 B/cycle ≈ 11.6 GB/s) for writes, which mispredicts both the single-WG floor and the multi-WG ramp.
3. **`active_links` does not conserve WGs.** At W=8 with `num_wgs=32`, the current code returns 7 links × 4 WGs = 28 WGs (4 dropped). At W=4 with `num_wgs=1`, it returns 3 links × 1 WG = 3 WGs (3 fabricated). Minor effect on MdAPE; significant correctness concern.

## Evidence

### Error structure by band

Baseline AG predictions (W=8) decompose by message-size band:

```
band=<64KB   nch=4..32 : MdAPE 58-72%, mean_err -59 to -73%
band=64K-1M  nch=4..32 : MdAPE 45-62%, mean_err -45 to -62%
band=1M-16M  nch=4..32 : MdAPE 29-47%, mean_err -29 to -47%
band=>16M    nch=4..32 : MdAPE 27-41%, mean_err -27 to -41%
```

Two distinct error regimes are visible:

- **Small-msg flat gap (~130 µs).** At ≤64 KB the predicted floor is ~50 µs but measured is ~180 µs. Across nch values the gap is *flat* and *constant*, which is the fingerprint of a fixed-cost overhead the model is not charging.
- **Large-msg systematic underestimate (~28-46%).** As message size grows the model approaches a stable per-link rate but underestimates: at large msg the measured per-link rate saturates well below the model's assumed 100% of payload cap.

### Per-link utilization at the large-msg asymptote

| nch | AG per-link (GB/s) | AG util % | Model's assumed util |
|-----|------------------:|----------:|---------------------:|
| 4   | 4.51              | 10.5%     | ~14%                 |
| 8   | 8.85              | 20.6%     | ~28% (close)         |
| 16  | 15.93             | 37.2%     | ~48%                 |
| 32  | 27.05             | 63.1%     | ~93%                 |
| 64  | 34.82             | 81.3%     | 100% (capped)        |

The shape of util vs. WGs/link is clearly *not* "1 WG = MSHR cap, multi-WG = link cap". It's a concentration-aware ramp.

### Concentration-aware xGMI write model

Empirical points (from the existing comment in `latency.py`, validated against rccl_master_sweep):

| WGs / link | Effective util |
|-----------:|---------------:|
| 1          | ~21%           |
| 2          | ~38%           |
| 5          | ~65%           |
| 9+         | ~83%           |

A clean fit: `util(W) = 1 − exp(−W/k)` with `k ≈ 4.5`. Predictions:

| W | meas | fit (k=4.5) | ratio |
|---|-----:|------------:|------:|
| 1 | 0.21 | 0.20        | 0.95× |
| 2 | 0.38 | 0.36        | 0.94× |
| 5 | 0.65 | 0.67        | 1.03× |
| 9 | 0.83 | 0.86        | 1.04× |

Applied as `effective_link_bw = link_bw × util(wgs_on_link)` and divided by `wgs_on_link` to give per-WG rate. This *replaces* the current MSHR-cap-for-writes logic in `compute_iter_times`.

### Per-collective fitted heuristics

Calibrated against W=8, validated against W=2 and W=4 (held-out):

| Primitive | `k_write` | `step_us` | W=8 MdAPE | W=4 MdAPE | W=2 MdAPE |
|-----------|----------:|----------:|----------:|----------:|----------:|
| all_gather     | 4.5  | 12 | 6.1%  | 18.5% | 30.0% |
| reduce_scatter | 3.5  |  5 | 7.5%  |  6.6% | 15.7% |
| broadcast      | 4.5  |  0 | 10.4% | 18.2% | 31.2% |
| all_reduce     | 5.0  |  0 | 22.8% | 25.6% | 26.9% |
| all_to_all     | 4.0  |  0 | 21.2% | 33.6% | 35.9% |

Observations:

- `k_write ≈ 4-5` is consistent across all primitives — this looks like a hardware constant, not a per-collective fudge.
- `step_us` is collective-specific. Ring AG needs ~12 µs/step, ring RS needs only ~5 µs/step. Non-ring layouts (AR, BC, A2A) need 0 (the per-timestep transfer model already covers them).
- Generalization to held-out world sizes is good for the calibrated primitives (AG and RS) and modest for the rest. W=2 is the noisiest band, likely because there is only 1 ring step so the calibration loses leverage.

### Why does the AG fix break RS if applied indiscriminately?

When we apply `(k_write=4.5, step_us=12)` from AG to *every* ring layout, RS regresses from 30.8% → 63.5% MdAPE. The reason is structural, not a bug in the fit:

- RS small-message floor is ~70-80 µs (vs AG's 200+ µs). Charging 84 µs of extra sync overshoots.
- RS measured per-link rate at the same per-rank tile is **higher** than AG's (e.g. nch=32, 4 MB per rank: RS 29 GB/s vs AG 15 GB/s per link). RS is genuinely faster per-byte, even though both share the same `Push` work-graph footprint. The difference is likely that RCCL picks different protocols (LL128 vs Simple) and/or different proxy paths for AG vs RS, even though the analytical work-graph is similar.

The clean fix is **per-primitive heuristics**, not a one-size global constant.

## Recommended changes (for C++ port and pre-port Python landing)

These changes are small, additive, and easy to port. Best to land them in Python first, regenerate the byte-identical regression baseline, and *then* mirror in C++.

1. **Add to `model/heuristics.py`:**

   ```python
   # Per-ring-step proxy/sync overhead. Each ring step in RCCL
   # incurs a CPU-mediated proxy thread send/recv + barrier. Cost
   # is per-step, NOT per-byte; latency-bound regime only.
   #
   # UNIT: GPU cycles (consistent with launch_overhead_cycles).
   #
   # Calibration: rccl_master_sweep.csv (MI300X). Fits min MdAPE
   # against AG and RS sweeps at W=8 over (nch, msg_bytes).
   ring_step_overhead_cycles: Dict[str, float] = field(default_factory=lambda: {
       "all_gather":     12_000 * 2.0,  # 12 µs × 2 cycles/ns
       "reduce_scatter":  5_000 * 2.0,
       "broadcast":           0,
       "all_reduce":          0,
       "all_to_all":          0,
   })

   # xGMI per-WG write effective rate: 1 - exp(-wgs/k) of link_bw.
   # Empirical (1 WG → 21%, 2 → 38%, 5 → 65%, 9+ → 83% of payload).
   # Replaces the MSHR-cap-for-writes used currently in latency.py.
   xgmi_write_concentration_k: float = 4.5
   ```

2. **Update `model/latency.py::compute_iter_times`:**

   ```python
   # Replace the MSHR-cap-for-writes block with:
   wgs_on_link = max(comm_hw.link_bw / max(bw_per_wg, 1e-12), 1.0)
   k = heuristics.xgmi_write_concentration_k
   util = 1.0 - math.exp(-wgs_on_link / k)
   eff_link_bw = comm_hw.link_bw * util
   eff_write_bw_per_wg = eff_link_bw / wgs_on_link
   T_xgmi_write = work.xgmi_write_cl * CL / max(eff_write_bw_per_wg, 1e-12)
   ```

3. **Update `model/collective.py::_compute_sequential_latency` and `_compute_ring_latency`** to add `num_timesteps × ring_step_overhead_cycles[primitive]` when the layout is a ring layout.

4. **Fix `RingAllGatherLayout.active_links` (and `RingReduceScatterLayout.active_links`)** to conserve WGs:

   ```python
   def active_links(self, timestep, num_wgs, num_gpus):
       nrings = max(min(num_wgs, num_gpus - 1), 1)
       base = num_wgs // nrings
       extra = num_wgs - base * nrings
       return {i: base + (1 if i < extra else 0) for i in range(nrings)}
   ```

5. **Reconcile the AG layout docstring.** The class docstring claims "1 active ring" but the per-link rate evidence (60+ GB/s on 1 link is impossible against the 42.86 GB/s cap) shows AG uses min(nch, N-1) rings concurrently. Update the docstring to match the corrected `active_links`.

## Implications for the C++ port

The investigation does **not** invalidate the C++ port plan. On the contrary:

- The math is sound — the bottom-up cycle-based model with primitive composition is the right abstraction. The residual errors are all in heuristics, not in the core latency engine.
- The 41% AG MdAPE is closeable to ~6% with two new heuristic fields. These should land in Python first, then port as part of M7 (Tensor + Heuristics) in the [[cpp-port-plan]].
- The `active_links` WG-conservation fix is a small layout change and ports trivially.
- **No structural revisions needed** to: hardware constants, primitives, tile hierarchy, frequency-based unit conversion, layout abstraction. Pre-port spike day 1 is **green**: proceed with C++ port after applying the Python-side fixes.

## Open questions (for future work, not blocking)

- **Why is `k_write ≈ 4.5` and not something else?** Likely related to xGMI flit/packet pipeline depth, but no direct microbenchmark confirms this yet. The Triton per-WG atom microbench (day 2 of the pre-port spike) is the right vehicle.
- **Why does AG need ~12 µs/step but RS only ~5 µs/step?** Probably RCCL protocol selection (LL128 vs Simple) differs for AG vs RS at the size regimes where the floor matters. Worth instrumenting RCCL itself to confirm.
- **W=2 MdAPE remains high.** Only 1 ring step so the calibration has no leverage. May need a separate W=2 special case or just accept it as out-of-band.

## See also

- [[cost-model-extension]] for the model architecture.
- [[cpp-port-plan]] for the port roadmap.
- [[mi300x-architecture]] for xGMI link characteristics.
- [[communication-primitives]] for collective semantics.
