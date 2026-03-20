#!/usr/bin/env python3
"""Feature extraction for the residual MLP.

Extracts 92 features per (problem, kernel) pair:
- Origami analytical (6): l_compute, l2_hit, mall_hit, n_mi, n_output_tiles, log(origami_latency)
- Tile geometry (11): mt_m/n/k (raw + log), mt_area, mt_volume, mt_aspect, mi_m, mi_n
- Problem-tile interaction (12): grid_m/n, total_tiles, k_iters, util_m/n/mn,
  cu_occupancy, arith_intensity, coverage_frac, tile_efficiency
- CU wave analysis (5): cu_waves, full_waves, wave_frac, wave_efficiency, cu_wave_tail
- Alignment (6): m/n/k_divides, m/n/k_waste
- Problem shape (5): log_m/n/k, mn_ratio, mk_ratio
- Tensile params (42): AG, DTLA/B, PLR, PGR, GSU, VWA/B, GRVWA/B, NTA/B, WS, SK, etc.
- Dtype/transpose (5): dtype_bits, is_bf16, is_f16, is_transA, is_transB
"""

import logging
import math
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Ordered list of feature names — defines the model input contract.
FEATURE_NAMES = [
    # Origami analytical (6)
    'l_compute', 'l2_hit', 'mall_hit', 'n_mi', 'n_output_tiles', 'log_origami_us',
    # Tile geometry (11)
    'mt_m', 'mt_n', 'mt_k', 'log_mt_m', 'log_mt_n', 'log_mt_k',
    'mt_area', 'mt_volume', 'mt_aspect', 'mi_m', 'mi_n',
    # Problem-tile interaction (12)
    'grid_m', 'grid_n', 'total_tiles', 'k_iters',
    'util_m', 'util_n', 'util_mn',
    'cu_occupancy', 'arith_intensity', 'coverage_frac', 'tile_efficiency',
    'cu_wave_tail',
    # CU wave analysis (5)
    'cu_waves', 'full_waves', 'wave_frac', 'wave_efficiency', 'remaining_tiles',
    # Alignment (6)
    'm_divides', 'n_divides', 'k_divides', 'm_waste', 'n_waste', 'k_waste',
    # Problem shape (5)
    'log_m', 'log_n', 'log_k', 'mn_ratio', 'mk_ratio',
    # Tensile params (37)
    'ag', 'dtla', 'dtlb', 'dtva', 'dtvb', 'plr', 'pgr', 'gsu',
    'ss', 'sso', 'svw', 'vwa', 'vwb', 'grvwa', 'grvwb', 'lrvw',
    'clr', 'sgrob', 'tin', 'spo', 'ntc', 'ntd', 'nta', 'ntb',
    'nepbs', 'ldsb', 'onll', 'ws', 'sk', 'dplb', 'nlca', 'nlcb',
    'has_cms', 'miwt_x', 'miwt_y', 'wg_x', 'wg_y',
    'lbsppa', 'lbsppb', 'lpa', 'lpb', 'afc',
    # Dtype/transpose (5)
    'dtype_bits', 'is_bf16', 'is_f16', 'is_transA', 'is_transB',
]

NUM_FEATURES = len(FEATURE_NAMES)


def _safe_log2(x: float) -> float:
    return math.log2(max(x, 1))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def get_clock_mhz(arch: str = 'gfx942') -> float:
    """Get clock speed in MHz for the given architecture."""
    clocks = {'gfx942': 1700.0, 'gfx950': 2100.0}
    return clocks.get(arch, 1700.0)


def make_hardware(arch: str = 'gfx942'):
    """Create origami hardware descriptor for the given architecture."""
    import origami

    arch_map = {
        'gfx942': (origami.architecture_t.gfx942, 304, 65536, 32768, 1700000),
        'gfx950': (origami.architecture_t.gfx950, 304, 65536, 32768, 2100000),
    }
    if arch not in arch_map:
        raise ValueError(f"Unsupported arch: {arch}. Supported: {list(arch_map.keys())}")

    arch_enum, n_cu, lds, l2, clock = arch_map[arch]
    return origami.get_hardware_for_arch(arch_enum, n_cu, lds, l2, clock)


def dtype_str_to_origami(dtype_str: str):
    """Convert hipBLASLt dtype string to origami data_type_t."""
    import origami
    mapping = {
        'bf16_r': origami.data_type_t.BFloat16,
        'f16_r': origami.data_type_t.Half,
        'f32_r': origami.data_type_t.Float,
    }
    return mapping.get(dtype_str, origami.data_type_t.BFloat16)


def dtype_bits(dtype_str: str) -> int:
    return {'bf16_r': 16, 'f16_r': 16, 'f32_r': 32, 'f64_r': 64, 'i8_r': 8}.get(dtype_str, 16)


def _get_mi_k_resolver(hw, mi_dtype):
    """Build a resolver for the actual MI_K dimension given (MI_M, MI_N)."""
    import origami
    valid_mis = hw.get_valid_matrix_instructions(mi_dtype)
    mi_k_options = {}
    for vmi in valid_mis:
        key = (vmi.m, vmi.n)
        if key not in mi_k_options:
            mi_k_options[key] = []
        mi_k_options[key].append(vmi.k)
    for key in mi_k_options:
        mi_k_options[key].sort()

    def resolve(mi_m, mi_n, mt_k):
        options = mi_k_options.get((mi_m, mi_n), [])
        if not options:
            return 1
        for k in reversed(options):
            if mt_k % k == 0:
                return k
        return options[0]

    return resolve


def extract_features_for_shape(
    hw, M: int, N: int, K: int,
    solutions: list[dict],
    dtype: str = 'bf16_r',
    transA: str = 'T', transB: str = 'N',
    n_cu: int = 304,
    clock_mhz: float = 1700.0,
) -> list[dict]:
    """Extract features and origami predictions for all solutions in a shape.

    Args:
        hw: origami hardware_t object
        M, N, K: problem dimensions
        solutions: list of dicts from bench.py (must have 'params', 'gflops', 'us')
        dtype: data type string
        transA, transB: transpose mode
        n_cu: number of CUs

    Returns:
        List of dicts with keys:
            {features, log_correction, gflops, us, kernel_sig, origami_us}
        features is a numpy array of shape (NUM_FEATURES,)
    """
    import origami

    odt = dtype_str_to_origami(dtype)
    mi_dtype = odt  # MI dtype = input dtype for bf16/f16

    p = origami.problem_t()
    p.size = origami.dim3_t(M, N, K)
    p.batch = 1
    p.a_transpose = origami.transpose_t.T if transA == 'T' else origami.transpose_t.N
    p.b_transpose = origami.transpose_t.N if transB == 'N' else origami.transpose_t.T
    p.a_dtype = odt
    p.b_dtype = odt
    p.c_dtype = odt
    p.d_dtype = odt
    p.mi_dtype = mi_dtype

    resolve_mi_k = _get_mi_k_resolver(hw, mi_dtype)

    is_transA = 1.0 if transA == 'T' else 0.0
    dbits = dtype_bits(dtype)

    # Problem-level features
    log_m = _safe_log2(M)
    log_n = _safe_log2(N)
    log_k = _safe_log2(K)
    mn_ratio = math.log2(max(M, 1) / max(N, 1)) if N > 0 else 0
    mk_ratio = math.log2(max(M, 1) / max(K, 1)) if K > 0 else 0
    flops = 2.0 * M * N * K
    bytes_read = (M * K + N * K) * (dbits / 8)
    arith_intensity = _safe_div(flops, bytes_read)

    results = []

    for sol in solutions:
        params = sol.get('params', {})
        if 'MT_M' not in params or 'MI_M' not in params:
            continue

        mt_m = params['MT_M']
        mt_n = params['MT_N']
        mt_k = params['MT_K']
        mi_m = params['MI_M']
        mi_n = params['MI_N']
        tensile_mi_k = params.get('MI_K', 1)

        # Resolve actual MI_K
        mi_k = resolve_mi_k(mi_m, mi_n, mt_k) if tensile_mi_k <= 1 else tensile_mi_k

        # Build origami config
        c = origami.config_t()
        c.mt = origami.dim3_t(mt_m, mt_n, mt_k)
        c.mi = origami.dim3_t(mi_m, mi_n, mi_k)
        c.occupancy = 4
        c.workgroup_mapping = 8

        if 'GRVWA' in params:
            c.grvw_a = params['GRVWA']
        if 'GRVWB' in params:
            c.grvw_b = params['GRVWB']
        if 'VWA' in params:
            c.vector_width_a = params['VWA']
        if 'VWB' in params:
            c.vector_width_b = params['VWB']
        if 'NTA' in params and params['NTA'] > 0:
            c.cache_hints_a = 4
        if 'NTB' in params and params['NTB'] > 0:
            c.cache_hints_b = 4

        try:
            tp = origami.tensile_params_t()
            if 'GSU' in params:
                tp.global_split_u = params['GSU']
            if 'DTLA' in params:
                tp.direct_to_lds_a = bool(params['DTLA'])
            if 'DTLB' in params:
                tp.direct_to_lds_b = bool(params['DTLB'])
            if 'DTVA' in params:
                tp.direct_to_vgpr_a = bool(params['DTVA'])
            if 'DTVB' in params:
                tp.direct_to_vgpr_b = bool(params['DTVB'])
            if 'PGR' in params:
                tp.prefetch_global_read = params['PGR']
            if 'NLCA' in params:
                tp.num_loads_coalesced_a = params['NLCA']
            if 'NLCB' in params:
                tp.num_loads_coalesced_b = params['NLCB']
            if 'MIWT_X' in params and 'MIWT_Y' in params:
                tp.wave_group_m = params['MIWT_X']
                tp.wave_group_n = params['MIWT_Y']
            if 'WG_X' in params and 'WG_Y' in params:
                wg_total = params['WG_X'] * params['WG_Y'] * params.get('WG_Z', 1)
                tp.wave_num = max(wg_total // 64, 1)
            c.set_tensile_params(tp)
        except Exception:
            pass

        # Origami predictions
        l_compute = 0.0
        l_l2_hit = 0.0
        l_mall_hit = 0.0
        n_output_tiles = 0
        n_mi_count = 0
        origami_us = 0.0

        try:
            n_mi_count = origami.compute_number_matrix_instructions(c.mt, c.mi)
            l_compute = float(origami.compute_mt_compute_latency(p, hw, c))
            l_l2_hit = origami.estimate_l2_hit(p, hw, c, 1)
            l_mall_hit = origami.estimate_mall_hit(p, hw, c, min(n_cu, 304), 1)
            n_output_tiles = origami.compute_number_of_output_tiles(
                c.mt.m, c.mt.n, p.size.m, p.size.n, p.batch)
        except Exception:
            pass

        try:
            latency_cycles = origami.compute_total_latency(p, hw, c, n_cu)
            if latency_cycles != float('inf') and latency_cycles > 0:
                # Convert from clock cycles to microseconds
                origami_us = latency_cycles / clock_mhz
        except Exception:
            pass

        actual_us = sol['us']
        if origami_us <= 0 or actual_us <= 0:
            continue

        log_correction = math.log(actual_us / origami_us)

        # Tile geometry
        mt_area = mt_m * mt_n
        mt_volume = mt_m * mt_n * mt_k
        mt_aspect = _safe_div(mt_m, mt_n, 1.0)

        # Problem-tile interaction
        grid_m = math.ceil(M / mt_m) if mt_m > 0 else 1
        grid_n = math.ceil(N / mt_n) if mt_n > 0 else 1
        total_tiles = grid_m * grid_n
        k_iters = math.ceil(K / mt_k) if mt_k > 0 else 1
        util_m = _safe_div(M, grid_m * mt_m, 1.0)
        util_n = _safe_div(N, grid_n * mt_n, 1.0)
        util_mn = util_m * util_n
        cu_occupancy = min(total_tiles / float(n_cu), 1.0)
        mn_product = max(M * N, 1)
        coverage_frac = mt_area / mn_product
        useful_output = M * N
        allocated_output = grid_m * mt_m * grid_n * mt_n
        tile_efficiency = _safe_div(useful_output, allocated_output, 1.0)

        # CU wave analysis
        cu_waves = total_tiles / float(n_cu)
        full_waves = int(cu_waves)
        remaining_tiles_val = total_tiles - full_waves * n_cu
        wave_frac = cu_waves - full_waves
        wave_efficiency = _safe_div(remaining_tiles_val, float(n_cu), cu_occupancy)
        cu_wave_tail = _safe_div(total_tiles % n_cu, n_cu, _safe_div(total_tiles, n_cu))

        # Alignment
        m_divides = 1.0 if (mt_m > 0 and M % mt_m == 0) else 0.0
        n_divides = 1.0 if (mt_n > 0 and N % mt_n == 0) else 0.0
        k_divides = 1.0 if (mt_k > 0 and K % mt_k == 0) else 0.0
        m_waste = _safe_div(grid_m * mt_m - M, max(M, 1)) if mt_m > 0 else 0
        n_waste = _safe_div(grid_n * mt_n - N, max(N, 1)) if mt_n > 0 else 0
        k_waste = 0.0
        if mt_k > 0:
            padded_k = math.ceil(K / mt_k) * mt_k
            k_waste = _safe_div(padded_k - K, max(K, 1))

        # Build feature vector in canonical order
        fvec = np.array([
            # Origami analytical (6)
            l_compute, l_l2_hit, l_mall_hit, n_mi_count, n_output_tiles,
            math.log(max(origami_us, 1e-9)),
            # Tile geometry (11)
            mt_m, mt_n, mt_k,
            _safe_log2(mt_m), _safe_log2(mt_n), _safe_log2(mt_k),
            mt_area, mt_volume, mt_aspect, mi_m, mi_n,
            # Problem-tile interaction (12)
            grid_m, grid_n, total_tiles, k_iters,
            util_m, util_n, util_mn,
            cu_occupancy, arith_intensity, coverage_frac, tile_efficiency,
            cu_wave_tail,
            # CU wave analysis (5)
            cu_waves, full_waves, wave_frac, wave_efficiency, remaining_tiles_val,
            # Alignment (6)
            m_divides, n_divides, k_divides, m_waste, n_waste, k_waste,
            # Problem shape (5)
            log_m, log_n, log_k, mn_ratio, mk_ratio,
            # Tensile params (37)
            params.get('AG', 0), params.get('DTLA', 0), params.get('DTLB', 0),
            params.get('DTVA', 0), params.get('DTVB', 0),
            params.get('PLR', 0), params.get('PGR', 2), params.get('GSU', 0),
            params.get('SS', 0), params.get('SSO', 0), params.get('SVW', 0),
            params.get('VWA', 1), params.get('VWB', 1),
            params.get('GRVWA', 1), params.get('GRVWB', 1), params.get('LRVW', 8),
            params.get('CLR', 0), params.get('SGROB', 0),
            params.get('TIN', 0), params.get('SPO', 0),
            params.get('NTC', 0), params.get('NTD', 0),
            params.get('NTA', 0), params.get('NTB', 0),
            params.get('NEPBS', 0), params.get('LDSB', 0), params.get('ONLL', 1),
            params.get('WS', 64), params.get('SK', 3),
            params.get('DPLB', 0), params.get('NLCA', 1), params.get('NLCB', 1),
            params.get('has_CMS', 0),
            params.get('MIWT_X', 2), params.get('MIWT_Y', 2),
            params.get('WG_X', 32), params.get('WG_Y', 8),
            params.get('LBSPPA', 0), params.get('LBSPPB', 0),
            params.get('LPA', 0), params.get('LPB', 0), params.get('AFC', 0),
            # Dtype/transpose (5)
            dbits,
            1.0 if dtype == 'bf16_r' else 0.0,
            1.0 if dtype == 'f16_r' else 0.0,
            is_transA,
            1.0 if transB == 'T' else 0.0,
        ], dtype=np.float32)

        assert len(fvec) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(fvec)}"

        results.append({
            'features': fvec,
            'log_correction': 0.0,  # placeholder, will be set to within-shape percentile
            'gflops': sol['gflops'],
            'us': actual_us,
            'origami_us': origami_us,
            'kernel_sig': sol.get('kernel_signature', ''),
        })

    return results
