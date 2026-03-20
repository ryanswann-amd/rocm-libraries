#!/usr/bin/env python3
"""hipblaslt-bench subprocess invocation and output parsing.

Runs hipblaslt-bench with --algo_method all to collect timing for every
available kernel solution on a given (M, N, K, dtype, transpose) problem.
Parses the CSV-like output and Tensile solution names into structured dicts.
"""

import csv
import io
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def parse_solution_params(sig: str) -> dict:
    """Extract key parameters from a Tensile solution name.

    Reuses the pattern from scripts/correlation_harness.py.
    """
    params = {}

    mt = re.search(r'MT(\d+)x(\d+)x(\d+)', sig)
    if mt:
        params['MT_M'] = int(mt.group(1))
        params['MT_N'] = int(mt.group(2))
        params['MT_K'] = int(mt.group(3))

    mi = re.search(r'MI(\d+)x(\d+)x(\d+)', sig)
    if mi:
        params['MI_M'] = int(mi.group(1))
        params['MI_N'] = int(mi.group(2))
        params['MI_K'] = int(mi.group(3))

    wg = re.search(r'WG(\d+)_(\d+)_(\d+)', sig)
    if wg:
        params['WG_X'] = int(wg.group(1))
        params['WG_Y'] = int(wg.group(2))
        params['WG_Z'] = int(wg.group(3))

    miwt = re.search(r'MIWT(\d+)_(\d+)', sig)
    if miwt:
        params['MIWT_X'] = int(miwt.group(1))
        params['MIWT_Y'] = int(miwt.group(2))

    for pname in [
        'AG', 'DTLA', 'DTLB', 'DTVA', 'DTVB', 'PLR', 'PGR', 'GSU',
        'SS', 'SSO', 'SVW', 'VWA', 'VWB', 'GRVWA', 'GRVWB', 'LRVW',
        'CLR', 'SGROB', 'TIN', 'SPO', 'NTC', 'NTD', 'NTA', 'NTB',
        'NEPBS', 'LDSB', 'ONLL', 'WS', 'SK', 'DPLB', 'NLCA', 'NLCB',
        'LBSPPA', 'LBSPPB', 'LPA', 'LPB', 'AFC',
    ]:
        m = re.search(rf'_{pname}(n?\d+)', sig)
        if m:
            val = m.group(1)
            if val.startswith('n'):
                params[pname] = -int(val[1:])
            else:
                params[pname] = int(val)

    params['has_CMS'] = 1 if '_CMS_' in sig or '_CMS' in sig else 0
    return params


def run_hipblaslt_bench(
    bench_path: str,
    M: int, N: int, K: int,
    transA: str = 'T', transB: str = 'N',
    dtype: str = 'bf16_r',
    iters: int = 20, cold_iters: int = 3,
    gpu_id: int = 0,
    timeout: int = 300,
) -> list[dict]:
    """Run hipblaslt-bench --algo_method all and parse all kernel solutions.

    Returns a list of dicts, one per kernel solution:
        {kernel_signature, macro_tile, gflops, us, algo_index, params}
    """
    cmd = [
        bench_path,
        '-m', str(M), '-n', str(N), '-k', str(K),
        '--transA', transA, '--transB', transB,
        '--a_type', dtype, '--b_type', dtype,
        '--c_type', dtype, '--d_type', dtype,
        '--compute_type', 'f32_r',
        '--algo_method', 'all',
        '--iters', str(iters), '--cold_iters', str(cold_iters),
        '--print_kernel_info', '--use_gpu_timer',
    ]

    env = dict(os.environ)
    env['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    env['HIPBLASLT_BENCH_PERF'] = '1'

    log.info("Bench %dx%dx%d on GPU %d", M, N, K, gpu_id)
    t0 = time.monotonic()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        log.warning("Bench %dx%dx%d timed out after %ds", M, N, K, timeout)
        return []
    except FileNotFoundError:
        log.error("hipblaslt-bench not found at %s", bench_path)
        return []

    elapsed = time.monotonic() - t0
    output = result.stdout + '\n' + result.stderr

    solutions = _parse_bench_output(output, M, N, K, dtype)
    log.info(
        "Bench %dx%dx%d: %d solutions in %.1fs",
        M, N, K, len(solutions), elapsed,
    )
    return solutions


def _parse_bench_output(output: str, M: int, N: int, K: int, dtype: str) -> list[dict]:
    """Parse hipblaslt-bench output into a list of solution dicts.

    hipblaslt-bench with --algo_method all prints a CSV header followed by
    one line per algorithm/kernel. Each line contains comma-separated fields.
    The --print_kernel_info flag adds kernel name info in stderr.
    """
    lines = output.strip().split('\n')
    solutions = []

    # Collect kernel signatures from --print_kernel_info lines
    kernel_sigs = []
    for line in lines:
        if '--Solution name:' in line:
            sig_match = re.search(r'--Solution name:\s*(\S+)', line)
            if sig_match:
                kernel_sigs.append(sig_match.group(1))

    # Find the CSV header line
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('transA,') or stripped.startswith('transA\t'):
            header_idx = i
            break

    if header_idx is None:
        # Try alternate: look for lines that start with T or N and have commas
        data_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0] in ('T', 'N') and ',' in stripped:
                data_lines.append(stripped)
        if not data_lines:
            return solutions
        # Parse data lines without header
        for idx, line in enumerate(data_lines):
            sol = _parse_data_line_positional(line, idx, kernel_sigs)
            if sol:
                solutions.append(sol)
        return solutions

    # Parse with header
    header_line = lines[header_idx].strip()
    # Handle both comma and tab separators
    sep = ',' if ',' in header_line else '\t'
    headers = [h.strip() for h in header_line.split(sep)]

    sig_idx = 0
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('#') or line.startswith('='):
            continue
        fields = [f.strip() for f in line.split(sep)]
        if len(fields) < len(headers):
            continue

        row = dict(zip(headers, fields))

        try:
            gflops = float(row.get('hipblaslt-Gflops', row.get('rocblas-Gflops', 0)))
            us = float(row.get('us', 0))
        except (ValueError, TypeError):
            continue

        if gflops <= 0 or us <= 0:
            continue

        # Get kernel signature
        sig = ''
        if sig_idx < len(kernel_sigs):
            sig = kernel_sigs[sig_idx]
        sig_idx += 1

        params = parse_solution_params(sig) if sig else {}
        mt_m = params.get('MT_M', 0)
        mt_n = params.get('MT_N', 0)
        mt_k = params.get('MT_K', 0)
        macro_tile = f"{mt_m}x{mt_n}x{mt_k}" if mt_m else ''

        solutions.append({
            'kernel_signature': sig,
            'macro_tile': macro_tile,
            'gflops': gflops,
            'us': us,
            'algo_index': len(solutions),
            'params': params,
        })

    return solutions


def _parse_data_line_positional(line: str, idx: int, kernel_sigs: list) -> Optional[dict]:
    """Parse a data line by position when no header is available."""
    fields = line.split(',')
    try:
        # hipblaslt-bench output format: transA,transB,M,N,K,...,hipblaslt-Gflops,us
        gflops = float(fields[-3])
        us = float(fields[-2]) if len(fields) > 2 else 0
    except (ValueError, IndexError):
        return None

    if gflops <= 0:
        return None

    sig = kernel_sigs[idx] if idx < len(kernel_sigs) else ''
    params = parse_solution_params(sig) if sig else {}
    mt_m = params.get('MT_M', 0)
    mt_n = params.get('MT_N', 0)
    mt_k = params.get('MT_K', 0)

    return {
        'kernel_signature': sig,
        'macro_tile': f"{mt_m}x{mt_n}x{mt_k}" if mt_m else '',
        'gflops': gflops,
        'us': us if us > 0 else (2.0 * int(fields[2]) * int(fields[3]) * int(fields[4]) / (gflops * 1e3)),
        'algo_index': idx,
        'params': params,
    }


def save_bench_csv(solutions: list[dict], path: Path, M: int, N: int, K: int):
    """Save benchmark results to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['kernel_signature', 'macro_tile', 'gflops', 'us']
    # Add all param keys from first solution
    param_keys = []
    if solutions:
        param_keys = sorted(solutions[0].get('params', {}).keys())
        fieldnames.extend(param_keys)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sol in solutions:
            row = {
                'kernel_signature': sol['kernel_signature'],
                'macro_tile': sol['macro_tile'],
                'gflops': sol['gflops'],
                'us': sol['us'],
            }
            for k in param_keys:
                row[k] = sol.get('params', {}).get(k, '')
            writer.writerow(row)
