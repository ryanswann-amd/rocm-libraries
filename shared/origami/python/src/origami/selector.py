# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import itertools
import logging
import math
from math import ceil
from typing import Iterable, Optional

import torch

import origami

"""
Origami: Analytical GEMM Solution Selection

Python bindings for the Origami C++ library.
"""

logger = logging.getLogger(__name__)

# ===========================================================================
# Triton-specific LDS estimation — only used when target_t == triton
# ===========================================================================

def estimate_triton_lds_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    bytes_a: float,
    bytes_b: float,
    num_stages: int = 2,
) -> float:
    """
    Estimate Triton kernel LDS (shared memory) usage in bytes for AMD GPUs.

    Triton's AMD backend uses swizzled_shared / amd_rotating_shared encodings
    which rearrange bank addressing without adding padding bytes.  The LDS
    footprint is therefore the raw tile bytes times the number of pipeline
    buffers:

      ns == 1:  max(A_bytes, B_bytes)   — no pipelining, sequential alloc
      ns >= 2:  (ns - 1) * (A_bytes + B_bytes)  — software-pipelined

    Validated against metadata.shared from compiled Triton kernels on gfx942
    (Triton 3.6.0+rocm7.2.0): 35/35 configs matched exactly.

    Args:
        block_m, block_n, block_k: Tile dimensions (MT_M, MT_N, MT_K).
        bytes_a, bytes_b: Bytes per element for A and B (e.g. 2 for bf16/fp16).
        num_stages: Pipeline stages (1, 2, or 3); Triton matmul uses 2 by default.

    Returns:
        Estimated total LDS usage in bytes.

    NOTE: This function is TRITON-SPECIFIC. It must only be called when
    target_t == triton in config_t. hipblaslt handles LDS internally.
    """
    a_bytes = block_m * block_k * bytes_a
    b_bytes = block_k * block_n * bytes_b
    if num_stages <= 1:
        return max(a_bytes, b_bytes)
    return (num_stages - 1) * (a_bytes + b_bytes)


def check_triton_lds_capacity(
    block_m: int,
    block_n: int,
    block_k: int,
    bytes_a: float,
    bytes_b: float,
    lds_capacity: int,
    num_stages: int = 2,
) -> bool:
    """Return True if estimated Triton LDS usage fits within lds_capacity.

    NOTE: TRITON-SPECIFIC — only call when target_t == triton.
    """
    usage = estimate_triton_lds_bytes(
        block_m, block_n, block_k, bytes_a, bytes_b, num_stages
    )
    return usage <= lds_capacity


class OrigamiMatmulSelector:
    """
    Analytical GEMM configuration selector for GPUs.
    
    This class uses the Origami analytical model to select optimal GEMM kernel
    configurations based on problem dimensions, data types, and hardware characteristics.
    It provides a high-level interface for integrating with PyTorch-based frameworks.
    
    The selector analyzes a set of candidate configurations and predicts their performance
    using analytical models of compute, memory, and synchronization costs. It returns
    the configuration with the lowest predicted latency.
    
    Attributes:
        dtype_to_str (dict): Mapping from PyTorch dtypes to Origami dtype strings.
    """
    # https://docs.pytorch.org/docs/stable/tensors.html
    dtype_to_str = {
        torch.float32: "f32",
        torch.complex64: "c32",
        torch.complex128: "c64",
        torch.float64: "f64",
        torch.float16: "f16",
        torch.int32: "i32",
        torch.bfloat16: "bf16",
        torch.int8: "i8",
        torch.float8_e5m2: "f8",
        torch.float8_e4m3fn: "f8",
    }
    # Add FP8 FNUZ variants if available (for non-gfx950 architectures)
    if hasattr(torch, "float8_e5m2fnuz"):
        dtype_to_str[torch.float8_e5m2fnuz] = "f8"
    if hasattr(torch, "float8_e4m3fnuz"):
        dtype_to_str[torch.float8_e4m3fnuz] = "f8"


    def __init__(
        self,
        config_gen: Iterable,
        m: int,
        n: int,
        k: int,
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
        a_stride: Optional[tuple[int, ...]] = None,
        b_stride: Optional[tuple[int, ...]] = None,
        batch: int = 1,
        mx_block_size=0,
        streamk=False
    ):
        """
        Initialize the Origami matmul configuration selector.
        
        Args:
            config_gen: Iterable of Triton-style config objects with kwargs containing
                       'BLOCK_M', 'BLOCK_N', 'BLOCK_K', and 'waves_per_eu'.
            m: M dimension of the GEMM (number of rows in A and output).
            n: N dimension of the GEMM (number of columns in B and output).
            k: K dimension of the GEMM (shared dimension between A and B).
            a_dtype: PyTorch dtype for matrix A.
            b_dtype: PyTorch dtype for matrix B.
            out_dtype: PyTorch dtype for output matrix.
            device: PyTorch CUDA device (must be ROCm-capable).
            a_stride: Tuple of PyTorch tensor strides for matrix A (default: None).
            b_stride: Tuple of PyTorch tensor strides for matrix B (default: None).
            batch: Batch size of the GEMM (default: 1).
            mx_block_size: Block size for MX format dtypes (default: 0 for non-MX types).
            streamk: Whether to use StreamK scheduling (default: False).
        
        Raises:
            RuntimeError: If no ROCm-capable device is detected.
            ValueError: If the hardware architecture is unsupported or data types are incompatible.
        """
        # Save tensor sizes
        self._m = m
        self._n = n
        self._k = k

        # Save tensor strides
        self._a_stride = a_stride
        self._b_stride = b_stride

        # Save batch size
        self._batch = batch

        # Save tensor dtypes as strings
        self._a_dtype_str   = OrigamiMatmulSelector.dtype_to_str.get(a_dtype, a_dtype)
        self._b_dtype_str   = OrigamiMatmulSelector.dtype_to_str.get(b_dtype, b_dtype)
        self._out_dtype_str = OrigamiMatmulSelector.dtype_to_str.get(out_dtype, out_dtype)
        
        # Save MX block size
        self._mx_block_size = mx_block_size

        # Helper function to get bits for both float, int, and MX dtypes
        mx_types = ["f4"]
        def get_dtype_bits(dtype):
            # Handle MX types (string-based)
            if dtype in mx_types:
                return origami.datatype_to_bits(origami.string_to_datatype(dtype))

            # Handle torch dtypes
            try:
                return torch.finfo(dtype).bits
            except TypeError:
                return torch.iinfo(dtype).bits
        self._a_dtype_bitsize = get_dtype_bits(a_dtype)
        self._b_dtype_bitsize = get_dtype_bits(b_dtype)
        self._out_dtype_bitsize = get_dtype_bits(out_dtype)

        # For matrix instruction latency lookup, use input dtype (not output dtype)
        # because the matrix instruction type is determined by input operand types
        # Example: FP8 inputs with BF16 output still uses FP8 matrix instructions
        # Set MI dtype - use string for MX types, otherwise lookup from dict
        if a_dtype in mx_types:
            self.mi_dtype = a_dtype
        else:
            input_dtype_for_mi = (
                a_dtype
                if get_dtype_bits(a_dtype) <= get_dtype_bits(b_dtype)
                else b_dtype
            )
            self.mi_dtype = OrigamiMatmulSelector.dtype_to_str.get(
                input_dtype_for_mi, OrigamiMatmulSelector.dtype_to_str.get(out_dtype)
            )

        # Get hardware info from Origami
        self._hardware = origami.get_hardware_for_device(device.index)
        self._N_CU = self._hardware.N_CU
        
        # Create Origami problem_t based on problem metadata
        self._problem = self._make_problem()

        # Create list of Origami config_t objects based on generator.
        self._configs = self._generate_configs(config_gen)

        # Run Origami solution selection
        self._result = origami.select_config(self._problem,
                                             self._hardware,
                                             self._configs)

        if streamk:
            self._grid = origami.select_grid_size(self._problem,
                                                  self._hardware,
                                                  self._result.config,
                                                  origami.grid_selection_t.k_split_aware,
                                                  self._hardware.N_CU)
        else:
            self._grid = origami.select_grid_size(self._problem,
                                                  self._hardware,
                                                  self._result.config,
                                                  origami.grid_selection_t.data_parallel,
                                                  self._hardware.N_CU)

        self._workgroup_mapping = (
            origami.select_workgroup_mapping(self._problem,
                                             self._hardware,
                                             self._result.config,
                                             self._grid)
        )


    @property
    def macrotile_m(self):
        """
        M dimension of the selected macrotile (block size in M dimension).
        
        Returns:
            int: Number of rows processed per workgroup.
        """
        return self._result.config.mt.m


    @property
    def macrotile_n(self):
        """
        N dimension of the selected macrotile (block size in N dimension).
        
        Returns:
            int: Number of columns processed per workgroup.
        """
        return self._result.config.mt.n


    @property
    def macrotile_k(self):
        """
        K dimension of the selected macrotile (block size in K dimension).
        
        Returns:
            int: Number of elements in the reduction dimension processed per iteration.
        """
        return self._result.config.mt.k


    @property
    def wgm(self):
        """
        Workgroup mapping parameter for tiling in the M-N plane.
        
        This controls how output tiles are grouped together for better cache locality.
        
        Returns:
            int: Workgroup mapping size.
        """
        return self._workgroup_mapping.wgm
    
    @property
    def wgmxcc(self):
        """
        Workgroup mapping size across XCCs (chiplets).
        
        For multi-chiplet GPUs (e.g., MI300 series), this controls how work is
        distributed across the different chiplets.
        
        Returns:
            int: Number of workgroups mapped per XCC.
        """
        return self._workgroup_mapping.wgmxcc
    
    @property
    def wgmxccchunk(self):
        """
        Workgroup mapping chunk size for XCC distribution.
        
        Controls the granularity of work distribution across chiplets.
        
        Returns:
            int: Chunk size for XCC workgroup mapping.
        """
        return self._workgroup_mapping.wgmxccchunk

    @property
    def number_of_cus(self):
        """
        Number of compute units (CUs) available on the target GPU.
        
        This is a hardware property that indicates the total parallelism available.
        Common values:
        - MI200 (gfx90a): 104 CUs
        - MI300A (gfx942): 228 CUs
        - MI300X (gfx942): 304 CUs
        - MI350 (gfx950): 256 CUs
        
        Returns:
            int: Total number of compute units on the device.
        """
        return self._hardware.N_CU


    @property
    def occupancy(self):
        """
        Number of wavefronts resident per SIMD unit (occupancy).
        
        Higher occupancy can hide memory latency but reduces register availability.
        Typical values are 1-4 for efficient schedules, though hardware supports more.
        
        Returns:
            int: Number of concurrent wavefronts per EU (1-64, typically 1-2 for performance).
        """
        return self._result.config.occupancy


    @property
    def even_k(self):
        """
        Whether the K dimension is evenly divisible by the macrotile K size.
        
        When True, the kernel can avoid bounds checking in the K loop, improving performance.
        When False, the kernel must handle the remainder iteration.
        
        Returns:
            bool: True if K is evenly divisible by macrotile_k, False otherwise.
        """
        return math.gcd(self._k, self.macrotile_k) == self.macrotile_k


    @property
    def grid_size(self):
        """
        Grid size (number of workgroups) to launch for the kernel.
        
        For StreamK scheduling, this is analytically determined to balance
        load across CUs while minimizing synchronization overhead.
        
        Returns:
            int: Number of workgroups to launch.
        """
        return self._grid


    # High-performance tile shortlists per GEMM shape category.
    # Derived from greedy set-cover analysis over 2287 MI300X GPU-measured shapes
    # (80K+ kernel timing rows). Each category's tiles are chosen to minimize
    # worst-case regret: the full set achieves mean=0.18%, max=9.78% regret
    # when origami picks the best tile within the category subset.
    # Tile format: (BLOCK_M, BLOCK_N, BLOCK_K)
    _TILE_SHORTLISTS = {
        # M <= 4: decode / single-token inference
        'decode': {
            (16, 64, 128), (16, 16, 256), (32, 32, 256), (16, 128, 128),
            (32, 32, 32), (16, 32, 128), (64, 64, 128), (16, 64, 64),
        },
        # M in [5, 128]: small batch
        'small_batch': {
            (16, 64, 128), (64, 64, 128), (128, 64, 128), (64, 64, 64),
            (16, 16, 256), (128, 128, 128), (64, 128, 128), (16, 32, 128),
            (64, 64, 256),
        },
        # M in [129, 1024]: medium batch / prefill
        'med_batch': {
            (128, 128, 128), (128, 64, 128), (64, 64, 128), (128, 256, 64),
            (16, 64, 128), (128, 64, 64), (64, 128, 128), (256, 256, 64),
            (256, 128, 64), (64, 128, 64),
        },
        # M > 1024: large batch / training
        'large': {
            (128, 128, 128), (128, 64, 64), (128, 128, 32), (64, 64, 64),
            (64, 64, 128), (256, 256, 64), (128, 256, 64), (128, 64, 128),
        },
    }

    @staticmethod
    def _get_shape_category(m: int) -> str:
        """Classify GEMM M dimension into a shape category for tile prefiltering."""
        if m <= 4:
            return 'decode'
        elif m <= 128:
            return 'small_batch'
        elif m <= 1024:
            return 'med_batch'
        else:
            return 'large'

    def _generate_configs(self, config_gen) -> [origami.config_t]:
        """
        Convert Triton-style configs to Origami config_t objects with
        dimension-aware prefiltering.

        First materializes all configs, then applies a shape-category-aware
        tile shortlist filter to restrict the candidate set. This dramatically
        reduces regret by removing tiles that are known to be poor choices
        for the problem's M dimension category. If no configs survive the
        filter, falls back to the full unfiltered set.

        Args:
            config_gen: Iterable of config objects with kwargs containing
                       'BLOCK_M', 'BLOCK_N', 'BLOCK_K', and 'waves_per_eu'.

        Returns:
            list[origami.config_t]: List of Origami configuration objects.
        """
        # Get recommended matrix instruction dimensions
        mi = self._hardware.get_recommended_matrix_instruction(self._problem.mi_dtype)

        # Materialize all configs first so we can filter
        all_configs = []
        for config in config_gen:
            bm = config.kwargs['BLOCK_M']
            bn = config.kwargs['BLOCK_N']
            bk = config.kwargs['BLOCK_K']

            mt = origami.dim3_t(bm, bn, bk)

            new_config           = origami.config_t()
            new_config.mt        = mt
            new_config.mi        = mi
            new_config.occupancy = config.kwargs['waves_per_eu']

            all_configs.append(((bm, bn, bk), new_config))

        # Apply dimension-aware tile prefilter
        category = self._get_shape_category(self._m)
        shortlist = self._TILE_SHORTLISTS.get(category)

        if shortlist is not None:
            filtered = [cfg for tile_key, cfg in all_configs
                        if tile_key in shortlist]
            # Fall back to full set if filter removes everything
            if filtered:
                return filtered

        return [cfg for _, cfg in all_configs]


    def _make_problem(self) -> origami.problem_t:
        """
        Create an Origami problem_t object from the GEMM problem specification.
        
        Converts PyTorch-style problem parameters (dimensions, dtypes) into
        Origami's internal problem representation. Assumes standard GEMM
        operation: D = A^T @ B + C (with A transposed).
        
        Returns:
            origami.problem_t: Problem specification for Origami analytical model.
        """
        # Create special dim3_t object for problem sizes
        size = origami.dim3_t(self._m, self._n, self._k)

        # Convert torch dtypes to Origami dtypes based on problem metadata
        a_origami_dtype = origami.string_to_datatype(self._a_dtype_str)
        b_origami_dtype = origami.string_to_datatype(self._b_dtype_str)
        c_origami_dtype = origami.string_to_datatype(self._out_dtype_str)

        # Calculate transpose types based on tensor strides
        a_transpose = self._check_transpose_type((self._m, self._k),
                                                 self._a_stride,
                                                 default=origami.transpose_t.T)
        b_transpose = self._check_transpose_type((self._k, self._n),
                                                 self._b_stride,
                                                 default=origami.transpose_t.N)

        # Create and set new problem_t values
        problem = origami.problem_t()
        problem.size            = size
        problem.batch           = self._batch
        problem.a_transpose     = a_transpose
        problem.b_transpose     = b_transpose
        problem.a_dtype         = a_origami_dtype
        problem.b_dtype         = b_origami_dtype
        problem.c_dtype         = c_origami_dtype
        problem.d_dtype         = c_origami_dtype
        problem.mi_dtype        = c_origami_dtype
        problem.a_mx_block_size = self._mx_block_size
        problem.b_mx_block_size = self._mx_block_size
    
        return problem


    def _check_transpose_type(
        self,
        shape: tuple[int, int],
        strides: Optional[tuple[int, ...]] = None,
        default: Optional[origami.transpose_t] = origami.transpose_t.N
    ) -> origami.transpose_t:
        """
        Determine transpose type from tensor shape and stride pattern.

        Analyzes the memory layout of a matrix by examining its strides to determine
        whether it is stored in row-major (non-transposed) or column-major (transposed)
        format. If strides are not provided, returns the default transpose type.

        Args:
            shape: Tuple of (m, n) dimensions for the matrix.
            strides: Optional tuple of PyTorch tensor strides. If None, the default
                    transpose type is returned without analysis.
            default: Transpose type to use when strides is None or as a starting point
                    (default: origami.transpose_t.N).

        Returns:
            origami.transpose_t: The determined transpose type.

        Raises:
            ValueError: If the matrix has a non-dense memory layout or an unsupported 
                       stride pattern.
        """
        # Start with supplied default transpose type as fallback
        transpose_t = default

        if strides is not None:
            m, n = shape
            # Make sure we check the last 2 values of the stride in case this
            # is a high-order tensor
            stride_m, stride_n = strides[-2], strides[-1]

            # Degenerate cases
            if m == 1 or n == 1:
                if stride_n == 1:
                    transpose_t = origami.transpose_t.N
                elif stride_m == 1:
                    transpose_t = origami.transpose_t.T
                else:
                    raise ValueError(
                        f"[ROCm:ORIGAMI]: A matrix with strides ({stride_m}, "
                        f"{stride_n}) appears to be a non-dense vector in memory."
                    )
            # Row-major dense case
            elif stride_n == 1:
                transpose_t = origami.transpose_t.N
            # Tranposed-view of row-major dense case
            elif stride_m == 1:
                transpose_t = origami.transpose_t.T
            # Unknown stride pattern
            else:
                raise ValueError(
                    f"[ROCm:ORIGAMI] A matrix with strides ({stride_m}, "
                    f"{stride_n}) is an unsupported layout." 
                )

        return transpose_t


# ===========================================================================
# TritonOrigamiMatmulSelector — Triton-gated specialization
# ===========================================================================
# GATING INVARIANT: This entire class is only instantiated when target_t ==
# triton in config_t.  No code in this class executes for hipblaslt /
# tensilelite workloads.  The existing OrigamiMatmulSelector class above
# remains the SOLE path for hipblaslt and is 100% untouched.
# ===========================================================================


class TritonOrigamiMatmulSelector:
    """
    Triton-specialized GEMM configuration selector.

    This class ports the tritonblas-specific specialization logic from the
    external tritonblas/origami.py into the core origami pipeline, gated
    exclusively to target_t == triton.  It provides:

      - Triton-specific LDS filtering (estimate_triton_lds_bytes with pipeline
        stages, swizzled layout — different from hipblaslt's raw tile check).
      - Self-generated config search space from per-arch MI inference and
        block_mn / block_k ranges.
      - Architecture-aware matrix instruction inference (gfx942, gfx950, gfx90a).
      - 256x256x64 near-square tile heuristic.
      - Work-stealing parameter selection (_select_ws_params).
      - Hierarchical split computation for multi-XCD GPUs.
      - Triton-specific StreamK grid model.
      - Cache hint variant generation (nontemporal A/B).
      - Graceful fallback when LDS filtering prunes all candidates.

    The existing OrigamiMatmulSelector (used by hipblaslt) is NOT modified in
    any way.  This class is a parallel, independent code path.
    """

    # Class-level wrapper for LDS estimation
    @staticmethod
    def estimate_triton_lds(
        block_m: int,
        block_n: int,
        block_k: int,
        bytes_a: float,
        bytes_b: float,
        num_stages: int = 2,
    ) -> float:
        """Class-level wrapper for estimate_triton_lds_bytes."""
        return estimate_triton_lds_bytes(
            block_m, block_n, block_k, bytes_a, bytes_b, num_stages
        )

    # dtype map — same as OrigamiMatmulSelector for API compatibility
    dtype_to_str = {
        torch.float32: "f32",
        torch.complex64: "c32",
        torch.complex128: "c64",
        torch.float64: "f64",
        torch.float16: "f16",
        torch.int32: "i32",
        torch.bfloat16: "bf16",
        torch.int8: "i8",
        torch.float8_e5m2: "f8",
        torch.float8_e4m3fn: "f8",
    }
    if hasattr(torch, "float8_e5m2fnuz"):
        dtype_to_str[torch.float8_e5m2fnuz] = "f8"
    if hasattr(torch, "float8_e4m3fnuz"):
        dtype_to_str[torch.float8_e4m3fnuz] = "f8"

    COUNTERS_PER_XCD = 4  # work-stealing: default, overridden by _select_ws_params()

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
        mx_block_size=0,
        streamk=False,
        total_cus: int = None,
        active_cus: int = None,
        num_stages: int = 2,
    ):
        """
        Initialize the Triton-specialized Origami matmul selector.

        This constructor generates its own config search space (no external
        config_gen iterable needed), applies Triton-specific LDS filtering,
        and runs the Origami analytical model to select the best config.

        All logic in this constructor is TRITON-SPECIFIC.  It must only be
        called when the caller intends to use target_t == triton.

        Args:
            m, n, k: GEMM dimensions.
            a_dtype, b_dtype, out_dtype: PyTorch dtypes.
            device: PyTorch CUDA device (must be ROCm-capable).
            mx_block_size: Block size for MX format dtypes (default: 0).
            streamk: Whether to use StreamK scheduling (default: False).
            total_cus: Override total CU count (e.g. for CU-mask sweeps).
            active_cus: Override active CU count (e.g. for CU-mask sweeps).
            num_stages: Pipeline stages for Triton LDS estimation (default: 2).

        Raises:
            RuntimeError: If no configs pass LDS checks.
            ValueError: If architecture or dtype is unsupported.
        """
        # --- TRITON GATE: set target to triton ---
        # Every config produced by this class has target_t == triton.

        # Save tensor sizes
        self._m = m
        self._n = n
        self._k = k
        self.streamk = streamk
        self._num_stages = num_stages

        # Save tensor dtypes as strings
        self._a_dtype_str = self.dtype_to_str.get(a_dtype, a_dtype)
        self._b_dtype_str = self.dtype_to_str.get(b_dtype, b_dtype)
        self._out_dtype_str = self.dtype_to_str.get(out_dtype, out_dtype)

        # Save MX block size
        self._mx_block_size = mx_block_size

        # Helper function to get bits for both float, int, and MX dtypes
        mx_types = ["f4"]

        def get_dtype_bits(dtype):
            if dtype in mx_types:
                return origami.datatype_to_bits(origami.string_to_datatype(dtype))
            try:
                return torch.finfo(dtype).bits
            except TypeError:
                return torch.iinfo(dtype).bits

        self._a_dtype_bitsize = get_dtype_bits(a_dtype)
        self._b_dtype_bitsize = get_dtype_bits(b_dtype)
        self._out_dtype_bitsize = get_dtype_bits(out_dtype)

        # MI dtype inference
        if a_dtype in mx_types:
            self.mi_dtype = a_dtype
        else:
            input_dtype_for_mi = (
                a_dtype
                if get_dtype_bits(a_dtype) <= get_dtype_bits(b_dtype)
                else b_dtype
            )
            self.mi_dtype = self.dtype_to_str.get(
                input_dtype_for_mi, self.dtype_to_str.get(out_dtype)
            )

        # Get hardware info from Origami
        self._hardware = origami.get_hardware_for_device(device.index)

        # Detect architecture name for MI instruction selection.
        if hasattr(self._hardware, 'arch') and hasattr(self._hardware.arch, 'name'):
            self._arch_name = self._hardware.arch.name
        else:
            _gcn = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
            self._arch_name = _gcn.split(":")[0] if _gcn else "unknown"

        # CU handling — override for CU-mask sweeps
        self._active_cus = active_cus
        if total_cus is not None:
            self._hardware.N_CU = total_cus
        self._N_CU = self._hardware.N_CU
        self._ACTIVE_CU = active_cus if active_cus is not None else self._N_CU

        # --- TRITON-SPECIFIC: Self-generate config search space ---
        self._block_mn_range = [16, 32, 64, 128, 256]
        self._block_k_range = [16, 32, 64, 128, 256, 512]
        self._kernel_occupancy_range = [1]
        self._configs = self._generate_default_configs()

        # Create Origami problem_t (needed for LDS check fallback)
        self._problem = self._make_problem()

        # --- TRITON-SPECIFIC: LDS filtering with fallback ---
        # Origami's check_lds_capacity uses raw tile size only; Triton allocates
        # num_stages buffers with swizzled layout — different formula.
        bytes_a = self._a_dtype_bitsize / 8
        bytes_b = self._b_dtype_bitsize / 8
        lds_cap = self._hardware.lds_capacity

        pre_filter_count = len(self._configs)
        self._configs = [
            c
            for c in self._configs
            if check_triton_lds_capacity(
                c.mt.m, c.mt.n, c.mt.k, bytes_a, bytes_b, lds_cap, self._num_stages
            )
        ]
        lds_pruned_count = pre_filter_count - len(self._configs)

        if not self._configs:
            # Fallback: origami's raw check (no Triton padding/stages) is more permissive.
            logger.warning(
                "[TRITON] All %d configs pruned by Triton LDS filter (cap=%d, stages=%d). "
                "Falling back to origami raw LDS check.",
                pre_filter_count, lds_cap, self._num_stages,
            )
            self._configs = self._generate_default_configs()
            self._configs = [
                c
                for c in self._configs
                if origami.check_lds_capacity(
                    self._hardware, c.mt, self._problem.a_dtype, self._problem.b_dtype
                )
            ]
        if not self._configs:
            # Should not happen on supported hardware (64KB+ LDS); small tiles always fit.
            raise RuntimeError(
                "[TRITON] No configs passed LDS checks; unexpected for supported hardware"
            )

        if lds_pruned_count > 0:
            logger.debug(
                "[TRITON] LDS filter pruned %d/%d configs (kept %d)",
                lds_pruned_count, pre_filter_count, len(self._configs),
            )

        # --- Run Origami solution selection (same analytical model, Triton configs) ---
        self._result = origami.select_config(
            self._problem, self._hardware, self._configs
        )

        # --- TRITON-SPECIFIC: 256x256x64 near-square heuristic ---
        if (check_triton_lds_capacity(256, 256, 64, bytes_a, bytes_b, lds_cap, self._num_stages) and
            ((self._result.config.mt.m == 256 and self._result.config.mt.n != 256) or
             (self._result.config.mt.m != 256 and self._result.config.mt.n == 256))):
            self._result.config.mt.m = 256
            self._result.config.mt.n = 256
            self._result.config.mt.k = 64

        # --- TRITON-SPECIFIC: Grid sizing ---
        if streamk:
            self._grid = self._compute_sk_grid()
        else:
            self._grid = self._hardware.N_CU

        # Handle different origami API versions for workgroup mapping
        _wg_result = origami.select_workgroup_mapping(
            self._problem, self._hardware, self._result.config, self._grid
        )
        if isinstance(_wg_result, tuple):
            if len(_wg_result) == 3:
                _, self._xcc_workgroup_mapping, self._workgroup_mapping = _wg_result
            else:
                self._xcc_workgroup_mapping, self._workgroup_mapping = _wg_result
        else:
            self._xcc_workgroup_mapping = _wg_result.wgmxcc
            self._workgroup_mapping = _wg_result.wgm

        # --- TRITON-SPECIFIC: Work-stealing parameters ---
        self._select_ws_params()

    # -----------------------------------------------------------------------
    # TRITON-SPECIFIC: Work-stealing parameter selection
    # -----------------------------------------------------------------------
    def _select_ws_params(self):
        """Select work-stealing parameters based on tile count.

        Empirically tuned on MI300X (8 XCDs, 304 CUs) via autotune sweeps
        across GEMM sizes 1K-16K.

        NOTE: TRITON-SPECIFIC — only called from TritonOrigamiMatmulSelector.
        """
        bm = self._result.config.mt.m
        bn = self._result.config.mt.n
        total_tiles = ((self._m + bm - 1) // bm) * ((self._n + bn - 1) // bn)
        tiles_m = (self._m + bm - 1) // bm

        if total_tiles <= 512:
            self.COUNTERS_PER_XCD = 8
        elif total_tiles <= 1536:
            self.COUNTERS_PER_XCD = 4
        elif total_tiles <= 2048:
            self.COUNTERS_PER_XCD = 2
        else:
            self.COUNTERS_PER_XCD = 1

        self._workgroup_mapping = min(8, tiles_m)

    # -----------------------------------------------------------------------
    # TRITON-SPECIFIC: Hierarchical split for multi-XCD work-stealing
    # -----------------------------------------------------------------------
    def hierarchical_split(self, num_xcds: int) -> tuple:
        """Compute optimal local/global tile split for hierarchical WS.

        Uses the full hardware CU count (not active CUs) so that the split
        is a topology-level constant, avoiding Triton recompilation when the
        active CU mask changes.

        Adaptive split based on tiles-per-CU density:
        - <=4 tiles/CU:  100% local (global counter overhead dominates)
        - >4 tiles/CU:  local_frac decreases linearly, floor at 50%

        Returns (local_per_xcd, global_tiles).

        NOTE: TRITON-SPECIFIC — only called from TritonOrigamiMatmulSelector.
        """
        bm = self._result.config.mt.m
        bn = self._result.config.mt.n
        total_tiles = ((self._m + bm - 1) // bm) * ((self._n + bn - 1) // bn)
        hw_cus = self._hardware.NUM_XCD * self._hardware.CU_per_L2
        tiles_per_cu = total_tiles / max(hw_cus, 1)

        local_frac = max(0.5, 1.0 - max(0.0, tiles_per_cu - 4.0) * 0.05)
        local_per_xcd = int(total_tiles * local_frac) // num_xcds
        local_per_xcd = max(local_per_xcd, 1)
        global_tiles = total_tiles - local_per_xcd * num_xcds
        return local_per_xcd, global_tiles

    # -----------------------------------------------------------------------
    # Properties — API-compatible with tritonblas origami.py
    # -----------------------------------------------------------------------
    @property
    def block_m(self):
        return self._result.config.mt.m

    @property
    def block_n(self):
        return self._result.config.mt.n

    @property
    def block_k(self):
        return self._result.config.mt.k

    @property
    def group_m(self):
        return self._workgroup_mapping

    @property
    def num_sms(self):
        return self._xcc_workgroup_mapping

    @property
    def num_stages(self):
        return self._num_stages

    @property
    def waves_per_eu(self):
        return self._result.config.occupancy

    @property
    def even_k(self):
        return self._k % self.block_k == 0

    @property
    def sk_grid(self):
        return self._grid

    # -----------------------------------------------------------------------
    # TRITON-SPECIFIC: StreamK grid model
    # -----------------------------------------------------------------------
    def _compute_sk_grid(self):
        """Triton-specific StreamK grid computation.

        NOTE: TRITON-SPECIFIC — different from hipblaslt StreamK.
        """
        split_factors = [8, 6, 4, 3, 2, 1]
        tile_fractions = [0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0]
        max_workspace = 128 * 1024 * 1024

        M, N, K = self._m, self._n, self._k
        BLK_M, BLK_N, BLK_K = self.block_m, self.block_n, self.block_k
        cu_count = self._hardware.N_CU

        tiles = ceil(M / BLK_M) * ceil(N / BLK_N)
        sk_grid = tiles
        iters_per_tile = max(1, ceil(K / BLK_K))

        if tiles > cu_count:
            virt_cu_count = cu_count
            min_even_tiles = tiles / virt_cu_count

            for frac in tile_fractions:
                frac_grid = int((tiles / (min_even_tiles + frac)) + 0.5)
                if (
                    tiles % frac_grid != 0
                    and self._partial_tile_size(frac_grid) > max_workspace
                ):
                    continue
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break

        elif tiles < cu_count:
            for factor in split_factors:
                split_grid = tiles * factor
                iters_per_cu = iters_per_tile // factor
                if split_grid <= cu_count and iters_per_cu >= 8:
                    sk_grid = split_grid
                    break

        if tiles % sk_grid != 0:
            sk_grid = tiles

        if tiles >= cu_count:
            last_wave_remainder = tiles % cu_count
            if (
                last_wave_remainder < 128
                and last_wave_remainder > 0
                and cu_count in [304, 80, 64]
            ):  # gfx942
                sk_grid = 256 if cu_count == 304 else 64
        return sk_grid

    def _partial_tile_size(self, sk_grid: int) -> int:
        """Compute workspace size for partial tiles (Triton StreamK).

        NOTE: TRITON-SPECIFIC.
        """
        BLK_M, BLK_N = self.block_m, self.block_n
        bytes_per_elem = self._out_dtype_bitsize // 8
        tile_size = BLK_M * BLK_N * bytes_per_elem
        return tile_size * sk_grid

    # -----------------------------------------------------------------------
    # TRITON-SPECIFIC: Self-generate config search space
    # -----------------------------------------------------------------------
    def _generate_default_configs(self):
        """Generate Triton config search space from per-arch MI inference.

        Creates the full Cartesian product of block_mn × block_mn × block_k ×
        occupancy × cache_hint_variants.  This is TRITON-SPECIFIC — the
        hipblaslt OrigamiMatmulSelector takes an external config_gen iterable
        instead.

        NOTE: TRITON-SPECIFIC — only called from TritonOrigamiMatmulSelector.
        """
        config_list = []
        mi = self._infer_matrix_instruction_dimensions()

        # Determine which cache_hints variants are needed based on shape
        a_bits = self._a_dtype_bitsize
        b_bits = self._b_dtype_bitsize
        M, N, K = self._m, self._n, self._k

        cache_hint_variants = [(0, 0)]  # default: no nontemporal hints

        k_mod = (K * a_bits) % 1024
        if k_mod == 0:
            for bk in self._block_k_range:
                mt_k_mod = (bk * a_bits) % 1024
                if mt_k_mod == 0:
                    cache_hint_variants.append((0, 4))  # nontemporal B
                    cache_hint_variants.append((4, 0))  # nontemporal A
                    break

        for blk_m, blk_n, blk_k, occupancy in itertools.product(
            self._block_mn_range,
            self._block_mn_range,
            self._block_k_range,
            self._kernel_occupancy_range,
        ):
            for hints_a, hints_b in cache_hint_variants:
                mt = origami.dim3_t(blk_m, blk_n, blk_k)

                new_config = origami.config_t()
                new_config.mt = mt
                new_config.mi = mi
                new_config.occupancy = occupancy
                # --- TRITON GATE: mark target as triton ---
                new_config.target = origami.target_t.triton
                if self.streamk:
                    new_config.grid_selection = origami.grid_selection_t.k_split_aware
                else:
                    new_config.grid_selection = origami.grid_selection_t.data_parallel
                # Set nontemporal cache hints for shapes that require them
                if hasattr(new_config, 'cache_hints_a'):
                    new_config.cache_hints_a = hints_a
                if hasattr(new_config, 'cache_hints_b'):
                    new_config.cache_hints_b = hints_b
                config_list.append(new_config)

        return config_list

    # -----------------------------------------------------------------------
    # TRITON-SPECIFIC: Problem creation (same structure, fixed layout)
    # -----------------------------------------------------------------------
    def _make_problem(self) -> origami.problem_t:
        """Create Origami problem_t for Triton GEMM.

        NOTE: TRITON-SPECIFIC — uses fixed T/N transpose (Triton convention),
        no stride-based inference.
        """
        size = origami.dim3_t(self._m, self._n, self._k)

        a_origami_dtype = origami.string_to_datatype(self._a_dtype_str)
        b_origami_dtype = origami.string_to_datatype(self._b_dtype_str)
        c_origami_dtype = origami.string_to_datatype(self._out_dtype_str)

        problem = origami.problem_t()
        problem.size = size
        problem.batch = 1
        problem.a_transpose = origami.transpose_t.T
        problem.b_transpose = origami.transpose_t.N
        problem.a_dtype = a_origami_dtype
        problem.b_dtype = b_origami_dtype
        problem.c_dtype = c_origami_dtype
        problem.d_dtype = c_origami_dtype
        problem.mi_dtype = c_origami_dtype
        problem.a_mx_block_size = self._mx_block_size
        problem.b_mx_block_size = self._mx_block_size

        return problem

    # -----------------------------------------------------------------------
    # TRITON-SPECIFIC: Architecture-aware matrix instruction inference
    # -----------------------------------------------------------------------
    def _infer_matrix_instruction_dimensions(self):
        """
        Infers matrix instruction dimensions based on hardware and dtype.

        This function handles the full arch × dtype matrix including gfx942,
        gfx950, gfx90a, and MI300A (228 CUs).  It also mutates the block_mn
        and block_k search ranges for sub-byte dtypes (FP8, FP4).

        NOTE: TRITON-SPECIFIC — hipblaslt uses
        hardware.get_recommended_matrix_instruction() instead.

        Raises:
            ValueError: If the architecture/dtype combination is unsupported.
        """
        largest_bitsize = max(self._a_dtype_bitsize, self._b_dtype_bitsize)

        mi_dim = None
        # gfx950
        if self._arch_name == "gfx950":
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 32)
            if largest_bitsize <= 8:
                if self._k % 256 == 0:
                    self._block_k_range = self._block_k_range + [256]
                else:
                    self._block_k_range = self._block_k_range + [128]
                self._block_mn_range = [32, 64, 128, 256]
                mi_dim = origami.dim3_t(16, 16, 128)
        # gfx942 (304 CUs full, 80 CUs partitioned, 64 CUs)
        if self._arch_name == "gfx942":
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            if largest_bitsize == 8:
                self._block_mn_range = self._block_mn_range + [512]
                self._block_k_range = self._block_k_range + [128, 256]
                mi_dim = origami.dim3_t(16, 16, 32)
            if largest_bitsize < 8:
                raise ValueError("gfx942 doesn't support F4/F6")
        if self._hardware.N_CU == 228:
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            if largest_bitsize == 8:
                self._block_mn_range = self._block_mn_range + [512]
                self._block_k_range = self._block_k_range + [128, 256]
                mi_dim = origami.dim3_t(16, 16, 32)
            if largest_bitsize < 8:
                raise ValueError("MI300A doesn't support F4/F6")
        # gfx90a
        if self._arch_name == "gfx90a":
            if largest_bitsize == 32:
                mi_dim = origami.dim3_t(16, 16, 4)
            if largest_bitsize == 16:
                mi_dim = origami.dim3_t(16, 16, 16)
            if largest_bitsize == 8:
                raise ValueError("MI200 doesn't support F8")
            if largest_bitsize < 8:
                raise ValueError("MI200 doesn't support F4/F6")
        if mi_dim is None:
            raise ValueError(
                f"No Valid Matrix Instruction for "
                f"{self._a_dtype_bitsize}-bit/{self._b_dtype_bitsize}-bit dtypes "
                f"on hardware with N_CU={self._hardware.N_CU}"
            )

        return mi_dim
