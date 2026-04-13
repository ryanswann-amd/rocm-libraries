# Test fixture: UNGATED Triton code (should FAIL)

import origami


# BUG: tritonblas logic in shared path, not guarded
def apply_tritonblas_lds_filter(config, hardware):
    triton_lds = config.mt.m * config.mt.k * 2
    if triton_lds > hardware.lds_capacity:
        return False
    return True


# This one IS properly gated — should NOT be flagged
def properly_gated(config):
    if config.target == origami.target_t.triton:
        triton_score = compute_triton_lds(config)
        return triton_score
    return default_score(config)
