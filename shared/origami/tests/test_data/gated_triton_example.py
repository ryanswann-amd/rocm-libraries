# Test fixture: PROPERLY gated Triton code (should PASS)

import origami


def select_config(config):
    if config.target == origami.target_t.triton:
        # Triton-specific LDS filtering
        triton_lds = compute_triton_lds(config)
        if triton_lds > hardware.lds_capacity:
            return None
        return apply_tritonblas_scoring(config)
    # Default tensilelite path — no triton references
    return default_selection(config)
