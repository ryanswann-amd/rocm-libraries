// Test fixture: PROPERLY gated Triton code (should PASS)

#include "origami/types.hpp"

namespace origami {

// This function is entirely inside a triton guard — should pass
void process_triton_configs(const config_t& config) {
    if (config.target == target_t::triton) {
        // Triton-specific LDS filtering
        double triton_lds_usage = compute_triton_lds(config);
        if (triton_lds_usage > hardware.lds_capacity) {
            // tritonblas LDS constraint violated
            return;
        }
        apply_triton_scoring(config);
    }
    // Default path — no triton references here
}

double select_config_for_target(const config_t& config) {
    if (config.target == target_t::triton) {
        return triton_selection_math(config);
    }
    // tensilelite path — unchanged
    return tensilelite_selection_math(config);
}

}  // namespace origami
