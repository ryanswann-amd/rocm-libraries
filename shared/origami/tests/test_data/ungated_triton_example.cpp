// Test fixture: UNGATED Triton code (should FAIL)

#include "origami/types.hpp"

namespace origami {

// BUG: This triton-specific function is NOT inside a guard
// It would run for ALL targets including tensilelite
double compute_triton_lds_filter(const config_t& config) {
    // This references triton without a guard — VIOLATION
    double triton_lds = config.mt.m * config.mt.k * 2;
    return triton_lds;
}

// BUG: tritonblas logic in shared path
void apply_tritonblas_scoring(const config_t& config, double& score) {
    score *= 0.9;  // tritonblas penalty
}

// This one IS properly gated — should NOT be flagged
void properly_gated_function(const config_t& config) {
    if (config.target == target_t::triton) {
        double triton_score = triton_specific_math(config);
    }
}

}  // namespace origami
