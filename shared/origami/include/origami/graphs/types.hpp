/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/**
 * @file
 * @brief origami::graphs (kirigami) — foundational types.
 *
 * Kirigami expresses an irregular GPU operation as a DAG of *workgroup* nodes
 * whose edges are derived, not declared: a producer workgroup and a consumer
 * workgroup are connected exactly when the indices one writes intersect the
 * indices the other reads. Nothing here knows what a kernel computes — only
 * which parts of which buffers it touches, and in what order the operators run.
 *
 * This header holds the vocabulary the rest of the module shares:
 *   - `index_t`, the element index into a flattened allocation. Deliberately
 *     64-bit: a flattened index space is the *product* of an allocation's
 *     dimensions, so an 8192x8192 buffer already needs 27 bits and a batched
 *     one overflows 32 quickly. Workgroup and iteration ordinals stay `int`,
 *     since they are bounded by a launch grid.
 *   - `env_t`, the binding from problem-dimension name to concrete size. A
 *     graph is written symbolically (`ceil_div(M, BM)`) and resolved once, at
 *     graph construction, against one of these.
 *   - `role_t`, the read/write discriminator that drives the derivation. Only
 *     Write-then-Read pairs produce edges; see core.hpp for why the three other
 *     hazard classes do not.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace origami::graphs {

/**
 * @brief An element index into the flattened index space of an allocation.
 *
 * 64-bit because a flattened index space is the product of the allocation's
 * dimensions, which overflows 32 bits for realistic batched tensors even when
 * every individual dimension is small.
 */
using index_t = std::int64_t;

/**
 * @brief Bindings from problem-dimension name to concrete size.
 *
 * Operators declare workgroup counts and access patterns symbolically, in terms
 * of names like "M" or "N". An env_t supplies the values for one particular
 * problem, and is applied once when a graph is constructed.
 */
using env_t = std::unordered_map<std::string, index_t>;

// ─── role_t: what an access does ──────────────────────────────────────
/**
 * @brief Whether an access pattern reads or writes its allocation.
 *
 * The derivation only pairs a producer's writes against a consumer's reads.
 * Read-after-read is not a dependency at all, and both write-ordering hazards
 * (write-after-read, write-after-write) are ordering constraints on the
 * *operator* sequence rather than fine-grained dataflow, so neither produces a
 * workgroup edge. core.hpp documents the consequence.
 */
enum class role_t : std::uint8_t {
  read,
  write,
};

/** @brief Canonical string names for each role_t, indexed by enum value. */
inline constexpr std::array<std::string_view, 2> ROLE_NAMES = {
    "read",
    "write",
};

/**
 * @brief Look up the canonical name for an access role.
 *
 * @param r Access role.
 * @return std::string_view Canonical name.
 */
constexpr std::string_view role_name(role_t r) noexcept {
  return ROLE_NAMES[static_cast<std::size_t>(r)];
}

/**
 * @brief Parse a canonical role name into the enum.
 *
 * Used at the string edge (Python bindings, trace import); throws rather than
 * returning a default so a typo surfaces at the boundary instead of silently
 * becoming a read and dropping every edge that access should have produced.
 *
 * @param name Canonical role name.
 * @return role_t Matching access role.
 * @throws std::invalid_argument If the name is not a known role.
 */
inline role_t role_from_name(std::string_view name) {
  for (std::size_t i = 0; i < ROLE_NAMES.size(); ++i) {
    if (ROLE_NAMES[i] == name) return static_cast<role_t>(i);
  }
  throw std::invalid_argument(std::string{"unknown access role: "} + std::string{name});
}

}  // namespace origami::graphs
