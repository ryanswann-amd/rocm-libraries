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
#include <utility>
#include <variant>

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
 *
 * Retained as the single-namespace spelling. `problem_t` and `config_t` below
 * are the two-namespace replacement; an env_t binds the problem namespace only.
 */
using env_t = std::unordered_map<std::string, index_t>;

// ─── scope_t: which namespace a symbol resolves against ───────────────
/**
 * @brief The namespace a symbolic name is looked up in.
 *
 * Keeping these apart is what stops a problem dimension and a tuning parameter
 * colliding on a shared name, and it is what lets one specification be
 * instantiated across a sweep: the problem is held fixed while the config
 * varies. `hardware` and `runtime` are addressable only from cost expressions,
 * since neither is known when an access pattern is resolved.
 */
enum class scope_t : std::uint8_t {
  problem,   ///< problem dimensions: M, N, K, element widths
  config,    ///< tuning parameters: BM, BN, BK, efficiencies
  hardware,  ///< machine description; cost expressions only
  runtime,   ///< live scheduler state; cost expressions only
};

/** @brief Canonical string names for each scope_t, indexed by enum value. */
inline constexpr std::array<std::string_view, 4> SCOPE_NAMES = {
    "problem",
    "config",
    "hardware",
    "runtime",
};

/**
 * @brief Look up the canonical name for a scope.
 *
 * @param s Scope.
 * @return std::string_view Canonical name.
 */
constexpr std::string_view scope_name(scope_t s) noexcept {
  return SCOPE_NAMES[static_cast<std::size_t>(s)];
}

// ─── parameter maps ───────────────────────────────────────────────────
/**
 * @brief One bound parameter: an exact integer, or a real.
 *
 * Both alternatives are needed because a config holds tile sizes and
 * efficiencies side by side. The distinction is load-bearing rather than
 * cosmetic: index arithmetic is exact 64-bit integer work, so a value used in
 * that position must not silently arrive as a float.
 */
using param_value_t = std::variant<index_t, double>;

/**
 * @brief A namespace of bound parameters, keyed by name.
 *
 * Not used directly; `problem_t` and `config_t` derive from it so the two
 * cannot be passed for one another.
 */
class param_map_t {
 public:
  using storage_t = std::unordered_map<std::string, param_value_t>;

  param_map_t() = default;

  /** @brief Bind a whole namespace at once. */
  explicit param_map_t(storage_t values) : values_(std::move(values)) {}

  /** @brief Bind or rebind one name. */
  void set(std::string name, param_value_t value) { values_[std::move(name)] = value; }

  /** @brief True when @p name is bound here. */
  bool contains(const std::string& name) const { return values_.count(name) != 0; }

  /** @brief Number of bound names. */
  std::size_t size() const { return values_.size(); }

  /** @brief True when nothing is bound. */
  bool empty() const { return values_.empty(); }

  /** @brief Every binding, for iteration and for the Python mapping protocol. */
  const storage_t& values() const { return values_; }

  /**
   * @brief Read a name that must be an exact integer.
   *
   * A real is accepted only when it is exactly integral, so `128.0` works and
   * `0.8` is rejected. The rejection is the point: an efficiency knob reaching
   * index arithmetic would silently truncate a tile bound.
   *
   * @param name Parameter name.
   * @param scope Scope this map represents, for the error message.
   * @return index_t The bound value.
   * @throws std::out_of_range If the name is unbound.
   * @throws std::invalid_argument If the value has a fractional part.
   */
  index_t index_at(const std::string& name, scope_t scope) const;

  /**
   * @brief Read a name as a real, accepting either alternative.
   *
   * @param name Parameter name.
   * @param scope Scope this map represents, for the error message.
   * @return double The bound value.
   * @throws std::out_of_range If the name is unbound.
   */
  double number_at(const std::string& name, scope_t scope) const;

 private:
  storage_t values_;
};

/** @brief Problem dimensions: the parameters every candidate shares. */
class problem_t : public param_map_t {
 public:
  using param_map_t::param_map_t;
};

/** @brief Tuning parameters: the parameters that differ between candidates. */
class config_t : public param_map_t {
 public:
  using param_map_t::param_map_t;
};

/**
 * @brief Everything a symbolic expression can be resolved against.
 *
 * Index expressions need the problem and the config. Cost expressions also
 * reach for hardware and live scheduler state, which is why those two scopes
 * exist on the cost side; see cost_expr.hpp.
 */
struct eval_context_t {
  problem_t problem;  ///< problem-scope bindings
  config_t config;    ///< config-scope bindings

  eval_context_t() = default;
  eval_context_t(problem_t p, config_t c) : problem(std::move(p)), config(std::move(c)) {}

  /**
   * @brief Bind the problem namespace from the single-namespace spelling.
   *
   * Implicit so that existing `env_t` call sites keep working. Note that it
   * copies, so library internals pass a context through rather than rebuilding
   * one per call.
   *
   * @param env Problem-scope bindings.
   */
  eval_context_t(const env_t& env);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Build a context binding the problem namespace only.
   *
   * The bridge from the single-namespace `env_t` spelling.
   *
   * @param env Problem-scope bindings.
   * @return eval_context_t Context with an empty config.
   */
  static eval_context_t from_env(const env_t& env);

  /**
   * @brief Resolve an integer-valued symbol.
   *
   * @param scope Namespace to look in.
   * @param name Symbol name.
   * @return index_t The bound value.
   * @throws std::out_of_range If unbound, or if the scope is not addressable.
   * @throws std::invalid_argument If the value is not integral.
   */
  index_t index_at(scope_t scope, const std::string& name) const;

  /** @brief Flatten back to the single-namespace spelling, config keys last. */
  env_t to_env() const;
};

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

// ─── parameter map and context, inline ────────────────────────────────

inline index_t param_map_t::index_at(const std::string& name, scope_t scope) const {
  const auto it = values_.find(name);
  if (it == values_.end()) {
    throw std::out_of_range(std::string{scope_name(scope)} + " parameter '" + name +
                            "' is unbound");
  }
  if (const auto* exact = std::get_if<index_t>(&it->second)) return *exact;

  const double real  = std::get<double>(it->second);
  const auto rounded = static_cast<index_t>(real);
  if (static_cast<double>(rounded) != real) {
    throw std::invalid_argument(std::string{scope_name(scope)} + " parameter '" + name +
                                "' is used where an exact integer is required, but is bound to " +
                                std::to_string(real));
  }
  return rounded;
}

inline double param_map_t::number_at(const std::string& name, scope_t scope) const {
  const auto it = values_.find(name);
  if (it == values_.end()) {
    throw std::out_of_range(std::string{scope_name(scope)} + " parameter '" + name +
                            "' is unbound");
  }
  if (const auto* exact = std::get_if<index_t>(&it->second)) { return static_cast<double>(*exact); }
  return std::get<double>(it->second);
}

inline eval_context_t eval_context_t::from_env(const env_t& env) {
  problem_t bound;
  for (const auto& [name, value] : env) bound.set(name, value);
  return eval_context_t{std::move(bound), config_t{}};
}

inline eval_context_t::eval_context_t(const env_t& env) : eval_context_t(from_env(env)) {}

inline index_t eval_context_t::index_at(scope_t scope, const std::string& name) const {
  switch (scope) {
    case scope_t::problem: return problem.index_at(name, scope);
    case scope_t::config: return config.index_at(name, scope);
    case scope_t::hardware:
    case scope_t::runtime: break;
  }
  throw std::out_of_range(std::string{"the "} + std::string{scope_name(scope)} +
                          " scope is not addressable from an index expression; it is available "
                          "only in cost expressions");
}

inline env_t eval_context_t::to_env() const {
  env_t out;
  for (const auto& [name, value] : problem.values()) {
    if (const auto* exact = std::get_if<index_t>(&value)) out[name] = *exact;
  }
  for (const auto& [name, value] : config.values()) {
    if (const auto* exact = std::get_if<index_t>(&value)) out[name] = *exact;
  }
  return out;
}

}  // namespace origami::graphs
