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
 * @brief origami::graphs — symbolic index-set algebra for access patterns.
 *
 * An access pattern maps a workgroup id to a *set of indices* into a flattened
 * allocation. The entire library rests on one question, asked once per
 * producer-consumer workgroup pair: do two such sets intersect, and in how many
 * elements? That question is the innermost loop of the derivation, so the
 * representation is chosen to answer it without materialising anything.
 *
 * Two index-set representations, held in a variant rather than behind a virtual
 * base — derivation is a full quadratic over workgroup-iteration pairs with no
 * pruning, so an allocation and an indirect call per intersection would dominate:
 *
 *   - `range_t`, an arithmetic progression `start, start+step, ...` of `count`
 *     elements. Two progressions intersect in O(1) by the Chinese Remainder
 *     Theorem, no matter how many elements they contain. This covers strided and
 *     affine tile access, which is essentially every GEMM and collective kernel.
 *   - `finite_set_t`, an explicit sorted index list. The escape hatch for
 *     arbitrary gather/scatter; intersecting it materialises.
 *
 * Above the sets sits `scalar_expr_t`, a small interpreter over an expression
 * tree. A workgroup count is rarely a literal: it is `ceil_div(M, BM) *
 * ceil_div(N, BN)` for problem dimensions known only at launch. Expressions are
 * built from `sym()` and plain integers, then evaluated against an `env_t` once
 * the enclosing graph binds concrete dimensions.
 *
 * Integer division follows Python's semantics (floor, not truncation) because
 * this is a port and schedule identity is the acceptance gate; see `floor_div`.
 */
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "origami/graphs/types.hpp"

namespace origami::graphs {

// ─── scalar_expr_t: symbolic non-negative integers ────────────────────

/** @brief Operator carried by an interior `scalar_expr_t` node. */
enum class expr_kind_t : std::uint8_t {
  constant,  ///< literal integer leaf
  symbol,    ///< named problem dimension leaf
  add,       ///< a + b
  sub,       ///< a - b
  mul,       ///< a * b
  floordiv,  ///< floor(a / b)
  ceildiv,   ///< ceil(a / b) — the tile-count operator
  mod,       ///< a mod b, Euclidean — the tile-decode operator
  minimum,   ///< min(a, b) — clipping a tile to a boundary
  maximum,   ///< max(a, b)
};

/**
 * @brief A free symbol together with the namespace it resolves against.
 *
 * Ordered by scope then name so diagnostics and test expectations are stable.
 */
struct scoped_symbol_t {
  scope_t scope = scope_t::problem;  ///< namespace the name resolves against
  std::string name;                  ///< symbol name

  /** @brief "problem.M" style rendering, for messages. */
  std::string str() const { return std::string{scope_name(scope)} + "." + name; }

  friend bool operator<(const scoped_symbol_t& a, const scoped_symbol_t& b) {
    if (a.scope != b.scope) return a.scope < b.scope;
    return a.name < b.name;
  }

  friend bool operator==(const scoped_symbol_t& a, const scoped_symbol_t& b) {
    return a.scope == b.scope && a.name == b.name;
  }
};

/**
 * @brief A symbolic non-negative integer: a workgroup count or a dimension.
 *
 * Value type wrapping a shared, immutable expression tree, so copies are a
 * refcount bump and expressions can be shared freely between operations. The
 * converting constructor from `index_t` is deliberately implicit, which is what
 * lets a builder take a literal wherever it takes an expression: both
 * `ceil_div(M, 128)` and `ceil_div(M, BM)` compile, and no entry point below
 * needs a second overload to accept the constant form.
 */
class scalar_expr_t {
 public:
  /** @brief Construct the literal zero. */
  scalar_expr_t();

  /**
   * @brief Wrap a literal integer as an expression.
   *
   * Implicit by design; see the class note.
   *
   * @param value Literal value.
   */
  scalar_expr_t(index_t value);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Evaluate against concrete dimension bindings.
   *
   * Binds the problem namespace only; a config-scope symbol left unresolved by
   * this overload throws. Prefer the `eval_context_t` overload.
   *
   * @param env Bindings for every free symbol in this expression.
   * @return index_t The evaluated value.
   * @throws std::out_of_range If a referenced dimension is unbound.
   * @throws std::domain_error On division by zero.
   */
  index_t eval(const env_t& env) const;

  /**
   * @brief Evaluate against both namespaces.
   *
   * @param ctx Problem and config bindings.
   * @return index_t The evaluated value.
   * @throws std::out_of_range If a referenced symbol is unbound.
   * @throws std::invalid_argument If a symbol is bound to a non-integral value.
   * @throws std::domain_error On division by zero.
   */
  index_t eval(const eval_context_t& ctx) const;

  /**
   * @brief Evaluate an expression that must already be concrete.
   *
   * @return index_t The evaluated value.
   * @throws std::out_of_range If the expression still has free symbols.
   */
  index_t eval() const;

  /**
   * @brief Names of the unbound dimensions this expression references.
   *
   * Ordered so error messages and test expectations are deterministic. Names
   * are bare, so a problem dimension and a config parameter sharing a name
   * appear once; `scoped_symbols` is the precise form.
   *
   * @return std::set<std::string> Free dimension names.
   */
  std::set<std::string> free_symbols() const;

  /**
   * @brief Free symbols with the namespace each resolves against.
   *
   * @return std::set<scoped_symbol_t> Free symbols, ordered by scope then name.
   */
  std::set<scoped_symbol_t> scoped_symbols() const;

  /** @brief True when this expression references no dimensions. */
  bool is_constant() const;

  /**
   * @brief Render the expression tree for diagnostics and graph summaries.
   *
   * @return std::string Infix rendering, e.g. "(M ceildiv 128)".
   */
  std::string str() const;

  /** @brief Structural node kind, exposed for testing and trace emission. */
  expr_kind_t kind() const;

 private:
  struct node_t;
  explicit scalar_expr_t(std::shared_ptr<const node_t> node);

  friend scalar_expr_t scoped_sym(scope_t scope, std::string name);
  friend scalar_expr_t make_binary(expr_kind_t kind,
                                   const scalar_expr_t& lhs,
                                   const scalar_expr_t& rhs);

  std::shared_ptr<const node_t> node_;
};

/**
 * @brief Reference a name in a chosen namespace.
 *
 * Index expressions may only reach the problem and config scopes; asking for
 * hardware or runtime here throws, because neither is known at the point an
 * access pattern is resolved.
 *
 * @param scope Namespace the name resolves against.
 * @param name Symbol name.
 * @return scalar_expr_t A symbol leaf.
 * @throws std::invalid_argument If the scope is not addressable from an index.
 */
scalar_expr_t scoped_sym(scope_t scope, std::string name);

/**
 * @brief Reference a problem dimension, e.g. `problem_sym("M")`.
 *
 * @param name Dimension name.
 * @return scalar_expr_t A problem-scope symbol leaf.
 */
scalar_expr_t problem_sym(std::string name);

/**
 * @brief Reference a tuning parameter, e.g. `config_sym("BM")`.
 *
 * @param name Parameter name.
 * @return scalar_expr_t A config-scope symbol leaf.
 */
scalar_expr_t config_sym(std::string name);

/**
 * @brief Reference a named problem dimension, e.g. `sym("M")`.
 *
 * The single-namespace spelling, equivalent to `problem_sym`.
 *
 * @param name Dimension name.
 * @return scalar_expr_t A problem-scope symbol leaf.
 */
scalar_expr_t sym(std::string name);

/**
 * @brief Build an interior expression node.
 *
 * Exposed because the arithmetic operators and `ceil_div` are free functions;
 * callers should prefer those.
 *
 * @param kind Operator to apply.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return scalar_expr_t The combined expression.
 */
scalar_expr_t make_binary(expr_kind_t kind, const scalar_expr_t& lhs, const scalar_expr_t& rhs);

scalar_expr_t operator+(const scalar_expr_t& lhs, const scalar_expr_t& rhs);
scalar_expr_t operator-(const scalar_expr_t& lhs, const scalar_expr_t& rhs);
scalar_expr_t operator*(const scalar_expr_t& lhs, const scalar_expr_t& rhs);
scalar_expr_t operator/(const scalar_expr_t& lhs, const scalar_expr_t& rhs);

/**
 * @brief Symbolic `ceil(a / b)` — the standard tile-count expression.
 *
 * @param a Numerator.
 * @param b Denominator.
 * @return scalar_expr_t The ceiling-division expression.
 */
scalar_expr_t ceil_div(const scalar_expr_t& a, const scalar_expr_t& b);

/**
 * @brief Symbolic `floor(a / b)`, the explicit spelling of `operator/`.
 *
 * @param a Numerator.
 * @param b Denominator.
 * @return scalar_expr_t The floor-division expression.
 */
scalar_expr_t floor_div(const scalar_expr_t& a, const scalar_expr_t& b);

/**
 * @brief Symbolic Euclidean modulo — the other half of a tile decode.
 *
 * @param a Value.
 * @param b Modulus.
 * @return scalar_expr_t The modulo expression.
 */
scalar_expr_t mod(const scalar_expr_t& a, const scalar_expr_t& b);

/**
 * @brief Symbolic `min(a, b)`, which is how a boundary tile is clipped.
 *
 * Kept on the integer expression rather than only on the cost expression so
 * that clipping a tile against a problem dimension stays exact.
 *
 * @param a First operand.
 * @param b Second operand.
 * @return scalar_expr_t The minimum expression.
 */
scalar_expr_t minimum(const scalar_expr_t& a, const scalar_expr_t& b);

/**
 * @brief Symbolic `max(a, b)`.
 *
 * @param a First operand.
 * @param b Second operand.
 * @return scalar_expr_t The maximum expression.
 */
scalar_expr_t maximum(const scalar_expr_t& a, const scalar_expr_t& b);

/**
 * @brief Workgroup count for a 2D tiled operator — the GEMM grid.
 *
 * `ceil_div(rows_dim, block_rows) * ceil_div(cols_dim, block_cols)`.
 *
 * @param rows_dim Total rows.
 * @param cols_dim Total columns.
 * @param block_rows Rows per tile.
 * @param block_cols Columns per tile.
 * @return scalar_expr_t The tile count.
 */
scalar_expr_t grid_2d(const scalar_expr_t& rows_dim,
                      const scalar_expr_t& cols_dim,
                      const scalar_expr_t& block_rows,
                      const scalar_expr_t& block_cols);

// ─── integer division with Python semantics ───────────────────────────

/**
 * @brief Floor division, rounding toward negative infinity.
 *
 * C++ `/` truncates toward zero, Python's `//` floors. They agree on
 * non-negative operands and disagree on negative ones, and the CRT intersection
 * below genuinely divides negative differences — so a truncating port silently
 * produces off-by-one starts on some progression pairs. Matching Python here is
 * what makes schedule identity achievable.
 *
 * @param a Numerator.
 * @param b Denominator.
 * @return index_t floor(a / b).
 * @throws std::domain_error If b is zero.
 */
index_t floor_div(index_t a, index_t b);

/**
 * @brief Ceiling division, correct for negative numerators.
 *
 * @param a Numerator.
 * @param b Denominator.
 * @return index_t ceil(a / b).
 * @throws std::domain_error If b is zero.
 */
index_t ceil_div_int(index_t a, index_t b);

/**
 * @brief Euclidean modulo, always returning a value with the sign of @p m.
 *
 * Python's `%` follows the divisor's sign; C++'s follows the dividend's. The
 * CRT solve depends on the Python behaviour.
 *
 * @param a Value.
 * @param m Modulus.
 * @return index_t a mod m, non-negative for positive m.
 * @throws std::domain_error If m is zero.
 */
index_t euclid_mod(index_t a, index_t m);

// ─── index sets ───────────────────────────────────────────────────────

/**
 * @brief An arithmetic progression: `start, start+step, ...` with `count` terms.
 *
 * Invariants: `step > 0` and `count >= 0`. An empty progression has
 * `count == 0`, and is the canonical empty index set.
 */
struct range_t {
  index_t start = 0;  ///< first element
  index_t step  = 1;  ///< common difference, strictly positive
  index_t count = 0;  ///< number of elements, zero for the empty set

  /** @brief Construct the empty progression. */
  range_t() = default;

  /**
   * @brief Construct and check the progression invariants.
   *
   * @param start First element.
   * @param step Common difference; must be positive.
   * @param count Element count; must be non-negative.
   * @throws std::invalid_argument If step is non-positive or count is negative.
   */
  range_t(index_t start, index_t step, index_t count);

  /**
   * @brief The last element of the progression.
   *
   * @return index_t The final element.
   * @throws std::logic_error If the progression is empty.
   */
  index_t last() const;

  /** @brief Number of elements. */
  index_t size() const { return count; }

  /** @brief True when the progression contains no elements. */
  bool empty() const { return count == 0; }
};

/**
 * @brief An explicit index set — the arbitrary-access escape hatch.
 *
 * Held as a sorted, deduplicated vector rather than a hash set: intersection
 * becomes a linear merge instead of a probe per element, iteration order is
 * deterministic (which matters for reproducible edge ordering), and the storage
 * is contiguous.
 */
class finite_set_t {
 public:
  /** @brief Construct the empty set. */
  finite_set_t() = default;

  /**
   * @brief Build from an arbitrary collection, sorting and deduplicating.
   *
   * @param values Indices, in any order, possibly with duplicates.
   * @return finite_set_t The normalised set.
   */
  static finite_set_t of(std::vector<index_t> values);

  /** @brief The sorted, deduplicated indices. */
  const std::vector<index_t>& indices() const { return indices_; }

  /** @brief Number of indices. */
  index_t size() const { return static_cast<index_t>(indices_.size()); }

  /** @brief True when the set contains no indices. */
  bool empty() const { return indices_.empty(); }

 private:
  std::vector<index_t> indices_;
};

/**
 * @brief A set of allocation indices: either a progression or an explicit list.
 *
 * A variant rather than a class hierarchy — see the file note on why the hot
 * loop cannot afford an indirection.
 */
using index_set_t = std::variant<range_t, finite_set_t>;

/** @brief The canonical empty index set. */
inline index_set_t empty_index_set() { return index_set_t{range_t{}}; }

/**
 * @brief Number of indices in a set.
 *
 * @param s Index set.
 * @return index_t Element count.
 */
index_t size(const index_set_t& s);

/**
 * @brief True when a set contains no indices.
 *
 * @param s Index set.
 * @return bool Whether the set is empty.
 */
bool is_empty(const index_set_t& s);

/**
 * @brief Materialise a set to a sorted vector of indices.
 *
 * Only for diagnostics and the fuzz oracle; the derivation never calls it.
 *
 * @param s Index set.
 * @return std::vector<index_t> Sorted indices.
 */
std::vector<index_t> to_vector(const index_set_t& s);

/**
 * @brief Intersect two index sets.
 *
 * O(1) when both are progressions, by CRT. Any other combination materialises
 * the smaller side and filters.
 *
 * @param a First set.
 * @param b Second set.
 * @return index_set_t The intersection.
 */
index_set_t intersect(const index_set_t& a, const index_set_t& b);

/**
 * @brief Cardinality of an intersection, without building the result.
 *
 * This is what the derivation actually needs — an edge's weight is `|W n R|`
 * and the intersection itself is discarded. Skipping construction avoids an
 * allocation per candidate pair in the quadratic loop.
 *
 * @param a First set.
 * @param b Second set.
 * @return index_t Number of shared indices.
 */
index_t intersect_size(const index_set_t& a, const index_set_t& b);

// ─── pattern builders ─────────────────────────────────────────────────
//
// An access pattern's index map is canonically `(wg, it, env) -> index_set_t`.
// `it` is the internal iteration of a software-pipelined kernel, and is what
// makes *partial* dependencies expressible: a consumer iteration can depend on
// only the producer iterations that wrote what it reads. Builders that describe
// monolithic access simply ignore it.

/** @brief Canonical index map: workgroup and iteration to the indices touched. */
using index_fn_t = std::function<index_set_t(int wg, int it, const eval_context_t& ctx)>;

/**
 * @brief Each workgroup owns a contiguous chunk of @p block indices.
 *
 * `wg -> [base + wg*block, base + (wg+1)*block)`.
 *
 * @param block Indices per workgroup.
 * @param base Offset of the first chunk.
 * @return index_fn_t The index map.
 */
index_fn_t contiguous(scalar_expr_t block, scalar_expr_t base = 0);

/**
 * @brief Each workgroup owns @p block indices starting at `base + wg*stride`.
 *
 * Differs from `contiguous` only when stride and block disagree, i.e. when
 * chunks overlap or leave gaps.
 *
 * @param block Indices per workgroup.
 * @param stride Distance between consecutive workgroups' start offsets.
 * @param base Offset of the first chunk.
 * @return index_fn_t The index map.
 */
index_fn_t strided(scalar_expr_t block, scalar_expr_t stride, scalar_expr_t base = 0);

/**
 * @brief A row band of a row-major 2D allocation, flattened.
 *
 * Workgroup `w` owns rows `[w*block_rows, (w+1)*block_rows)`, which flattens to
 * `[base + w*block_rows*row_len, ...)`. `row_len` is typically a symbolic column
 * dimension of the operator's runtime argument.
 *
 * @param block_rows Rows per workgroup.
 * @param row_len Elements per row.
 * @param base Offset of the first band.
 * @return index_fn_t The index map.
 */
index_fn_t rows(scalar_expr_t block_rows, scalar_expr_t row_len, scalar_expr_t base = 0);

/**
 * @brief Iteration-aware access: a block streamed over @p num_iters iterations.
 *
 * Workgroup `w` owns a `block` chunk; iteration `it` touches the `it`-th
 * contiguous sub-chunk of it, of size `ceil(block / num_iters)`, with the last
 * iteration clipped to the block end. This is the unit of a partial dependency.
 *
 * @param block Indices per workgroup, across all iterations.
 * @param num_iters Iterations the block is streamed over; clamped to at least 1.
 * @param base Offset of the first chunk.
 * @return index_fn_t The index map.
 */
index_fn_t streamed(scalar_expr_t block, scalar_expr_t num_iters, scalar_expr_t base = 0);

/**
 * @brief Wrap an arbitrary index-producing callable as an index map.
 *
 * The escape hatch for gather/scatter with no closed form, and the fallback
 * rather than the default. It is invoked once per workgroup-iteration pair
 * inside the quadratic derivation and its result is materialised, whereas the
 * closed-form builders above intersect as arithmetic progressions in constant
 * time. Reach for one of those where the access has a closed form.
 *
 * @param fn Callable returning the indices touched by one workgroup-iteration.
 * @return index_fn_t The index map.
 */
index_fn_t index_offsets(
    std::function<std::vector<index_t>(int wg, int it, const eval_context_t& ctx)> fn);

}  // namespace origami::graphs
