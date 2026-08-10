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

#include "origami/graphs/symbolic.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace origami::graphs {

// ─── integer division with Python semantics ───────────────────────────

index_t floor_div(index_t a, index_t b) {
  if (b == 0) throw std::domain_error("floor_div: division by zero");
  index_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

index_t ceil_div_int(index_t a, index_t b) {
  if (b == 0) throw std::domain_error("ceil_div_int: division by zero");
  return -floor_div(-a, b);
}

index_t euclid_mod(index_t a, index_t m) {
  if (m == 0) throw std::domain_error("euclid_mod: division by zero");
  index_t r = a % m;
  if (r != 0 && ((r < 0) != (m < 0))) r += m;
  return r;
}

namespace {

/** Extended Euclid, mirroring the reference implementation exactly. */
struct egcd_result_t {
  index_t g = 0;
  index_t x = 0;
  index_t y = 0;
};

egcd_result_t egcd(index_t a, index_t b) {
  if (b == 0) return {a, 1, 0};
  const egcd_result_t r = egcd(b, euclid_mod(a, b));
  return {r.g, r.y, r.x - floor_div(a, b) * r.y};
}

/**
 * (a * b) mod m without overflowing the intermediate product.
 *
 * The CRT solve multiplies a progression offset by a modular inverse, and both
 * factors can approach the modulus, so the product overflows int64 for large
 * strides even though the result never does. `__int128` would be the easy fix
 * but is a compiler extension, and origami is strict ISO C++17, so the wide
 * case falls back to doubling-and-adding in unsigned arithmetic where every
 * intermediate provably stays below 2^64.
 */
index_t mul_mod(index_t a, index_t b, index_t m) {
  a = euclid_mod(a, m);
  b = euclid_mod(b, m);

  // Below this bound the direct product cannot exceed the signed 64-bit range.
  constexpr index_t kDirectLimit = 3037000499LL;  // floor(sqrt(2^63 - 1))
  if (m <= kDirectLimit) return (a * b) % m;

  auto um           = static_cast<std::uint64_t>(m);
  auto ua           = static_cast<std::uint64_t>(a);
  auto ub           = static_cast<std::uint64_t>(b);
  std::uint64_t acc = 0;
  while (ub > 0) {
    if ((ub & 1U) != 0U) acc = (acc + ua) % um;
    ua = (ua + ua) % um;
    ub >>= 1;
  }
  return static_cast<index_t>(acc);
}

}  // namespace

// ─── scalar_expr_t ────────────────────────────────────────────────────

struct scalar_expr_t::node_t {
  expr_kind_t kind = expr_kind_t::constant;
  index_t value    = 0;
  std::string name;
  std::shared_ptr<const node_t> left;
  std::shared_ptr<const node_t> right;

  index_t eval(const env_t& env) const {
    switch (kind) {
      case expr_kind_t::constant: return value;
      case expr_kind_t::symbol: {
        const auto it = env.find(name);
        if (it == env.end()) {
          throw std::out_of_range("dimension '" + name +
                                  "' is unbound; bind it in the graph's env_t");
        }
        return it->second;
      }
      default: break;
    }
    const index_t a = left->eval(env);
    const index_t b = right->eval(env);
    switch (kind) {
      case expr_kind_t::add: return a + b;
      case expr_kind_t::sub: return a - b;
      case expr_kind_t::mul: return a * b;
      case expr_kind_t::floordiv: return floor_div(a, b);
      case expr_kind_t::ceildiv: return ceil_div_int(a, b);
      default: throw std::logic_error("scalar_expr_t: unhandled expression kind");
    }
  }

  void collect_symbols(std::set<std::string>& out) const {
    if (kind == expr_kind_t::symbol) {
      out.insert(name);
      return;
    }
    if (left) left->collect_symbols(out);
    if (right) right->collect_symbols(out);
  }

  std::string str() const {
    switch (kind) {
      case expr_kind_t::constant: return std::to_string(value);
      case expr_kind_t::symbol: return name;
      case expr_kind_t::add: return "(" + left->str() + " + " + right->str() + ")";
      case expr_kind_t::sub: return "(" + left->str() + " - " + right->str() + ")";
      case expr_kind_t::mul: return "(" + left->str() + " * " + right->str() + ")";
      case expr_kind_t::floordiv: return "(" + left->str() + " // " + right->str() + ")";
      case expr_kind_t::ceildiv: return "ceil_div(" + left->str() + ", " + right->str() + ")";
    }
    throw std::logic_error("scalar_expr_t: unhandled expression kind");
  }
};

scalar_expr_t::scalar_expr_t() : scalar_expr_t(index_t{0}) {}

scalar_expr_t::scalar_expr_t(index_t value) {
  auto n   = std::make_shared<node_t>();
  n->kind  = expr_kind_t::constant;
  n->value = value;
  node_    = std::move(n);
}

scalar_expr_t::scalar_expr_t(std::shared_ptr<const node_t> node) : node_(std::move(node)) {}

index_t scalar_expr_t::eval(const env_t& env) const { return node_->eval(env); }

index_t scalar_expr_t::eval() const {
  static const env_t kEmpty;
  return node_->eval(kEmpty);
}

std::set<std::string> scalar_expr_t::free_symbols() const {
  std::set<std::string> out;
  node_->collect_symbols(out);
  return out;
}

bool scalar_expr_t::is_constant() const { return free_symbols().empty(); }

std::string scalar_expr_t::str() const { return node_->str(); }

expr_kind_t scalar_expr_t::kind() const { return node_->kind; }

scalar_expr_t sym(std::string name) {
  auto n  = std::make_shared<scalar_expr_t::node_t>();
  n->kind = expr_kind_t::symbol;
  n->name = std::move(name);
  return scalar_expr_t{std::move(n)};
}

scalar_expr_t make_binary(expr_kind_t kind, const scalar_expr_t& lhs, const scalar_expr_t& rhs) {
  auto n   = std::make_shared<scalar_expr_t::node_t>();
  n->kind  = kind;
  n->left  = lhs.node_;
  n->right = rhs.node_;
  return scalar_expr_t{std::move(n)};
}

scalar_expr_t operator+(const scalar_expr_t& lhs, const scalar_expr_t& rhs) {
  return make_binary(expr_kind_t::add, lhs, rhs);
}

scalar_expr_t operator-(const scalar_expr_t& lhs, const scalar_expr_t& rhs) {
  return make_binary(expr_kind_t::sub, lhs, rhs);
}

scalar_expr_t operator*(const scalar_expr_t& lhs, const scalar_expr_t& rhs) {
  return make_binary(expr_kind_t::mul, lhs, rhs);
}

scalar_expr_t operator/(const scalar_expr_t& lhs, const scalar_expr_t& rhs) {
  return make_binary(expr_kind_t::floordiv, lhs, rhs);
}

scalar_expr_t ceil_div(const scalar_expr_t& a, const scalar_expr_t& b) {
  return make_binary(expr_kind_t::ceildiv, a, b);
}

scalar_expr_t grid_2d(const scalar_expr_t& rows_dim,
                      const scalar_expr_t& cols_dim,
                      const scalar_expr_t& block_rows,
                      const scalar_expr_t& block_cols) {
  return ceil_div(rows_dim, block_rows) * ceil_div(cols_dim, block_cols);
}

// ─── range_t / finite_set_t ───────────────────────────────────────────

range_t::range_t(index_t start, index_t step, index_t count)
    : start(start), step(step), count(count) {
  if (step <= 0) throw std::invalid_argument("range_t: step must be positive");
  if (count < 0) throw std::invalid_argument("range_t: count must be non-negative");
}

index_t range_t::last() const {
  if (count == 0) throw std::logic_error("range_t::last on an empty range");
  return start + step * (count - 1);
}

finite_set_t finite_set_t::of(std::vector<index_t> values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  finite_set_t out;
  out.indices_ = std::move(values);
  return out;
}

// ─── index-set queries ────────────────────────────────────────────────

index_t size(const index_set_t& s) {
  return std::visit([](const auto& v) { return v.size(); }, s);
}

bool is_empty(const index_set_t& s) {
  return std::visit([](const auto& v) { return v.empty(); }, s);
}

std::vector<index_t> to_vector(const index_set_t& s) {
  if (const auto* r = std::get_if<range_t>(&s)) {
    std::vector<index_t> out;
    out.reserve(static_cast<std::size_t>(r->count));
    for (index_t i = 0; i < r->count; ++i) out.push_back(r->start + r->step * i);
    return out;
  }
  return std::get<finite_set_t>(s).indices();
}

namespace {

/** True when @p v lies on the progression @p r. */
bool range_contains(const range_t& r, index_t v) {
  if (r.count == 0) return false;
  if (v < r.start || v > r.last()) return false;
  return euclid_mod(v - r.start, r.step) == 0;
}

/**
 * O(1) intersection of two arithmetic progressions.
 *
 * The overlap of two progressions is itself a progression with step lcm(da, db),
 * existing only when the starts are congruent modulo gcd(da, db). CRT gives the
 * first aligned value; the rest is clamping that to the overlapping window.
 */
index_set_t intersect_ranges(const range_t& a, const range_t& b) {
  if (a.count == 0 || b.count == 0) return empty_index_set();

  const index_t lo = std::max(a.start, b.start);
  const index_t hi = std::min(a.last(), b.last());
  if (lo > hi) return empty_index_set();

  const index_t da = a.step;
  const index_t db = b.step;
  const index_t g  = std::gcd(da, db);
  if (euclid_mod(b.start - a.start, g) != 0) return empty_index_set();  // never align

  const index_t lcm  = da / g * db;
  const index_t diff = floor_div(b.start - a.start, g);
  const index_t m    = db / g;
  // (da/g) * p == 1 (mod db/g), so x0 satisfies both congruences.
  const index_t p  = egcd(da / g, m).x;
  const index_t x0 = a.start + da * mul_mod(diff, p, m);

  const index_t start = x0 + ceil_div_int(lo - x0, lcm) * lcm;
  if (start > hi) return empty_index_set();
  return index_set_t{range_t{start, lcm, floor_div(hi - start, lcm) + 1}};
}

/** Intersect an explicit set against a progression, preserving sorted order. */
finite_set_t intersect_finite_range(const finite_set_t& f, const range_t& r) {
  std::vector<index_t> out;
  for (index_t v : f.indices()) {
    if (range_contains(r, v)) out.push_back(v);
  }
  return finite_set_t::of(std::move(out));
}

}  // namespace

index_set_t intersect(const index_set_t& a, const index_set_t& b) {
  const auto* ra = std::get_if<range_t>(&a);
  const auto* rb = std::get_if<range_t>(&b);
  if (ra && rb) return intersect_ranges(*ra, *rb);
  if (ra) return index_set_t{intersect_finite_range(std::get<finite_set_t>(b), *ra)};
  if (rb) return index_set_t{intersect_finite_range(std::get<finite_set_t>(a), *rb)};

  const auto& fa = std::get<finite_set_t>(a).indices();
  const auto& fb = std::get<finite_set_t>(b).indices();
  std::vector<index_t> out;
  std::set_intersection(fa.begin(), fa.end(), fb.begin(), fb.end(), std::back_inserter(out));
  return index_set_t{finite_set_t::of(std::move(out))};
}

index_t intersect_size(const index_set_t& a, const index_set_t& b) {
  const auto* ra = std::get_if<range_t>(&a);
  const auto* rb = std::get_if<range_t>(&b);
  // Both progressions: CRT allocates nothing, so just build and measure.
  if (ra && rb) return size(intersect_ranges(*ra, *rb));

  // Mixed and explicit cases can be counted without materialising a result.
  if (ra || rb) {
    const range_t& r      = ra ? *ra : *rb;
    const finite_set_t& f = std::get<finite_set_t>(ra ? b : a);
    index_t n             = 0;
    for (index_t v : f.indices()) {
      if (range_contains(r, v)) ++n;
    }
    return n;
  }

  const auto& fa = std::get<finite_set_t>(a).indices();
  const auto& fb = std::get<finite_set_t>(b).indices();
  index_t n      = 0;
  std::size_t i = 0, j = 0;
  while (i < fa.size() && j < fb.size()) {
    if (fa[i] < fb[j]) {
      ++i;
    } else if (fb[j] < fa[i]) {
      ++j;
    } else {
      ++n;
      ++i;
      ++j;
    }
  }
  return n;
}

// ─── pattern builders ─────────────────────────────────────────────────

index_fn_t contiguous(scalar_expr_t block, scalar_expr_t base) {
  return [block, base](int wg, int, const env_t& env) -> index_set_t {
    const index_t b = block.eval(env);
    return index_set_t{range_t{base.eval(env) + static_cast<index_t>(wg) * b, 1, b}};
  };
}

index_fn_t strided(scalar_expr_t block, scalar_expr_t stride, scalar_expr_t base) {
  return [block, stride, base](int wg, int, const env_t& env) -> index_set_t {
    const index_t start = base.eval(env) + static_cast<index_t>(wg) * stride.eval(env);
    return index_set_t{range_t{start, 1, block.eval(env)}};
  };
}

index_fn_t rows(scalar_expr_t block_rows, scalar_expr_t row_len, scalar_expr_t base) {
  return [block_rows, row_len, base](int wg, int, const env_t& env) -> index_set_t {
    const index_t span = block_rows.eval(env) * row_len.eval(env);
    return index_set_t{range_t{base.eval(env) + static_cast<index_t>(wg) * span, 1, span}};
  };
}

index_fn_t streamed(scalar_expr_t block, scalar_expr_t num_iters, scalar_expr_t base) {
  return [block, num_iters, base](int wg, int it, const env_t& env) -> index_set_t {
    const index_t b      = block.eval(env);
    const index_t k      = std::max<index_t>(num_iters.eval(env), 1);
    const index_t chunk  = ceil_div_int(b, k);
    const index_t base_i = base.eval(env) + static_cast<index_t>(wg) * b;
    const index_t start  = base_i + static_cast<index_t>(it) * chunk;
    const index_t count  = std::max<index_t>(0, std::min(chunk, base_i + b - start));
    return index_set_t{range_t{start, 1, count}};
  };
}

index_fn_t callable_set(std::function<std::vector<index_t>(int wg, int it, const env_t& env)> fn) {
  return [fn = std::move(fn)](int wg, int it, const env_t& env) -> index_set_t {
    return index_set_t{finite_set_t::of(fn(wg, it, env))};
  };
}

}  // namespace origami::graphs
