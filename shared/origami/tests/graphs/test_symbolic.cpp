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
#include <stdexcept>
#include <vector>

#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/**
 * Deterministic 64-bit LCG.
 *
 * The fuzz gate must reproduce exactly across machines and runs — a failure
 * that cannot be replayed is not a usable signal — so the generator is fixed
 * here rather than taken from <random>, whose engines are portable but whose
 * distributions are not.
 */
class lcg_t {
 public:
  explicit lcg_t(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return state_ >> 17;
  }

  /** Uniform in [lo, hi]. */
  index_t range(index_t lo, index_t hi) {
    return lo + static_cast<index_t>(next() % static_cast<std::uint64_t>(hi - lo + 1));
  }

 private:
  std::uint64_t state_;
};

/** Brute-force intersection, used only as the fuzz oracle. */
std::vector<index_t> brute_force_intersect(const range_t& a, const range_t& b) {
  const std::vector<index_t> va = to_vector(index_set_t{a});
  const std::vector<index_t> vb = to_vector(index_set_t{b});
  std::vector<index_t> out;
  std::set_intersection(va.begin(), va.end(), vb.begin(), vb.end(), std::back_inserter(out));
  return out;
}

}  // namespace

// ─── scalar expressions ───────────────────────────────────────────────

TEST(scalar_expr_evaluates_against_bindings) {
  const scalar_expr_t m = sym("M");
  const scalar_expr_t n = sym("N");

  env_t env;
  env["M"] = 8192;
  env["N"] = 4096;

  CHECK((m + n).eval(env) == 12288);
  CHECK((m - n).eval(env) == 4096);
  CHECK((m * 2).eval(env) == 16384);
  CHECK((m / 3).eval(env) == 2730);
  CHECK(ceil_div(m, 3).eval(env) == 2731);
}

TEST(scalar_expr_literals_coerce_on_both_sides) {
  const scalar_expr_t m = sym("M");
  env_t env;
  env["M"] = 10;

  CHECK((2 + m).eval(env) == 12);
  CHECK((m + 2).eval(env) == 12);
  CHECK((100 - m).eval(env) == 90);
}

TEST(grid_2d_is_the_product_of_tile_counts) {
  env_t env;
  env["M"] = 8192;
  env["N"] = 8192;

  // 8192/256 = 32 tiles each way.
  CHECK(grid_2d(sym("M"), sym("N"), 256, 256).eval(env) == 1024);
  // Non-divisible dimensions round up independently.
  CHECK(grid_2d(300, 300, 256, 256).eval() == 4);
}

TEST(scalar_expr_reports_free_symbols) {
  const scalar_expr_t e         = grid_2d(sym("M"), sym("N"), 128, sym("BN"));
  const std::set<std::string> f = e.free_symbols();

  CHECK(f.size() == 3);
  CHECK(f.count("M") == 1);
  CHECK(f.count("N") == 1);
  CHECK(f.count("BN") == 1);
  CHECK(!e.is_constant());
  CHECK(scalar_expr_t{7}.is_constant());
}

TEST(scalar_expr_unbound_dimension_throws) {
  bool threw = false;
  try {
    sym("M").eval(env_t{});
  } catch (const std::out_of_range&) { threw = true; }
  CHECK(threw);
}

// ─── integer division follows Python, not C++ ─────────────────────────

TEST(division_helpers_match_python_semantics) {
  // C++ / would give -3 here; Python's // gives -4, and the CRT solve depends
  // on the floor behaviour.
  CHECK(floor_div(-7, 2) == -4);
  CHECK(floor_div(7, -2) == -4);
  CHECK(floor_div(7, 2) == 3);
  CHECK(floor_div(-8, 2) == -4);

  CHECK(ceil_div_int(7, 2) == 4);
  CHECK(ceil_div_int(-7, 2) == -3);
  CHECK(ceil_div_int(8, 2) == 4);

  // C++ % would give -1 here; Python's % gives 2.
  CHECK(euclid_mod(-7, 3) == 2);
  CHECK(euclid_mod(7, 3) == 1);
  CHECK(euclid_mod(-6, 3) == 0);
}

TEST(division_by_zero_throws) {
  bool threw = false;
  try {
    floor_div(1, 0);
  } catch (const std::domain_error&) { threw = true; }
  CHECK(threw);
}

// ─── index sets ───────────────────────────────────────────────────────

TEST(range_enforces_its_invariants) {
  bool threw_step = false;
  try {
    range_t{0, 0, 4};
  } catch (const std::invalid_argument&) { threw_step = true; }
  CHECK(threw_step);

  bool threw_count = false;
  try {
    range_t{0, 1, -1};
  } catch (const std::invalid_argument&) { threw_count = true; }
  CHECK(threw_count);

  CHECK(range_t{}.empty());

  // Bound to a local first: a braced initializer's commas would otherwise split
  // the assertion macro's argument list.
  const range_t r{5, 3, 4};
  CHECK(r.last() == 14);
}

TEST(finite_set_sorts_and_deduplicates) {
  const finite_set_t f = finite_set_t::of({5, 1, 5, 3, 1});
  CHECK(f.size() == 3);
  CHECK(f.indices()[0] == 1);
  CHECK(f.indices()[1] == 3);
  CHECK(f.indices()[2] == 5);
}

TEST(contiguous_ranges_intersect_on_their_overlap) {
  // [0,10) against [5,15) shares [5,10) = 5 elements.
  const index_set_t a = index_set_t{range_t{0, 1, 10}};
  const index_set_t b = index_set_t{range_t{5, 1, 10}};
  CHECK(intersect_size(a, b) == 5);

  const index_set_t r = intersect(a, b);
  CHECK(size(r) == 5);
  CHECK(to_vector(r).front() == 5);
  CHECK(to_vector(r).back() == 9);
}

TEST(disjoint_ranges_intersect_empty) {
  const index_set_t a = index_set_t{range_t{0, 1, 10}};
  const index_set_t b = index_set_t{range_t{100, 1, 10}};
  CHECK(intersect_size(a, b) == 0);
  CHECK(is_empty(intersect(a, b)));
}

TEST(strided_ranges_align_only_on_the_lcm) {
  // Evens and multiples of three overlap on multiples of six.
  const index_set_t evens  = index_set_t{range_t{0, 2, 50}};  // 0,2,...,98
  const index_set_t threes = index_set_t{range_t{0, 3, 34}};  // 0,3,...,99
  const index_set_t r      = intersect(evens, threes);

  const auto* rr = std::get_if<range_t>(&r);
  CHECK(rr != nullptr);
  if (rr != nullptr) {
    CHECK(rr->step == 6);
    CHECK(rr->start == 0);
    // 0,6,...,96 -> 17 elements.
    CHECK(rr->count == 17);
  }
}

TEST(offset_progressions_that_never_align_are_empty) {
  // Odd and even indices share nothing regardless of overlap in value range.
  const index_set_t evens = index_set_t{range_t{0, 2, 50}};
  const index_set_t odds  = index_set_t{range_t{1, 2, 50}};
  CHECK(intersect_size(evens, odds) == 0);
}

TEST(empty_range_intersects_to_empty) {
  const index_set_t a = index_set_t{range_t{}};
  const index_set_t b = index_set_t{range_t{0, 1, 10}};
  CHECK(intersect_size(a, b) == 0);
  CHECK(intersect_size(b, a) == 0);
}

TEST(mixed_and_explicit_sets_intersect) {
  const index_set_t f = index_set_t{finite_set_t::of({1, 4, 7, 9, 12})};
  const index_set_t r = index_set_t{range_t{0, 3, 5}};  // 0,3,6,9,12

  // Shared: 9 and 12.
  CHECK(intersect_size(f, r) == 2);
  CHECK(intersect_size(r, f) == 2);
  CHECK(size(intersect(f, r)) == 2);
  CHECK(to_vector(intersect(r, f))[0] == 9);

  const index_set_t g = index_set_t{finite_set_t::of({4, 9, 100})};
  CHECK(intersect_size(f, g) == 2);
  CHECK(size(intersect(f, g)) == 2);
}

TEST(crt_intersection_matches_brute_force_over_20k_pairs) {
  lcg_t rng(0xC0FFEEULL);
  int mismatches = 0;

  for (int trial = 0; trial < 20000; ++trial) {
    const range_t a{rng.range(0, 60), rng.range(1, 12), rng.range(0, 25)};
    const range_t b{rng.range(0, 60), rng.range(1, 12), rng.range(0, 25)};

    const std::vector<index_t> expected = brute_force_intersect(a, b);
    const index_set_t got               = intersect(index_set_t{a}, index_set_t{b});

    if (size(got) != static_cast<index_t>(expected.size()) || to_vector(got) != expected ||
        intersect_size(index_set_t{a}, index_set_t{b}) != static_cast<index_t>(expected.size())) {
      ++mismatches;
    }
  }

  CHECK(mismatches == 0);
}

// ─── pattern builders ─────────────────────────────────────────────────

TEST(contiguous_gives_each_workgroup_its_own_block) {
  const index_fn_t f = contiguous(64);
  const env_t env;

  CHECK(size(f(0, 0, env)) == 64);
  CHECK(to_vector(f(0, 0, env)).front() == 0);
  CHECK(to_vector(f(3, 0, env)).front() == 192);
  // Adjacent workgroups tile the space without overlapping.
  CHECK(intersect_size(f(0, 0, env), f(1, 0, env)) == 0);
}

TEST(contiguous_resolves_a_symbolic_block) {
  const index_fn_t f = contiguous(sym("BM"), 1000);
  env_t env;
  env["BM"] = 8;

  CHECK(size(f(2, 0, env)) == 8);
  CHECK(to_vector(f(2, 0, env)).front() == 1016);
}

TEST(strided_overlaps_when_stride_is_shorter_than_block) {
  const index_fn_t f = strided(10, 4);
  const env_t env;

  CHECK(to_vector(f(1, 0, env)).front() == 4);
  // Blocks of 10 every 4 indices overlap in 6.
  CHECK(intersect_size(f(0, 0, env), f(1, 0, env)) == 6);
}

TEST(rows_flattens_a_row_band) {
  const index_fn_t f = rows(2, sym("N"));
  env_t env;
  env["N"] = 128;

  CHECK(size(f(0, 0, env)) == 256);
  CHECK(to_vector(f(1, 0, env)).front() == 256);
  CHECK(intersect_size(f(0, 0, env), f(1, 0, env)) == 0);
}

TEST(streamed_partitions_a_block_across_iterations) {
  const index_fn_t f = streamed(100, 4);
  const env_t env;

  // Four iterations of 25 partition the block exactly.
  CHECK(size(f(0, 0, env)) == 25);
  CHECK(size(f(0, 3, env)) == 25);
  CHECK(to_vector(f(0, 1, env)).front() == 25);
  CHECK(intersect_size(f(0, 0, env), f(0, 1, env)) == 0);

  index_t total = 0;
  for (int it = 0; it < 4; ++it) total += size(f(0, it, env));
  CHECK(total == 100);
}

TEST(streamed_clips_the_final_iteration) {
  // 10 over 4 iterations: chunks of 3, 3, 3, and a clipped 1.
  const index_fn_t f = streamed(10, 4);
  const env_t env;

  CHECK(size(f(0, 0, env)) == 3);
  CHECK(size(f(0, 3, env)) == 1);

  index_t total = 0;
  for (int it = 0; it < 4; ++it) total += size(f(0, it, env));
  CHECK(total == 10);
}

TEST(index_offsets_wraps_arbitrary_indices) {
  const index_fn_t f = index_offsets([](int wg, int, const eval_context_t&) {
    return std::vector<index_t>{wg * 10, wg * 10 + 5, wg * 10};
  });
  const eval_context_t ctx;

  // Duplicates collapse.
  CHECK(size(f(2, 0, ctx)) == 2);
  CHECK(to_vector(f(2, 0, ctx))[0] == 20);
  CHECK(to_vector(f(2, 0, ctx))[1] == 25);
}

ORIGAMI_TEST_MAIN()
