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

// Test harness for the origami::graphs suite.
//
// Exposes the same four-macro surface the comm suite uses — TEST, CHECK,
// CHECK_NEAR, ORIGAMI_TEST_MAIN — over two backends:
//
//   Catch2 (default)     the upstream CMake build, matching tests/comm exactly,
//                        so these cases run in CI alongside every other suite.
//   ORIGAMI_GRAPHS_NO_CATCH2
//                        a ~60-line self-contained runner. The graphs core is
//                        pure C++17 with no HIP and no third-party dependency,
//                        and this backend keeps that property testable: the
//                        whole suite compiles and runs with nothing but a C++17
//                        compiler, on a machine with no ROCm and no network to
//                        fetch Catch2 from. That is a property worth being able
//                        to demonstrate, not just assert.
//
// Both backends must stay behaviourally identical, so test bodies never name
// either one.
#pragma once

#if defined(ORIGAMI_GRAPHS_NO_CATCH2)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace origami::graphs::test {

using case_fn_t = void (*)();

/** @brief Registered test cases, in registration order. */
inline std::vector<std::pair<std::string, case_fn_t>>& cases() {
  static std::vector<std::pair<std::string, case_fn_t>> registry;
  return registry;
}

/** @brief Failure count for the case currently running. */
inline int& current_failures() {
  static int failures = 0;
  return failures;
}

/** @brief Register one case; returns a value so it can initialise a global. */
inline bool register_case(const char* name, case_fn_t fn) {
  cases().emplace_back(name, fn);
  return true;
}

/** @brief Record one assertion result, printing a location on failure. */
inline void record(bool ok, const char* expr, const char* file, int line) {
  if (ok) return;
  ++current_failures();
  std::printf("    FAILED %s:%d: %s\n", file, line, expr);
}

/**
 * @brief Run every registered case; returns a process exit status.
 *
 * Duplicate names are rejected up front because Catch2 rejects them too, and a
 * name colliding across two files is easy to do by accident. Without this check
 * the standalone backend would happily run both and the failure would only
 * appear in the Catch2 build, which is the slower place to find out.
 */
inline int run_all() {
  std::vector<std::string> names;
  for (const auto& [name, fn] : cases()) {
    (void)fn;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  const auto clash = std::adjacent_find(names.begin(), names.end());
  if (clash != names.end()) {
    std::printf("duplicate test case name: %s\n", clash->c_str());
    return 1;
  }

  int failed_cases = 0;
  for (const auto& [name, fn] : cases()) {
    current_failures() = 0;
    fn();
    if (current_failures() > 0) {
      ++failed_cases;
      std::printf("  FAIL  %s (%d assertion(s))\n", name.c_str(), current_failures());
    } else {
      std::printf("  ok    %s\n", name.c_str());
    }
  }
  std::printf("\n%zu case(s), %d failed\n", cases().size(), failed_cases);
  return failed_cases == 0 ? 0 : 1;
}

}  // namespace origami::graphs::test

#define TEST(name)                                                                            \
  static void name();                                                                         \
  static const bool name##_registered = ::origami::graphs::test::register_case(#name, &name); \
  static void name()

#define CHECK(cond) ::origami::graphs::test::record((cond), #cond, __FILE__, __LINE__)

#define CHECK_NEAR(a, b, tol)                                                                 \
  ::origami::graphs::test::record(                                                            \
      std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= static_cast<double>(tol), \
      #a " ~= " #b,                                                                           \
      __FILE__,                                                                               \
      __LINE__)

#define ORIGAMI_TEST_MAIN()

#else  // Catch2 backend

#include <catch2/catch_test_macros.hpp>

#include <cmath>

// Catch2 already provides CHECK(cond) with the matching non-fatal semantics
// (records the failure, continues the test), so we only add the rest.
#define TEST(name) TEST_CASE("graphs: " #name)

#define CHECK_NEAR(a, b, tol) \
  CHECK(std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= static_cast<double>(tol))

#define ORIGAMI_TEST_MAIN()

#endif
