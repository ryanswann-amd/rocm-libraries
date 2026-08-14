// SPDX-License-Identifier: MIT
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file cost_expr.hpp
 * @brief origami::graphs — deferred, real-valued cost expressions.
 *
 * `scalar_expr_t` describes shapes and grid counts. It is deliberately
 * non-negative-integer valued, because index arithmetic has to be exact for
 * range intersection to stay decidable. Durations are a different animal: they
 * are real, they divide by rates, and they take minima and maxima against each
 * other. Weakening the integer type to carry them would cost the exactness the
 * derivation depends on, so this is a sibling rather than a generalisation.
 *
 * The two meet in one place. A `cost_expr_t` can embed a `scalar_expr_t`, which
 * evaluates exactly in integer arithmetic and only then becomes a double. That
 * is what lets a tile count be computed exactly and a rate be applied to it in
 * the same expression.
 *
 * Four namespaces are addressable here, against two on the index side:
 *
 *   - `problem` and `config`, as everywhere else.
 *   - `hardware`, the machine description, which is not known when a graph is
 *     instantiated because a graph is device-independent. It arrives at ranking.
 *   - `runtime`, live scheduler state such as the number of CUs currently busy.
 *     This is the whole reason cost is deferred rather than a number: the same
 *     node is worth a different amount depending on what else is resident.
 *
 * A cost expression is built once per node and evaluated once per dispatch, so
 * construction cost is amortised while evaluation stays a tree walk.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>

#include "origami/graphs/symbolic.hpp"
#include "origami/graphs/types.hpp"

namespace origami::graphs {

/**
 * @brief Everything a cost expression can be resolved against.
 *
 * The problem and config come from the graph, which recorded them when it was
 * instantiated. Hardware is supplied at ranking. The runtime map is refilled by
 * the scheduler at every dispatch.
 */
struct cost_context_t {
  problem_t problem;     ///< problem-scope bindings
  config_t config;       ///< config-scope bindings
  param_map_t hardware;  ///< machine description, by canonical name
  param_map_t runtime;   ///< live scheduler state, by canonical name

  /**
   * @brief Resolve a symbol in any of the four scopes.
   *
   * @param scope Namespace to look in.
   * @param name Symbol name.
   * @return double The bound value.
   * @throws std::out_of_range If the symbol is unbound.
   */
  double number_at(scope_t scope, const std::string& name) const;

  /** @brief The problem and config, for evaluating an embedded index expression. */
  eval_context_t index_context() const { return eval_context_t{problem, config}; }
};

/** @brief Operator carried by an interior `cost_expr_t` node. */
enum class cost_expr_kind_t : std::uint8_t {
  constant,  ///< literal real leaf
  symbol,    ///< named leaf in one of the four scopes
  integer,   ///< an embedded scalar_expr_t, evaluated exactly then widened
  add,       ///< a + b
  sub,       ///< a - b
  mul,       ///< a * b
  div,       ///< a / b
  minimum,   ///< min(a, b)
  maximum,   ///< max(a, b) — the roofline operator
};

/**
 * @brief A deferred real number: a duration, a rate, or a byte count.
 *
 * Value type over a shared immutable tree, so copies are a refcount bump and a
 * sub-expression can be shared between the nodes of an operation.
 */
class cost_expr_t {
 public:
  /** @brief Construct the literal zero. */
  cost_expr_t();

  /**
   * @brief Wrap a literal real.
   *
   * Implicit, so a builder taking a `cost_expr_t` also accepts `2.0`.
   *
   * @param value Literal value.
   */
  cost_expr_t(double value);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Embed an exact integer expression.
   *
   * Implicit, so index arithmetic composes with rates without ceremony. The
   * embedded expression is evaluated in integer arithmetic and widened only
   * afterwards, so a tile count is never computed in floating point.
   *
   * @param expr Integer expression over the problem and config scopes.
   */
  cost_expr_t(scalar_expr_t expr);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Evaluate against concrete bindings.
   *
   * @param ctx Problem, config, hardware and runtime bindings.
   * @return double The evaluated value.
   * @throws std::out_of_range If a referenced symbol is unbound.
   * @throws std::domain_error On division by zero.
   */
  double eval(const cost_context_t& ctx) const;

  /**
   * @brief Every free symbol, including those inside embedded index expressions.
   *
   * @return std::set<scoped_symbol_t> Free symbols, ordered by scope then name.
   */
  std::set<scoped_symbol_t> scoped_symbols() const;

  /**
   * @brief Render the tree for diagnostics.
   *
   * @return std::string Infix rendering.
   */
  std::string str() const;

  /** @brief Structural node kind, exposed for testing. */
  cost_expr_kind_t kind() const;

 private:
  struct node_t;
  explicit cost_expr_t(std::shared_ptr<const node_t> node);

  friend cost_expr_t cost_sym(scope_t scope, std::string name);
  friend cost_expr_t make_cost_binary(cost_expr_kind_t kind,
                                      const cost_expr_t& lhs,
                                      const cost_expr_t& rhs);

  std::shared_ptr<const node_t> node_;
};

/**
 * @brief Reference a name in any of the four scopes.
 *
 * @param scope Namespace the name resolves against.
 * @param name Symbol name.
 * @return cost_expr_t A symbol leaf.
 */
cost_expr_t cost_sym(scope_t scope, std::string name);

/**
 * @brief Reference a machine characteristic, e.g. `hardware_sym("hbm_read_bw")`.
 *
 * Most names `hardware_params` publishes are per-cycle, the unit a cost
 * expression should evaluate to. A handful are per-second convenience forms
 * instead, and dividing by one of those lands the whole expression back in
 * seconds with no clock symbol anywhere in sight to warn a reader. See
 * `hardware_params`'s doc for exactly which names are which before dividing
 * by one you have not checked.
 *
 * @param name Canonical hardware parameter name; see `hardware_params`.
 * @return cost_expr_t A hardware-scope symbol leaf.
 */
cost_expr_t hardware_sym(std::string name);

/**
 * @brief Reference live scheduler state, e.g. `runtime_sym("active_cus")`.
 *
 * @param name Canonical runtime parameter name.
 * @return cost_expr_t A runtime-scope symbol leaf.
 */
cost_expr_t runtime_sym(std::string name);

/**
 * @brief Build an interior cost node; prefer the operators below.
 *
 * @param kind Operator to apply.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return cost_expr_t The combined expression.
 */
cost_expr_t make_cost_binary(cost_expr_kind_t kind, const cost_expr_t& lhs, const cost_expr_t& rhs);

cost_expr_t operator+(const cost_expr_t& lhs, const cost_expr_t& rhs);
cost_expr_t operator-(const cost_expr_t& lhs, const cost_expr_t& rhs);
cost_expr_t operator*(const cost_expr_t& lhs, const cost_expr_t& rhs);
cost_expr_t operator/(const cost_expr_t& lhs, const cost_expr_t& rhs);

/**
 * @brief Real-valued `min(a, b)`.
 *
 * @param a First operand.
 * @param b Second operand.
 * @return cost_expr_t The minimum expression.
 */
cost_expr_t minimum(const cost_expr_t& a, const cost_expr_t& b);

/**
 * @brief Real-valued `max(a, b)` — how a roofline is written.
 *
 * @param a First operand.
 * @param b Second operand.
 * @return cost_expr_t The maximum expression.
 */
cost_expr_t maximum(const cost_expr_t& a, const cost_expr_t& b);

}  // namespace origami::graphs
