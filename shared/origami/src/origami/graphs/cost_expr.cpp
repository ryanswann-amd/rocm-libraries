// SPDX-License-Identifier: MIT
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file cost_expr.cpp
 * @brief Implementation of the deferred real-valued cost expression.
 */

#include "origami/graphs/cost_expr.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace origami::graphs {

// ─── cost_context_t ───────────────────────────────────────────────────

double cost_context_t::number_at(scope_t scope, const std::string& name) const {
  switch (scope) {
    case scope_t::problem: return problem.number_at(name, scope);
    case scope_t::config: return config.number_at(name, scope);
    case scope_t::hardware: return hardware.number_at(name, scope);
    case scope_t::runtime: return runtime.number_at(name, scope);
  }
  throw std::logic_error("cost_context_t: unhandled scope");
}

// ─── cost_expr_t ──────────────────────────────────────────────────────

struct cost_expr_t::node_t {
  cost_expr_kind_t kind = cost_expr_kind_t::constant;
  double value          = 0.0;
  std::string name;
  scope_t scope = scope_t::problem;
  scalar_expr_t integer;
  std::shared_ptr<const node_t> left;
  std::shared_ptr<const node_t> right;

  double eval(const cost_context_t& ctx) const {
    switch (kind) {
      case cost_expr_kind_t::constant: return value;
      case cost_expr_kind_t::symbol: return ctx.number_at(scope, name);
      // Evaluated in exact integer arithmetic, widened only afterwards, so a
      // tile count is never computed in floating point.
      case cost_expr_kind_t::integer: return static_cast<double>(integer.eval(ctx.index_context()));
      default: break;
    }

    const double a = left->eval(ctx);
    const double b = right->eval(ctx);
    switch (kind) {
      case cost_expr_kind_t::add: return a + b;
      case cost_expr_kind_t::sub: return a - b;
      case cost_expr_kind_t::mul: return a * b;
      case cost_expr_kind_t::div:
        // Rejected rather than returning an infinity: a zero rate is a
        // modelling mistake, and an inf would propagate into a ranking as a
        // merely very slow candidate instead of an error.
        if (b == 0.0) throw std::domain_error("cost_expr_t: division by zero");
        return a / b;
      case cost_expr_kind_t::minimum: return std::min(a, b);
      case cost_expr_kind_t::maximum: return std::max(a, b);
      default: throw std::logic_error("cost_expr_t: unhandled expression kind");
    }
  }

  void collect_scoped(std::set<scoped_symbol_t>& out) const {
    switch (kind) {
      case cost_expr_kind_t::symbol: out.insert(scoped_symbol_t{scope, name}); return;
      case cost_expr_kind_t::integer: {
        for (const scoped_symbol_t& s : integer.scoped_symbols()) out.insert(s);
        return;
      }
      default: break;
    }
    if (left) left->collect_scoped(out);
    if (right) right->collect_scoped(out);
  }

  std::string str() const {
    switch (kind) {
      case cost_expr_kind_t::constant: return std::to_string(value);
      case cost_expr_kind_t::symbol: return std::string{scope_name(scope)} + "." + name;
      case cost_expr_kind_t::integer: return integer.str();
      case cost_expr_kind_t::add: return "(" + left->str() + " + " + right->str() + ")";
      case cost_expr_kind_t::sub: return "(" + left->str() + " - " + right->str() + ")";
      case cost_expr_kind_t::mul: return "(" + left->str() + " * " + right->str() + ")";
      case cost_expr_kind_t::div: return "(" + left->str() + " / " + right->str() + ")";
      case cost_expr_kind_t::minimum: return "min(" + left->str() + ", " + right->str() + ")";
      case cost_expr_kind_t::maximum: return "max(" + left->str() + ", " + right->str() + ")";
    }
    throw std::logic_error("cost_expr_t: unhandled expression kind");
  }
};

cost_expr_t::cost_expr_t() : cost_expr_t(0.0) {}

cost_expr_t::cost_expr_t(double value) {
  auto n   = std::make_shared<node_t>();
  n->kind  = cost_expr_kind_t::constant;
  n->value = value;
  node_    = std::move(n);
}

cost_expr_t::cost_expr_t(scalar_expr_t expr) {
  // A bare symbol crossing into a cost expression is resolved as a real, while
  // any compound expression keeps its exact integer evaluation. The distinction
  // is not a special case so much as the absence of one: exactness is a
  // property of arithmetic, and a lone symbol performs none. It is what lets
  // `config("compute_efficiency")` be 0.8 while `ceil_div(N, BN)` a few
  // characters away still rounds like an integer.
  if (expr.kind() == expr_kind_t::symbol) {
    const std::set<scoped_symbol_t> named = expr.scoped_symbols();
    node_ = cost_sym(named.begin()->scope, named.begin()->name).node_;
    return;
  }

  auto n     = std::make_shared<node_t>();
  n->kind    = cost_expr_kind_t::integer;
  n->integer = std::move(expr);
  node_      = std::move(n);
}

cost_expr_t::cost_expr_t(std::shared_ptr<const node_t> node) : node_(std::move(node)) {}

double cost_expr_t::eval(const cost_context_t& ctx) const { return node_->eval(ctx); }

std::set<scoped_symbol_t> cost_expr_t::scoped_symbols() const {
  std::set<scoped_symbol_t> out;
  node_->collect_scoped(out);
  return out;
}

std::string cost_expr_t::str() const { return node_->str(); }

cost_expr_kind_t cost_expr_t::kind() const { return node_->kind; }

cost_expr_t cost_sym(scope_t scope, std::string name) {
  auto n   = std::make_shared<cost_expr_t::node_t>();
  n->kind  = cost_expr_kind_t::symbol;
  n->name  = std::move(name);
  n->scope = scope;
  return cost_expr_t{std::move(n)};
}

cost_expr_t hardware_sym(std::string name) { return cost_sym(scope_t::hardware, std::move(name)); }

cost_expr_t runtime_sym(std::string name) { return cost_sym(scope_t::runtime, std::move(name)); }

cost_expr_t make_cost_binary(cost_expr_kind_t kind,
                             const cost_expr_t& lhs,
                             const cost_expr_t& rhs) {
  auto n   = std::make_shared<cost_expr_t::node_t>();
  n->kind  = kind;
  n->left  = lhs.node_;
  n->right = rhs.node_;
  return cost_expr_t{std::move(n)};
}

cost_expr_t operator+(const cost_expr_t& lhs, const cost_expr_t& rhs) {
  return make_cost_binary(cost_expr_kind_t::add, lhs, rhs);
}

cost_expr_t operator-(const cost_expr_t& lhs, const cost_expr_t& rhs) {
  return make_cost_binary(cost_expr_kind_t::sub, lhs, rhs);
}

cost_expr_t operator*(const cost_expr_t& lhs, const cost_expr_t& rhs) {
  return make_cost_binary(cost_expr_kind_t::mul, lhs, rhs);
}

cost_expr_t operator/(const cost_expr_t& lhs, const cost_expr_t& rhs) {
  return make_cost_binary(cost_expr_kind_t::div, lhs, rhs);
}

cost_expr_t minimum(const cost_expr_t& a, const cost_expr_t& b) {
  return make_cost_binary(cost_expr_kind_t::minimum, a, b);
}

cost_expr_t maximum(const cost_expr_t& a, const cost_expr_t& b) {
  return make_cost_binary(cost_expr_kind_t::maximum, a, b);
}

}  // namespace origami::graphs
