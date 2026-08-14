// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// origami.graphs — Python bindings for the workgroup dependency graph model.
//
// Bound in its own translation unit rather than inline in bindings.cpp the way
// origami.comm is, purely because of size: the graphs surface spans five layers
// and would bury the GEMM and comm bindings it sits next to.
//
// Two places where the Python surface deliberately differs from the C++ one:
//
//   - Schedules expose start/duration/lane as *methods* taking a node, rather
//     than as the dictionaries they are in C++. Those dictionaries are keyed by
//     a custom hash map, and a caller reaching for `s.start[node]` in Python
//     wants a lookup, not the container.
//   - `comm_spec_t::work_graph` is built from a list of primitive names rather
//     than from bound primitive structs. Binding seven structs and a variant to
//     express `["load", "store", "push"]` would be a lot of surface for no
//     expressiveness; the peer-taking primitives take a shared `peer` argument.
//
// `cost_model_t` and `runtime_t` are subclassable from Python, through the
// trampolines below. A cost model is the seam where hardware enters and a
// runtime is the seam where a scheduling policy does; a caller who has a machine
// or a policy the library does not ship should not have to write C++ to say so.
// Iris is the first such caller: its layout kernel pins workgroups to lanes for
// the whole launch, which is not how any shipped policy dispatches.
//
// `runtime_t`'s trampoline replaces the one the seconds-era `cost_runtime_t`
// carried. The surface is smaller, but the capability is not the same: a
// Python `cost_runtime_t` wrote a whole scheduler and so could emit arbitrary
// start times — a non-greedy or backfilling policy, or a replayed measurement
// — whereas a Python `runtime_t` only supplies a priority, a placement, an
// order and an optional gate while simulate() owns the loop and its greedy
// dispatch. Nothing concrete is blocked today (test_graphs.py's SerialRuntime
// is a lane-pinning policy expressed entirely in those terms), and
// schedule_t.place is still bound, so a hand-built schedule can still be
// traced — it just cannot be fed to predict_latency.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "origami/graphs/origami_graphs.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace og = origami::graphs;
namespace oc = origami::comm;

namespace {

/** Lets a Python class price nodes and hops, in cycles. */
struct py_cost_model_t : og::cost_model_t {
  NB_TRAMPOLINE(og::cost_model_t, 3);

  double node_cycles(const og::wg_node_t& node) const override {
    NB_OVERRIDE_PURE(node_cycles, node);
  }
  double edge_cycles(const og::edge_t& edge) const override { NB_OVERRIDE_PURE(edge_cycles, edge); }
  double node_cycles_at(const og::wg_node_t& node, int active_cus) const override {
    NB_OVERRIDE(node_cycles_at, node, active_cus);
  }
};

/** Lets a Python class choose lanes. */
struct py_placement_t : og::placement_t {
  NB_TRAMPOLINE(og::placement_t, 2);

  /** As `py_runtime_t::name`: a reference needs somewhere to point. */
  const std::string& name() const override {
    nb::detail::ticket nb_ticket(nb_trampoline, "name", true);
    name_ = nb::cast<std::string>(nb_trampoline.base().attr(nb_ticket.key)());
    return name_;
  }

  int choose(const og::placement_context_t& ctx) const override { NB_OVERRIDE_PURE(choose, ctx); }

 private:
  mutable std::string name_;
};

/** Lets a Python class be a scheduling policy simulate() can drive. */
struct py_runtime_t : og::runtime_t {
  NB_TRAMPOLINE(og::runtime_t, 5);

  /**
   * `name()` and `placement()` hand back references, and Python hands back fresh
   * objects, so the macros cannot be used for either: they would cast into a
   * temporary and return a reference to it. Keeping the last answer in the
   * trampoline gives the reference something to point at for as long as any
   * caller can hold it — which is all simulate() needs, since it asks once and
   * uses the answer for the length of one schedule.
   */
  const std::string& name() const override {
    nb::detail::ticket nb_ticket(nb_trampoline, "name", true);
    name_ = nb::cast<std::string>(nb_trampoline.base().attr(nb_ticket.key)());
    return name_;
  }

  const og::placement_t& placement() const override {
    nb::detail::ticket nb_ticket(nb_trampoline, "placement", true);
    placement_ = nb_trampoline.base().attr(nb_ticket.key)();
    return nb::cast<const og::placement_t&>(placement_);
  }

  og::wg_node_map_t<std::size_t> priority(const og::wg_graph_t& graph) const override {
    NB_OVERRIDE_PURE(priority, graph);
  }

  og::queue_order_t order() const override { NB_OVERRIDE(order); }

  bool gated(const og::wg_node_t& node, const og::sim_state_t& state) const override {
    NB_OVERRIDE(gated, node, state);
  }

 private:
  mutable std::string name_;
  mutable nb::object placement_;
};

/** Build a comm work graph from primitive names, as the C++ tests do inline. */
std::vector<oc::op_t> work_graph_from_names(const std::vector<std::string>& names, int peer) {
  std::vector<oc::op_t> out;
  out.reserve(names.size());
  for (const std::string& name : names) {
    if (name == "load") {
      out.emplace_back(oc::load_t{});
    } else if (name == "store") {
      out.emplace_back(oc::store_t{});
    } else if (name == "push") {
      out.emplace_back(oc::push_t{peer});
    } else if (name == "pull") {
      out.emplace_back(oc::pull_t{peer});
    } else if (name == "reduce") {
      out.emplace_back(oc::reduce_t{});
    } else if (name == "signal") {
      out.emplace_back(oc::signal_t{peer});
    } else if (name == "wait") {
      out.emplace_back(oc::wait_t{peer});
    } else {
      throw std::invalid_argument("unknown comm primitive '" + name +
                                  "' (expected load, store, push, pull, reduce, signal or wait)");
    }
  }
  return out;
}

/** Borrow a list of graphs as the pointer vector the ranking API takes. */
std::vector<const og::wg_graph_t*> as_pointers(const std::vector<og::wg_graph_t*>& graphs) {
  std::vector<const og::wg_graph_t*> out;
  out.reserve(graphs.size());
  for (og::wg_graph_t* g : graphs) out.push_back(g);
  return out;
}

}  // namespace

void bind_graphs(nb::module_& m) {
  auto g = m.def_submodule("graphs",
                           "Workgroup dependency graphs: derive who feeds whom, schedule it, "
                           "price it, and rank the alternatives.");

  // ─── nodes, edges and access patterns ───────────────────────────────

  nb::class_<og::wg_node_t>(g, "wg_node_t", "One workgroup-iteration: the atom of a schedule.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](og::wg_node_t* self, int op, int wg, int it) {
            new (self) og::wg_node_t{op, wg, it};
          },
          "op"_a,
          "wg"_a = 0,
          "it"_a = 0)
      .def_rw("op", &og::wg_node_t::op)
      .def_rw("wg", &og::wg_node_t::wg)
      .def_rw("it", &og::wg_node_t::it)
      .def(
          "__eq__",
          [](const og::wg_node_t& a, const og::wg_node_t& b) { return a == b; },
          nb::is_operator())
      .def("__hash__",
           [](const og::wg_node_t& n) { return static_cast<std::size_t>(og::wg_node_hash_t{}(n)); })
      .def("__repr__", [](const og::wg_node_t& n) {
        return "wg_node_t(op=" + std::to_string(n.op) + ", wg=" + std::to_string(n.wg) +
               ", it=" + std::to_string(n.it) + ")";
      });

  nb::class_<og::edge_t>(g, "edge_t", "A dependency: src must finish before dst may start.")
      .def_ro("src", &og::edge_t::src)
      .def_ro("dst", &og::edge_t::dst)
      .def_ro("allocation", &og::edge_t::allocation)
      .def_ro("weight", &og::edge_t::weight);

  // None of the graphs enums call export_values(), which comm does. Their value
  // names collide: role_t's `read` and `write` are the pattern builders, and
  // both cost_kind_t and runtime_kind_t have a `roofline`. Qualifying them —
  // `runtime_kind_t.roofline` — costs nothing and says which one is meant.
  nb::enum_<og::role_t>(g, "role_t", "Whether an access reads or writes.")
      .value("read", og::role_t::read)
      .value("write", og::role_t::write);

  // ─── namespaces ─────────────────────────────────────────────────────

  nb::enum_<og::scope_t>(g, "scope_t", "The namespace a symbolic name resolves against.")
      .value("problem", og::scope_t::problem)
      .value("config", og::scope_t::config)
      .value("hardware", og::scope_t::hardware)
      .value("runtime", og::scope_t::runtime);

  // problem_t and config_t are separate types over the same storage, so one
  // cannot be passed where the other is meant. Both present the mapping
  // protocol, which is what makes `config["BM"]` and `dict(config)` work.
  const auto bind_params = [&g](auto* tag, const char* name, og::scope_t scope, const char* doc) {
    using params_t = std::remove_pointer_t<decltype(tag)>;
    return nb::class_<params_t, og::param_map_t>(g, name, doc)
        .def(nb::init<>())
        .def(
            "__init__",
            [](params_t* self, og::param_map_t::storage_t values) {
              new (self) params_t(std::move(values));
            },
            "values"_a)
        .def(
            "__getitem__",
            [scope](const params_t& self, const std::string& key) {
              if (!self.contains(key)) throw nb::key_error(key.c_str());
              return self.values().at(key);
            },
            "key"_a)
        .def(
            "__setitem__",
            [](params_t& self, std::string key, og::param_value_t value) {
              self.set(std::move(key), value);
            },
            "key"_a,
            "value"_a)
        .def("__contains__",
             [](const params_t& self, const std::string& key) { return self.contains(key); })
        .def("__len__", &params_t::size)
        .def("__iter__",
             [](const params_t& self) {
               nb::list out;
               for (const auto& [key, value] : self.values()) out.append(key);
               return nb::iter(out);
             })
        .def("keys",
             [](const params_t& self) {
               nb::list out;
               for (const auto& [key, value] : self.values()) out.append(key);
               return out;
             })
        .def(
            "get",
            [](const params_t& self, const std::string& key, nb::object fallback) -> nb::object {
              if (!self.contains(key)) return fallback;
              return nb::cast(self.values().at(key));
            },
            "key"_a,
            "default"_a = nb::none())
        .def("__repr__", [name](const params_t& self) {
          return std::string{name} + "(" + std::to_string(self.size()) + " bound)";
        });
  };

  nb::class_<og::param_map_t>(g, "param_map_t", "A namespace of bound parameters.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](og::param_map_t* self, og::param_map_t::storage_t values) {
            new (self) og::param_map_t(std::move(values));
          },
          "values"_a)
      .def("__len__", &og::param_map_t::size)
      .def("__contains__",
           [](const og::param_map_t& self, const std::string& key) { return self.contains(key); })
      .def(
          "__getitem__",
          [](const og::param_map_t& self, const std::string& key) {
            if (!self.contains(key)) throw nb::key_error(key.c_str());
            return self.values().at(key);
          },
          "key"_a)
      .def(
          "__setitem__",
          [](og::param_map_t& self, std::string key, og::param_value_t value) {
            self.set(std::move(key), value);
          },
          "key"_a,
          "value"_a);

  // So a hardware scope can be given as a plain dict, which is the shortest
  // path when a model needs a name no device description carries.
  nb::implicitly_convertible<nb::dict, og::param_map_t>();

  bind_params(static_cast<og::problem_t*>(nullptr),
              "problem_t",
              og::scope_t::problem,
              "Problem dimensions: what every candidate shares.");
  bind_params(static_cast<og::config_t*>(nullptr),
              "config_t",
              og::scope_t::config,
              "Tuning parameters: what differs between candidates.");

  nb::implicitly_convertible<nb::dict, og::problem_t>();
  nb::implicitly_convertible<nb::dict, og::config_t>();

  nb::class_<og::eval_context_t>(
      g, "eval_context_t", "The problem and config an index map is resolved against.")
      .def(nb::init<>())
      .def(nb::init<og::problem_t, og::config_t>(), "problem"_a, "config"_a)
      .def_rw("problem", &og::eval_context_t::problem)
      .def_rw("config", &og::eval_context_t::config);

  // ─── integer expressions ────────────────────────────────────────────

  // Bound with an implicit conversion from int, which is what makes
  // `contiguous(4)` work in Python the way `contiguous(4)` works in C++.
  // The operators carry nb::is_operator() so that mixing an integer expression
  // with a cost expression falls through to the cost expression's reflected
  // operator rather than raising: that fallthrough is exactly how a tile's
  // exact element count meets a rate.
  nb::class_<og::scalar_expr_t>(
      g, "scalar_expr_t", "A non-negative integer, possibly symbolic in a problem dimension.")
      .def(nb::init<origami::graphs::index_t>(), "value"_a)
      .def("eval",
           nb::overload_cast<const origami::graphs::env_t&>(&og::scalar_expr_t::eval, nb::const_),
           "env"_a,
           "Resolve against problem-scope bindings.")
      .def("eval",
           nb::overload_cast<const og::eval_context_t&>(&og::scalar_expr_t::eval, nb::const_),
           "ctx"_a,
           "Resolve against both namespaces.")
      .def("eval",
           nb::overload_cast<>(&og::scalar_expr_t::eval, nb::const_),
           "Resolve an expression that is already concrete.")
      .def("str", &og::scalar_expr_t::str)
      .def("__repr__", &og::scalar_expr_t::str)
      .def(
          "__add__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return a + b; },
          nb::is_operator())
      .def(
          "__radd__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return b + a; },
          nb::is_operator())
      .def(
          "__sub__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return a - b; },
          nb::is_operator())
      .def(
          "__rsub__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return b - a; },
          nb::is_operator())
      .def(
          "__mul__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return a * b; },
          nb::is_operator())
      .def(
          "__rmul__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return b * a; },
          nb::is_operator())
      .def(
          "__truediv__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return a / b; },
          nb::is_operator())
      .def(
          "__rtruediv__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return b / a; },
          nb::is_operator())
      .def(
          "__floordiv__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return a / b; },
          nb::is_operator())
      .def(
          "__mod__",
          [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return og::mod(a, b); },
          nb::is_operator());
  nb::implicitly_convertible<origami::graphs::index_t, og::scalar_expr_t>();

  g.def("sym", &og::sym, "name"_a, "A free problem dimension, bound later by an env.");
  g.def("problem",
        &og::problem_sym,
        "name"_a,
        "A problem dimension: M, N, K, an element width. Resolved when a "
        "specification is instantiated.");
  g.def("config",
        &og::config_sym,
        "name"_a,
        "A tuning parameter: BM, BN, an occupancy. What varies across the "
        "candidates derived from one specification.");
  g.def("ceil_div", &og::ceil_div, "a"_a, "b"_a, "Rounding-up division over expressions.");
  g.def("floor_div",
        nb::overload_cast<const og::scalar_expr_t&, const og::scalar_expr_t&>(&og::floor_div),
        "a"_a,
        "b"_a,
        "Rounding-down division over expressions.");
  g.def("mod",
        nb::overload_cast<const og::scalar_expr_t&, const og::scalar_expr_t&>(&og::mod),
        "a"_a,
        "b"_a,
        "Euclidean modulo over expressions — the other half of a tile decode.");

  // ─── cost expressions ───────────────────────────────────────────────

  nb::class_<og::cost_expr_t>(g,
                              "cost_expr_t",
                              "A deferred real number: a duration, a rate, or a byte count. "
                              "Sibling to scalar_expr_t rather than a generalisation of it, so "
                              "that index arithmetic stays exact.")
      .def(nb::init<double>(), "value"_a)
      .def(nb::init<og::scalar_expr_t>(), "expr"_a)
      .def("eval",
           &og::cost_expr_t::eval,
           "ctx"_a,
           "Resolve against problem, config, hardware and runtime bindings.")
      .def("str", &og::cost_expr_t::str)
      .def("__repr__", &og::cost_expr_t::str)
      .def(
          "__add__",
          [](const og::cost_expr_t& a, const og::cost_expr_t& b) { return a + b; },
          nb::is_operator())
      .def(
          "__radd__",
          [](const og::cost_expr_t& a, const og::cost_expr_t& b) { return b + a; },
          nb::is_operator())
      .def(
          "__sub__",
          [](const og::cost_expr_t& a, const og::cost_expr_t& b) { return a - b; },
          nb::is_operator())
      .def(
          "__rsub__",
          [](const og::cost_expr_t& a, const og::cost_expr_t& b) { return b - a; },
          nb::is_operator())
      .def(
          "__mul__",
          [](const og::cost_expr_t& a, const og::cost_expr_t& b) { return a * b; },
          nb::is_operator())
      .def(
          "__rmul__",
          [](const og::cost_expr_t& a, const og::cost_expr_t& b) { return b * a; },
          nb::is_operator())
      .def(
          "__truediv__",
          [](const og::cost_expr_t& a, const og::cost_expr_t& b) { return a / b; },
          nb::is_operator())
      .def(
          "__rtruediv__",
          [](const og::cost_expr_t& a, const og::cost_expr_t& b) { return b / a; },
          nb::is_operator());
  nb::implicitly_convertible<double, og::cost_expr_t>();
  nb::implicitly_convertible<origami::graphs::index_t, og::cost_expr_t>();
  nb::implicitly_convertible<og::scalar_expr_t, og::cost_expr_t>();

  nb::class_<og::cost_context_t>(
      g, "cost_context_t", "Everything a cost expression can be resolved against.")
      .def(nb::init<>())
      .def_rw("problem", &og::cost_context_t::problem)
      .def_rw("config", &og::cost_context_t::config)
      .def_rw("hardware", &og::cost_context_t::hardware)
      .def_rw("runtime", &og::cost_context_t::runtime);

  g.def("hardware",
        &og::hardware_sym,
        "name"_a,
        "A machine characteristic. Available only in cost expressions, because "
        "a graph is device-independent and hardware arrives at ranking.");
  g.def("runtime",
        &og::runtime_sym,
        "name"_a,
        "Live scheduler state, such as active_cus. The reason a cost is an "
        "expression rather than a number: the same node is worth a different "
        "amount depending on what else is resident.");

  // Both arities are bound, integer first. Two integer expressions stay
  // integer, which keeps a clipped tile bound exact; anything touching a real
  // becomes a cost expression.
  g.def("minimum",
        nb::overload_cast<const og::scalar_expr_t&, const og::scalar_expr_t&>(&og::minimum),
        "a"_a,
        "b"_a,
        "Exact integer min — how a boundary tile is clipped.");
  g.def("minimum",
        nb::overload_cast<const og::cost_expr_t&, const og::cost_expr_t&>(&og::minimum),
        "a"_a,
        "b"_a,
        "Real-valued min.");
  g.def("maximum",
        nb::overload_cast<const og::scalar_expr_t&, const og::scalar_expr_t&>(&og::maximum),
        "a"_a,
        "b"_a,
        "Exact integer max.");
  g.def("maximum",
        nb::overload_cast<const og::cost_expr_t&, const og::cost_expr_t&>(&og::maximum),
        "a"_a,
        "b"_a,
        "Real-valued max — how a roofline is written.");

  nb::class_<og::allocation_t>(g, "allocation_t", "A logical buffer.")
      .def(
          "__init__",
          [](og::allocation_t* self, std::string name, std::vector<og::scalar_expr_t> shape) {
            new (self) og::allocation_t(std::move(name), std::move(shape));
          },
          "name"_a,
          "shape"_a,
          "Shape entries are element counts, and may be symbolic — a specification "
          "declares [M, K] and learns the sizes when it is instantiated. There is "
          "deliberately no dtype: the derivation counts elements, and bytes are the "
          "cost layer's business.")
      .def_prop_ro("name", &og::allocation_t::name)
      .def("__repr__",
           [](const og::allocation_t& a) { return "allocation_t('" + a.name() + "')"; });

  // index_set_t is a variant, so the two alternatives are bound and nanobind
  // maps the variant itself. That makes an access pattern inspectable from
  // Python rather than an opaque handle.
  nb::class_<og::range_t>(g, "range_t", "An arithmetic progression of indices.")
      .def(nb::init<>())
      .def_ro("start", &og::range_t::start)
      .def_ro("step", &og::range_t::step)
      .def_ro("count", &og::range_t::count);

  nb::class_<og::finite_set_t>(g, "finite_set_t", "An explicit, sorted, deduplicated index list.")
      .def(nb::init<>())
      .def_static("of", &og::finite_set_t::of, "values"_a)
      .def("indices", &og::finite_set_t::indices)
      .def("size", &og::finite_set_t::size)
      .def("empty", &og::finite_set_t::empty);

  nb::class_<og::access_pattern_t>(g, "access_pattern_t", "Which indices a workgroup touches.");

  g.def("contiguous",
        &og::contiguous,
        "block"_a,
        "base"_a = og::scalar_expr_t{origami::graphs::index_t{0}},
        "Each workgroup gets its own dense block of `block` indices.");
  g.def("strided",
        &og::strided,
        "block"_a,
        "stride"_a,
        "base"_a = og::scalar_expr_t{origami::graphs::index_t{0}},
        "Blocks of `block` indices spaced `stride` apart, so they overlap when stride < block.");
  g.def("rows",
        &og::rows,
        "block_rows"_a,
        "row_len"_a,
        "base"_a = og::scalar_expr_t{origami::graphs::index_t{0}},
        "A flattened band of whole rows.");
  g.def("streamed",
        &og::streamed,
        "block"_a,
        "num_iters"_a,
        "base"_a = og::scalar_expr_t{origami::graphs::index_t{0}},
        "Split a block across a workgroup's iterations, one chunk per iteration.");
  g.def("index_offsets",
        &og::index_offsets,
        "fn"_a,
        "Wrap fn(wg, it, ctx) -> list[int] as an index map, where ctx.problem and "
        "ctx.config are the two namespaces. The escape hatch for access with no "
        "closed form, and the way to index a table a kernel model already "
        "computed. Called once per workgroup-iteration inside a quadratic "
        "derivation and its result materialised, so prefer a closed-form builder "
        "where one exists.");

  // read and write are function templates over the pattern callable, so they are
  // instantiated here at index_fn_t rather than taken by address.
  g.def(
      "read",
      [](og::allocation_t allocation, og::index_fn_t indices, std::string name) {
        return og::read(std::move(allocation), std::move(indices), std::move(name));
      },
      "allocation"_a,
      "pattern"_a,
      "name"_a = "",
      "Declare a read access.");
  g.def(
      "write",
      [](og::allocation_t allocation, og::index_fn_t indices, std::string name) {
        return og::write(std::move(allocation), std::move(indices), std::move(name));
      },
      "allocation"_a,
      "pattern"_a,
      "name"_a = "",
      "Declare a write access.");

  // ─── operations and graphs ──────────────────────────────────────────

  // The grid and the pipeline depth take a whole expression, not just an int:
  // a specification's grid is `ceil_div(M, BM) * ceil_div(N, BN)` and is not
  // known until a problem and a config are bound.
  nb::class_<og::operation_t>(g, "operation_t", "A kernel launch: a grid of workgroups.")
      .def(
          "__init__",
          [](og::operation_t* self, std::string name, og::scalar_expr_t num_workgroups) {
            new (self) og::operation_t(std::move(name), std::move(num_workgroups));
          },
          "name"_a,
          "num_workgroups"_a)
      .def_rw("name", &og::operation_t::name)
      .def_rw("access_patterns", &og::operation_t::access_patterns)
      .def_prop_rw(
          "num_iters",
          [](const og::operation_t& op) { return op.num_iters; },
          [](og::operation_t& op, og::scalar_expr_t n) { op.num_iters = std::move(n); },
          "Internal pipeline depth; at least 1.")
      .def_rw("node_cost",
              &og::operation_t::node_cost,
              "fn(node) -> cost_expr_t, giving the deferred price of one node, in "
              "cycles. Every node derived from this operation uses the same "
              "function, and it receives the node, so a boundary workgroup can be "
              "charged for the tile it actually owns. See hardware_params for "
              "which hardware-scope names are per-cycle and which are "
              "per-second.");

  nb::class_<og::wg_graph_t>(g, "wg_graph_t", "The interface both graph backends satisfy.")
      .def("name", &og::wg_graph_t::name)
      .def("num_operations", &og::wg_graph_t::num_operations)
      .def("op_name", &og::wg_graph_t::op_name, "op"_a)
      .def("op_index", &og::wg_graph_t::op_index, "name"_a)
      .def("nodes", &og::wg_graph_t::nodes)
      .def("num_nodes", &og::wg_graph_t::num_nodes)
      .def("edges", &og::wg_graph_t::edges)
      .def("label", &og::wg_graph_t::label, "node"_a)
      .def("summary", &og::wg_graph_t::summary)
      .def("wg_count", &og::wg_graph_t::wg_count, "op"_a)
      .def("iter_count", &og::wg_graph_t::iter_count, "op"_a)
      .def("successor_edges", &og::wg_graph_t::successor_edges, "node"_a)
      .def("predecessor_edges", &og::wg_graph_t::predecessor_edges, "node"_a)
      .def("set_cost",
           &og::wg_graph_t::set_cost,
           "cost"_a,
           "Attach the cost model that prices this graph's nodes.")
      .def("cost", &og::wg_graph_t::cost_ptr, "The attached cost model, or None.");

  nb::class_<og::graph_t, og::wg_graph_t>(
      g, "graph_t", "Edges derived from access patterns: write-to-read intersection.")
      .def(
          "__init__",
          [](og::graph_t* self,
             std::vector<og::operation_t> ops,
             origami::graphs::env_t dims,
             std::string name) {
            new (self) og::graph_t(std::move(ops), std::move(dims), std::move(name));
          },
          "operations"_a,
          "dims"_a = origami::graphs::env_t{},
          "name"_a = "")
      .def("dims", &og::graph_t::dims, "Bindings for the symbolic dimensions.")
      .def("problem", &og::graph_t::problem, "The problem this graph was instantiated for.")
      .def("config", &og::graph_t::config, "The configuration this graph was instantiated for.")
      .def("to_free", &og::graph_t::to_free, "Snapshot as an explicit-edge graph.");

  // ─── specifications ─────────────────────────────────────────────────

  nb::class_<og::graph_spec_t>(g,
                               "graph_spec_t",
                               "A reusable, hardware-free description of a kernel. One of these "
                               "instantiates to many graphs, which is what an autotuner wants.")
      .def(nb::init<>())
      .def(nb::init<std::vector<og::operation_t>>(), "operations"_a)
      .def("operations", &og::graph_spec_t::operations)
      .def("add", &og::graph_spec_t::add, "operation"_a, nb::rv_policy::reference_internal)
      .def("__len__", &og::graph_spec_t::size)
      .def("instantiate",
           &og::graph_spec_t::instantiate,
           "problem"_a,
           "config"_a,
           "name"_a = "",
           "Bind a problem and a configuration and derive the graph. The result is "
           "unpriced even when the operations carry costs, because pricing needs "
           "hardware; rank_graphs supplies it.");

  nb::class_<og::free_graph_t, og::wg_graph_t>(
      g, "free_graph_t", "Edges stated outright, for topologies no access pattern describes.")
      .def(nb::init<std::string>(), "name"_a = "")
      .def("add_op",
           &og::free_graph_t::add_op,
           "name"_a,
           "num_workgroups"_a,
           "num_iters"_a = 1,
           "Declare an operation; returns its index.")
      .def("add_edge",
           nb::overload_cast<const og::wg_node_t&,
                             const og::wg_node_t&,
                             std::string,
                             origami::graphs::index_t>(&og::free_graph_t::add_edge),
           "src"_a,
           "dst"_a,
           "allocation"_a = "",
           "weight"_a     = 1);

  // ─── analysis ───────────────────────────────────────────────────────

  nb::class_<og::critical_path_t>(g, "critical_path_t", "Longest path and the chain achieving it.")
      .def_ro("makespan", &og::critical_path_t::makespan)
      .def_ro("path", &og::critical_path_t::path);

  g.def("topological_order", &og::topological_order, "graph"_a, "Every node, predecessors first.");
  g.def("is_acyclic", &og::is_acyclic, "graph"_a, "Whether a topological order exists.");
  g.def(
      "critical_path",
      [](const og::wg_graph_t& graph, std::optional<og::node_cost_fn_t> node_cost) {
        return node_cost ? og::critical_path(graph, *node_cost) : og::critical_path(graph);
      },
      "graph"_a,
      "node_cost"_a = nb::none(),
      "Longest path under unbounded resources; unit node cost unless given one.");

  // ─── simulate(): the cycle-based scheduler ──────────────────────────
  //
  // A runtime_t here is a policy only -- priority, placement, order, an
  // optional gate -- and carries no lane count or duration of its own; those
  // are simulate()'s options and the cost model's job, respectively.

  nb::class_<og::schedule_t>(
      g, "schedule_t", "A cycle-based schedule: start, duration and lane per atom.")
      .def(nb::init<>())
      .def_rw("runtime", &og::schedule_t::runtime)
      .def_rw("lanes", &og::schedule_t::lanes)
      .def_rw("order", &og::schedule_t::order)
      .def("place",
           &og::schedule_t::place,
           "node"_a,
           "start"_a,
           "duration"_a,
           "lane"_a,
           "Record one atom's start, duration and lane.")
      .def(
          "start",
          [](const og::schedule_t& s, const og::wg_node_t& n) { return s.start.at(n); },
          "node"_a)
      .def(
          "duration",
          [](const og::schedule_t& s, const og::wg_node_t& n) { return s.duration.at(n); },
          "node"_a)
      .def(
          "lane",
          [](const og::schedule_t& s, const og::wg_node_t& n) { return s.lane.at(n); },
          "node"_a)
      .def("finish", &og::schedule_t::finish, "node"_a)
      .def("makespan", &og::schedule_t::makespan)
      .def("num_lanes", &og::schedule_t::num_lanes)
      .def("busy_by_op", &og::schedule_t::busy_by_op)
      .def("utilization", &og::schedule_t::utilization)
      .def("summary", &og::schedule_t::summary);

  nb::class_<og::simulate_options_t>(g, "simulate_options_t", "Machine and costing knobs.")
      .def(nb::init<>())
      .def_rw("lanes", &og::simulate_options_t::lanes)
      .def_rw("serialize_wg_iters", &og::simulate_options_t::serialize_wg_iters)
      .def_rw("reprice_on_dispatch", &og::simulate_options_t::reprice_on_dispatch)
      .def_rw("skip_prefix", &og::simulate_options_t::skip_prefix);

  nb::class_<og::cost_model_t, py_cost_model_t>(
      g,
      "cost_model_t",
      "Anything that can price a node and a hop, in cycles, for simulate(). Subclass this to "
      "price a machine the shipped models do not describe: override node_cycles and "
      "edge_cycles, and node_cycles_at if contention matters.")
      .def(nb::init<>())
      .def("node_cycles", &og::cost_model_t::node_cycles, "node"_a)
      .def("edge_cycles", &og::cost_model_t::edge_cycles, "edge"_a)
      .def("node_cycles_at", &og::cost_model_t::node_cycles_at, "node"_a, "active_cus"_a);

  nb::class_<og::unit_cost_t, og::cost_model_t>(
      g,
      "unit_cost_t",
      "One cycle per node, nothing per hop -- turns simulate() into a scheduler that counts "
      "workgroups rather than pricing them.")
      .def(nb::init<>());

  // ─── placement ──────────────────────────────────────────────────────

  nb::class_<og::placement_context_t>(
      g, "placement_context_t", "What a placement is allowed to see.")
      .def_prop_ro("node", [](const og::placement_context_t& c) { return c.node; })
      .def_prop_ro("launch_index", [](const og::placement_context_t& c) { return c.launch_index; })
      .def_prop_ro(
          "free_at",
          [](const og::placement_context_t& c) { return c.free_at; },
          "Per-lane free time, in cycles. Its length is the machine's width.");

  nb::class_<og::placement_t, py_placement_t>(
      g,
      "placement_t",
      "Chooses which lane a dispatched node runs on. Deliberately not the scheduler: a "
      "placement cannot reorder work or read a cost model. Subclass this and override "
      "choose(ctx) to express a lane rule the library does not ship, such as pinning a "
      "workgroup to one lane for a whole launch.")
      .def(nb::init<>())
      .def("name", &og::placement_t::name)
      .def("choose", &og::placement_t::choose, "ctx"_a);

  nb::class_<og::earliest_free_placement_t, og::placement_t>(
      g, "earliest_free_placement_t", "The lane that frees soonest, ties going to the lower index.")
      .def(nb::init<>());

  nb::class_<og::pooled_placement_t, og::placement_t>(
      g,
      "pooled_placement_t",
      "Earliest-free, but restricted to an operation's pool of lanes: how a concurrent comm "
      "and GEMM launch partitions the compute units so the two genuinely overlap.")
      .def(nb::init<og::lane_pool_t, const og::wg_graph_t&, int>(),
           "pool"_a,
           "graph"_a,
           "lane_count"_a);

  nb::class_<og::xcd_placement_t, og::placement_t>(
      g,
      "xcd_placement_t",
      "Chiplet dispatch: a die by launch index, then the compute unit in that die which frees "
      "soonest.")
      .def(nb::init<int, int>(), "num_xcds"_a, "cus_per_xcd"_a)
      .def("lane_count", &og::xcd_placement_t::lane_count);

  nb::enum_<og::queue_order_t>(
      g, "queue_order_t", "Whether readiness or the policy's rank dominates dispatch order.")
      .value("ready_first", og::queue_order_t::ready_first)
      .value("rank_first", og::queue_order_t::rank_first);

  nb::class_<og::sim_state_t>(
      g, "sim_state_t", "What a gate may see: enough to sequence operations, no more.")
      .def_prop_ro("retired_by_op", [](const og::sim_state_t& s) { return s.retired_by_op; })
      .def_prop_ro("in_flight_by_op", [](const og::sim_state_t& s) { return s.in_flight_by_op; })
      .def_prop_ro("total_by_op", [](const og::sim_state_t& s) { return s.total_by_op; });

  nb::class_<og::runtime_t, py_runtime_t>(
      g,
      "runtime_t",
      "A scheduling policy: a priority ranking, a placement, and an optional gate. Says "
      "nothing about cost; applying a runtime and then a cost model is what simulate() does. "
      "Subclass this to dispatch the way some kernel actually does: override priority(graph) "
      "and placement(), plus name(), order() and gated() as needed.")
      .def(nb::init<>())
      .def("name", &og::runtime_t::name)
      .def("priority", &og::runtime_t::priority, "graph"_a)
      .def("placement", &og::runtime_t::placement, nb::rv_policy::reference_internal)
      .def("order", &og::runtime_t::order)
      .def("gated", &og::runtime_t::gated, "node"_a, "state"_a);

  // The four shipped policies over runtime_t.
  nb::class_<og::breadth_first_runtime_t, og::runtime_t>(
      g,
      "breadth_first_runtime_t",
      "Whole operation, sync, next operation: the no-overlap baseline. The only "
      "policy that needs a gate, since independent operations are both ready at "
      "once and no priority alone can keep them from overlapping.")
      .def(nb::init<>());

  nb::class_<og::asap_runtime_t, og::runtime_t>(
      g,
      "asap_runtime_t",
      "Producer-greedy: ready work in canonical order, earliest-ready first. This is also "
      "what runtime_kind_t.roofline names: the two differ in where a node's duration comes "
      "from, not in the order work is issued.")
      .def(nb::init<>());

  nb::class_<og::depth_first_runtime_t, og::runtime_t>(
      g, "depth_first_runtime_t", "Chain-greedy: dive down one producer-consumer chain.")
      .def(nb::init<>());

  nb::class_<og::xcd_runtime_t, og::runtime_t>(
      g,
      "xcd_runtime_t",
      "Chiplet dispatch in launch order: rank-first rather than ready-first, because a real "
      "dispatcher issues workgroups in grid order rather than picking whichever is ready. "
      "Requires simulate_options_t.lanes == lane_count().")
      .def(nb::init<int, int>(), "num_xcds"_a = 8, "cus_per_xcd"_a = 38)
      .def("lane_count", &og::xcd_runtime_t::lane_count);

  nb::class_<og::runtime_with_placement_t, og::runtime_t>(
      g,
      "runtime_with_placement_t",
      "A policy with its placement replaced, everything else -- name, priority, order, gate "
      "-- forwarded from the base policy. The library's own answer to 'take a policy, "
      "override its placement', e.g. to partition lanes between a concurrent comm and GEMM "
      "launch with pooled_placement_t while keeping the base policy's issuing order.")
      .def(nb::init<const og::runtime_t&, const og::placement_t&>(),
           "base"_a,
           "placement"_a,
           nb::keep_alive<1, 2>(),
           nb::keep_alive<1, 3>());

  g.def("simulate",
        &og::simulate,
        "graph"_a,
        "runtime"_a,
        "cost"_a,
        "options"_a = og::simulate_options_t{},
        "Schedule a graph under a policy, pricing it with a cost model.");

  // ─── the seconds boundary ───────────────────────────────────────────
  //
  // Everything above is in cycles. clock_t/fixed_clock_t/timed_schedule_t/
  // to_seconds are the only names that turn a *schedule's* cycles into a
  // duration -- the roofline arm bound further below also names a clock
  // (compute_clock_ghz) and data-sheet quantities in seconds (hbm_bw, link_bw,
  // link_latency), because HBM and link are continuous rates in their own
  // clock domain and need one to become cycles in the first place. The point
  // stands either way: a frequency moves with DVFS, with the part and with
  // what else is resident, so a model that divides by one without naming it
  // has folded a runtime condition into a static answer.

  nb::class_<og::clock_t>(g, "clock_t", "A clock, so seconds are a boundary concern.")
      .def("ghz", &og::clock_t::ghz);

  nb::class_<og::fixed_clock_t, og::clock_t>(g, "fixed_clock_t", "A clock that does not vary.")
      .def(nb::init<double>(), "ghz"_a);

  nb::class_<og::timed_schedule_t>(
      g, "timed_schedule_t", "A schedule in seconds: the same layout, converted for display.")
      .def(nb::init<>())
      .def_rw("runtime", &og::timed_schedule_t::runtime)
      .def_rw("units", &og::timed_schedule_t::units)
      .def_rw("order", &og::timed_schedule_t::order)
      .def(
          "start",
          [](const og::timed_schedule_t& s, const og::wg_node_t& n) { return s.start.at(n); },
          "node"_a)
      .def(
          "duration",
          [](const og::timed_schedule_t& s, const og::wg_node_t& n) { return s.duration.at(n); },
          "node"_a)
      .def(
          "lane",
          [](const og::timed_schedule_t& s, const og::wg_node_t& n) { return s.lane.at(n); },
          "node"_a)
      .def("finish", &og::timed_schedule_t::finish, "node"_a)
      .def("makespan", &og::timed_schedule_t::makespan)
      .def("summary", &og::timed_schedule_t::summary);

  g.def("to_seconds",
        &og::to_seconds,
        "schedule"_a,
        "clock"_a,
        "scale"_a = 1.0,
        "units"_a = "s",
        "Convert a schedule from cycles to seconds -- the only place a *schedule's* cycles "
        "become a duration; pass scale=1e6 and units='us' for microseconds.");

  // ─── cost models ────────────────────────────────────────────────────

  nb::enum_<og::cost_kind_t>(g, "cost_kind_t", "Which arm prices an operation.")
      .value("comm", og::cost_kind_t::comm)
      .value("gemm", og::cost_kind_t::gemm)
      .value("roofline", og::cost_kind_t::roofline)
      .value("custom", og::cost_kind_t::custom);

  nb::class_<og::roofline_hardware_t>(
      g, "roofline_hardware_t", "Headline rates for the roofline arm; MI300X-ish by default.")
      .def(nb::init<>())
      .def_rw("peak_flops_per_cycle",
              &og::roofline_hardware_t::peak_flops_per_cycle,
              "FLOP/cycle, aggregate across the part -- clock-free, since matrix "
              "throughput is CUs x FLOP/cycle/CU x clock, so the compute roof "
              "using this directly needs no clock and no conversion.")
      .def_rw("hbm_bw",
              &og::roofline_hardware_t::hbm_bw,
              "Bytes/s to local memory, a data-sheet figure. Scaled into cycles by "
              "compute_clock_ghz, since HBM is a continuous rate in its own clock "
              "domain rather than a per-cycle property of the compute engine.")
      .def_rw("link_bw",
              &og::roofline_hardware_t::link_bw,
              "Bytes/s off-GPU, a data-sheet figure. Scaled into cycles by "
              "compute_clock_ghz, same as hbm_bw.")
      .def_rw("link_latency",
              &og::roofline_hardware_t::link_latency,
              "Fixed per-hop latency, in seconds. Scaled into cycles by "
              "compute_clock_ghz.")
      .def_rw("compute_clock_ghz",
              &og::roofline_hardware_t::compute_clock_ghz,
              "Engine clock, GHz. Converts hbm_bw, link_bw and link_latency from "
              "seconds into cycles; has no bearing on the compute roof, since "
              "peak_flops_per_cycle is already clock-free. Must be positive.")
      .def("peak_flops",
           &og::roofline_hardware_t::peak_flops,
           "Spec-sheet FLOP/s at compute_clock_ghz; derived from peak_flops_per_cycle, not "
           "independently settable.")
      .def("ridge_intensity", &og::roofline_hardware_t::ridge_intensity);

  nb::class_<og::roofline_spec_t>(g, "roofline_spec_t", "Per-workgroup work on the three axes.")
      .def(nb::init<>())
      .def_rw("flops", &og::roofline_spec_t::flops)
      .def_rw("hbm_bytes", &og::roofline_spec_t::hbm_bytes)
      .def_rw("link_bytes", &og::roofline_spec_t::link_bytes)
      .def_static("gemm_tile",
                  &og::roofline_spec_t::gemm_tile,
                  "bm"_a,
                  "bn"_a,
                  "k"_a,
                  "dtype_bytes"_a = 2.0)
      .def_static("comm_step",
                  &og::roofline_spec_t::comm_step,
                  "elements"_a,
                  "dtype_bytes"_a = 2.0,
                  "fan"_a         = 1.0);

  g.def("roofline_cycles",
        &og::roofline_cycles,
        "spec"_a,
        "hardware"_a = og::roofline_hardware_t{},
        "The largest of the compute, HBM and link roofs, in cycles. Compute uses "
        "hardware.peak_flops_per_cycle directly; HBM and link are scaled from seconds by "
        "hardware.compute_clock_ghz.");
  g.def("roofline_bound",
        &og::roofline_bound,
        "spec"_a,
        "hardware"_a = og::roofline_hardware_t{},
        "Which roof the workgroup sits on.");

  nb::class_<og::comm_spec_t>(g, "comm_spec_t", "One workgroup of a communication operation.")
      .def(
          "__init__",
          [](og::comm_spec_t* self,
             std::vector<std::string> work_graph,
             std::size_t wg_tile_bytes,
             int num_wgs,
             double bw_per_wg,
             int peer,
             std::optional<oc::primitive_t> primitive) {
            new (self) og::comm_spec_t{};
            self->work_graph    = work_graph_from_names(work_graph, peer);
            self->wg_tile_bytes = wg_tile_bytes;
            self->num_wgs       = num_wgs;
            self->bw_per_wg     = bw_per_wg;
            self->primitive     = primitive;
          },
          "work_graph"_a,
          "wg_tile_bytes"_a,
          "num_wgs"_a   = 1,
          "bw_per_wg"_a = 0.0,
          "peer"_a      = 1,
          "primitive"_a = nb::none(),
          "work_graph is a list of primitive names: load, store, push, pull, reduce, signal, "
          "wait. The peer-taking ones all use `peer`.")
      .def_rw("wg_tile_bytes", &og::comm_spec_t::wg_tile_bytes)
      .def_rw("num_wgs", &og::comm_spec_t::num_wgs)
      .def_rw("bw_per_wg", &og::comm_spec_t::bw_per_wg)
      .def_rw("tile", &og::comm_spec_t::tile)
      .def_rw("primitive", &og::comm_spec_t::primitive);

  g.def(
      "comm_cycles",
      [](const og::comm_spec_t& spec, const oc::system_t& system, std::optional<int> active_cus) {
        return og::comm_cycles(spec, system, active_cus);
      },
      "spec"_a,
      "system"_a,
      "active_cus"_a = nb::none(),
      "Per-workgroup-tile latency from origami.comm's calibrated model, in cycles.");

  nb::class_<og::op_cost_t>(g, "op_cost_t", "How one operation is priced.")
      .def_static("from_roofline", &og::op_cost_t::from_roofline, "spec"_a)
      .def_static("from_comm", &og::op_cost_t::from_comm, "spec"_a)
      .def_static(
          "from_custom", &og::op_cost_t::from_custom, "fn"_a, "fn(node, active_cus) -> cycles.")
      .def("kind", &og::op_cost_t::kind);

  nb::class_<og::cost_settings_t>(
      g,
      "cost_settings_t",
      "Machine description shared by a table. If a cost_table_t uses both comm and roofline "
      "entries, comm_system's clock must equal roofline_hardware.compute_clock_ghz.")
      .def(nb::init<>())
      .def_rw("roofline_hardware", &og::cost_settings_t::roofline_hardware)
      .def_rw("comm_system",
              &og::cost_settings_t::comm_system,
              "Required by the comm arm; build one with origami.comm.make_system.")
      .def_rw("default_cycles", &og::cost_settings_t::default_cycles);

  nb::class_<og::cost_table_t, og::cost_model_t>(
      g, "cost_table_t", "A cost model assembled from one entry per operation.")
      .def(nb::init<const og::wg_graph_t&, og::cost_settings_t>(),
           "graph"_a,
           "settings"_a = og::cost_settings_t{},
           nb::keep_alive<1, 2>())
      .def("set",
           &og::cost_table_t::set,
           "op_name"_a,
           "cost"_a,
           nb::rv_policy::reference_internal,
           "Price an operation; returns the table so calls chain.")
      .def("has", &og::cost_table_t::has, "op"_a)
      .def("kind_of", &og::cost_table_t::kind_of, "op"_a);

  nb::class_<og::expr_cost_t, og::cost_model_t>(
      g,
      "expr_cost_t",
      "Prices a graph from its operations' deferred cost expressions, so a "
      "symbolic cost satisfies the interface every runtime already speaks.")
      .def(nb::init<const og::graph_t&, og::param_map_t>(), "graph"_a, "hardware"_a)
      .def_static("priced",
                  &og::expr_cost_t::priced,
                  "graph"_a,
                  "True when at least one operation carries a cost function.");

  g.def("attach_expr_cost",
        &og::attach_expr_cost,
        "graph"_a,
        "hardware"_a,
        "Price a specification-derived graph for one machine. Returns False when "
        "no operation carries a cost function.");

  g.def("hardware_params",
        nb::overload_cast<const origami::comm::hardware_t&>(&og::hardware_params),
        "hardware"_a,
        "Hardware-scope bindings derived from a machine description. Publishes "
        "both per-cycle rates (hbm_read_bw, tcp_bw, valu_rate, ...) and their "
        "per-second convenience forms (peak_flops_per_cu, hbm_bytes_per_second, "
        "hbm_read_bytes_per_second, hbm_write_bytes_per_second, "
        "tcp_bytes_per_second_per_cu, l2_bytes_per_second_per_cu, "
        "mall_bytes_per_second) -- note hbm_bytes_per_second is the unqualified "
        "spelling of 'HBM bandwidth', not the default; hbm_read_bw is. Dividing a "
        "cycle-valued quantity by one of the seven per-second names silently "
        "lands the result back in seconds, with no clock symbol anywhere in the "
        "expression to catch a reviewer's eye, so prefer the per-cycle sibling. "
        "The result is an ordinary map: add any name your cost expressions "
        "reference that the part itself cannot report, such as a peak "
        "matrix-core rate, which depends on the instruction shape a kernel "
        "chose -- but bind it per cycle, like hbm_read_bw, not per second like "
        "hbm_bytes_per_second.");

  // ─── ranking ────────────────────────────────────────────────────────

  nb::class_<og::xcd_options_t>(
      g, "xcd_options_t", "Chiplet geometry, used only by runtime_kind_t.xcd.")
      .def(nb::init<>())
      .def_rw("num_xcds", &og::xcd_options_t::num_xcds)
      .def_rw("cus_per_xcd", &og::xcd_options_t::cus_per_xcd)
      .def_rw("skip_prefix", &og::xcd_options_t::skip_prefix);

  nb::enum_<og::runtime_kind_t>(
      g,
      "runtime_kind_t",
      "Which scheduling policy a candidate uses, and how it is priced: these name a "
      "(policy, cost source) pair, which is why asap and roofline are the same policy.")
      .value("breadth_first", og::runtime_kind_t::breadth_first)
      .value("asap", og::runtime_kind_t::asap)
      .value("depth_first", og::runtime_kind_t::depth_first)
      .value("roofline", og::runtime_kind_t::roofline)
      .value("event_driven", og::runtime_kind_t::event_driven)
      .value("xcd", og::runtime_kind_t::xcd);

  g.def("needs_cost_model",
        &og::needs_cost_model,
        "kind"_a,
        "Whether the kind needs a real cost model. breadth_first/asap/depth_first run "
        "through simulate() charging a flat wg_duration cycles per node, so they schedule "
        "without one; the other three price every node from a cost model and fail without it.");

  nb::class_<og::graph_config_t>(g, "graph_config_t", "One candidate schedule.")
      .def(nb::init<>())
      .def_rw("runtime", &og::graph_config_t::runtime)
      .def_rw("lanes", &og::graph_config_t::lanes)
      .def_rw("serialize_wg_iters", &og::graph_config_t::serialize_wg_iters)
      .def_rw("lane_pool", &og::graph_config_t::lane_pool)
      .def_rw("wg_duration",
              &og::graph_config_t::wg_duration,
              "Cycles charged per node by the kinds that use no cost model.")
      .def_rw("xcd", &og::graph_config_t::xcd)
      .def_rw("runtime_override",
              &og::graph_config_t::runtime_override,
              "A policy to use in place of the one `runtime` names. Unlike the seconds-era "
              "override it replaces, the other fields still apply: a runtime_t is a policy "
              "and nothing more, so lanes, serialize_wg_iters and the cost model still do "
              "their jobs.")
      .def_rw("name", &og::graph_config_t::name);

  nb::class_<og::prediction_result_t>(
      g, "prediction_result_t", "A candidate's predicted cost, in cycles.")
      .def_ro("latency", &og::prediction_result_t::latency)
      .def_ro("config", &og::prediction_result_t::config)
      .def_ro("graph_index", &og::prediction_result_t::graph_index)
      .def_ro("graph_name", &og::prediction_result_t::graph_name)
      .def_ro("rejected", &og::prediction_result_t::rejected)
      .def_ro("rejection", &og::prediction_result_t::rejection);

  g.attr("REJECTED_LATENCY") = og::kRejectedLatency;

  g.def("rejection_reason",
        &og::rejection_reason,
        "graph"_a,
        "config"_a,
        "Why a candidate cannot be scheduled; empty when it can.");
  g.def("predict_latency",
        &og::predict_latency,
        "graph"_a,
        "config"_a,
        "cost"_a = nb::none(),
        "Makespan for one graph under one candidate schedule, in cycles, or "
        "REJECTED_LATENCY when the candidate is infeasible.");

  g.def("rank_configs",
        &og::rank_configs,
        "graph"_a,
        "configs"_a,
        "cost"_a = nb::none(),
        "Rank candidate schedules for one graph, best first.");
  g.def("select_config", &og::select_config, "graph"_a, "configs"_a, "cost"_a = nb::none());
  g.def("select_topk_configs",
        &og::select_topk_configs,
        "graph"_a,
        "configs"_a,
        "topk"_a,
        "cost"_a = nb::none());

  g.def(
      "rank_graphs",
      [](const std::vector<og::wg_graph_t*>& graphs,
         const std::vector<og::graph_config_t>& configs,
         const og::cost_model_t* cost) {
        return og::rank_graphs(as_pointers(graphs), configs, cost);
      },
      "graphs"_a,
      "configs"_a,
      "cost"_a = nb::none(),
      "Rank the whole graph-by-schedule cross product, best first.");

  g.def(
      "rank_graphs",
      [](const std::vector<og::wg_graph_t*>& graphs,
         const std::vector<og::graph_config_t>& configs,
         const og::param_map_t& hardware,
         const og::cost_model_t* cost) {
        return og::rank_graphs(as_pointers(graphs), configs, hardware, cost);
      },
      "graphs"_a,
      "configs"_a,
      "hardware"_a,
      "cost"_a = nb::none(),
      "Rank against a machine. Any graph carrying specification-owned node costs "
      "is priced with these hardware bindings, and any scheduling resource left "
      "unset — lanes, chiplet geometry — is filled from them.");

  g.def(
      "rank_graphs",
      [](const std::vector<og::wg_graph_t*>& graphs,
         const std::vector<og::graph_config_t>& configs,
         const origami::comm::hardware_t& hardware,
         const og::cost_model_t* cost) {
        return og::rank_graphs(as_pointers(graphs), configs, hardware, cost);
      },
      "graphs"_a,
      "configs"_a,
      "hardware"_a,
      "cost"_a = nb::none(),
      "Rank against a machine description, converted with hardware_params.");
  g.def(
      "select_graph",
      [](const std::vector<og::wg_graph_t*>& graphs,
         const std::vector<og::graph_config_t>& configs,
         const og::cost_model_t* cost) {
        return og::select_graph(as_pointers(graphs), configs, cost);
      },
      "graphs"_a,
      "configs"_a,
      "cost"_a = nb::none());
  g.def(
      "select_topk_graphs",
      [](const std::vector<og::wg_graph_t*>& graphs,
         const std::vector<og::graph_config_t>& configs,
         std::size_t topk,
         const og::cost_model_t* cost) {
        return og::select_topk_graphs(as_pointers(graphs), configs, topk, cost);
      },
      "graphs"_a,
      "configs"_a,
      "topk"_a,
      "cost"_a = nb::none());

  // ─── trace ──────────────────────────────────────────────────────────

  nb::class_<og::trace_options_t>(g, "trace_options_t", "Presentation choices for a trace.")
      .def(nb::init<>())
      .def_rw("process_name", &og::trace_options_t::process_name)
      .def_rw("scale", &og::trace_options_t::scale)
      .def_rw("flow_edges", &og::trace_options_t::flow_edges)
      .def_rw("node_arguments", &og::trace_options_t::node_arguments);

  g.def("chrome_trace",
        nb::overload_cast<const og::wg_graph_t&,
                          const og::timed_schedule_t&,
                          const og::trace_options_t&>(&og::chrome_trace),
        "graph"_a,
        "schedule"_a,
        "options"_a = og::trace_options_t{},
        "Chrome Trace Event JSON, loadable in chrome://tracing or Perfetto. A viewer's axis "
        "is time, so the schedule to hand it is a to_seconds(s, clock, 1e6, 'us').");
  g.def("chrome_trace",
        nb::overload_cast<const og::wg_graph_t&, const og::schedule_t&, const og::trace_options_t&>(
            &og::chrome_trace),
        "graph"_a,
        "schedule"_a,
        "options"_a = og::trace_options_t{});

  g.def("write_chrome_trace",
        nb::overload_cast<const og::wg_graph_t&,
                          const og::timed_schedule_t&,
                          const std::string&,
                          const og::trace_options_t&>(&og::write_chrome_trace),
        "graph"_a,
        "schedule"_a,
        "path"_a,
        "options"_a = og::trace_options_t{});
  g.def("write_chrome_trace",
        nb::overload_cast<const og::wg_graph_t&,
                          const og::schedule_t&,
                          const std::string&,
                          const og::trace_options_t&>(&og::write_chrome_trace),
        "graph"_a,
        "schedule"_a,
        "path"_a,
        "options"_a = og::trace_options_t{});
}
