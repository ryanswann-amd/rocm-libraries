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
// `cost_model_t` and `cost_runtime_t` are subclassable from Python, through the
// trampolines below. The cost model is the seam where hardware enters and the
// runtime is the seam where a scheduling policy does; a caller who has a machine
// or a policy the library does not ship should not have to write C++ to say so.
// Iris is the first such caller: its layout kernel pins workgroups to lanes for
// the whole launch, which is not how any runtime here dispatches.

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

/** Lets a Python class price nodes and hops. */
struct py_cost_model_t : og::cost_model_t {
  NB_TRAMPOLINE(og::cost_model_t, 3);

  double node_cost(const og::wg_node_t& node) const override { NB_OVERRIDE_PURE(node_cost, node); }
  double edge_cost(const og::edge_t& edge) const override { NB_OVERRIDE_PURE(edge_cost, edge); }
  double node_cost_at(const og::wg_node_t& node, int active_cus) const override {
    NB_OVERRIDE(node_cost_at, node, active_cus);
  }
};

/** Lets a Python class be a scheduling policy. */
struct py_cost_runtime_t : og::cost_runtime_t {
  NB_TRAMPOLINE(og::cost_runtime_t, 2);

  /**
   * `name()` hands back a reference, and Python hands back a fresh string, so
   * the macro cannot be used here: it would cast into a temporary and return a
   * reference to it. Keeping the last answer in the trampoline gives the
   * reference something to point at for as long as any caller can hold it.
   */
  const std::string& name() const override {
    nb::detail::ticket nb_ticket(nb_trampoline, "name", true);
    name_ = nb::cast<std::string>(nb_trampoline.base().attr(nb_ticket.key)());
    return name_;
  }

  og::cost_schedule_t schedule(const og::wg_graph_t& graph) const override {
    NB_OVERRIDE_PURE(schedule, graph);
  }

 private:
  mutable std::string name_;
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

  // Bound with an implicit conversion from int, which is what makes
  // `contiguous(4)` work in Python the way `contiguous(4)` works in C++.
  nb::class_<og::scalar_expr_t>(
      g, "scalar_expr_t", "A non-negative integer, possibly symbolic in a problem dimension.")
      .def(nb::init<origami::graphs::index_t>(), "value"_a)
      .def("eval",
           nb::overload_cast<const origami::graphs::env_t&>(&og::scalar_expr_t::eval, nb::const_),
           "env"_a,
           "Resolve against dimension bindings.")
      .def("eval",
           nb::overload_cast<>(&og::scalar_expr_t::eval, nb::const_),
           "Resolve an expression that is already concrete.")
      .def("__add__", [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return a + b; })
      .def("__sub__", [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return a - b; })
      .def("__mul__", [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return a * b; })
      .def("__truediv__",
           [](const og::scalar_expr_t& a, const og::scalar_expr_t& b) { return a / b; });
  nb::implicitly_convertible<origami::graphs::index_t, og::scalar_expr_t>();

  g.def("sym", &og::sym, "name"_a, "A free problem dimension, bound later by an env.");
  g.def("ceil_div", &og::ceil_div, "a"_a, "b"_a, "Rounding-up division over expressions.");

  nb::class_<og::allocation_t>(g, "allocation_t", "A logical buffer.")
      .def(
          "__init__",
          [](og::allocation_t* self, std::string name, std::vector<origami::graphs::index_t> s) {
            new (self) og::allocation_t(std::move(name),
                                        std::vector<og::scalar_expr_t>(s.begin(), s.end()));
          },
          "name"_a,
          "shape"_a,
          "Shape entries are element counts. There is deliberately no dtype: the "
          "derivation counts elements, and bytes are the cost layer's business.")
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
  g.def("callable_set",
        &og::callable_set,
        "fn"_a,
        "Wrap fn(wg, it, env) -> list[int] as an index map. The escape hatch for "
        "access with no closed form, and the way to index a table a kernel model "
        "already computed. Called once per workgroup-iteration inside a quadratic "
        "derivation, so prefer a closed-form builder where one exists.");

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

  nb::class_<og::operation_t>(g, "operation_t", "A kernel launch: a grid of workgroups.")
      .def(
          "__init__",
          [](og::operation_t* self, std::string name, origami::graphs::index_t num_workgroups) {
            new (self) og::operation_t(std::move(name), og::scalar_expr_t{num_workgroups});
          },
          "name"_a,
          "num_workgroups"_a)
      .def_rw("name", &og::operation_t::name)
      .def_rw("access_patterns", &og::operation_t::access_patterns)
      .def_prop_rw(
          "num_iters",
          [](const og::operation_t& op) { return op.num_iters; },
          [](og::operation_t& op, origami::graphs::index_t n) {
            op.num_iters = og::scalar_expr_t{n};
          },
          "Internal pipeline depth; at least 1.");

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
      .def("to_free", &og::graph_t::to_free, "Snapshot as an explicit-edge graph.");

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

  // ─── integer-timestep runtimes ──────────────────────────────────────

  nb::class_<og::schedule_t>(g, "schedule_t", "An integer-timestep schedule.")
      .def_ro("runtime", &og::schedule_t::runtime)
      .def_ro("lanes", &og::schedule_t::lanes)
      .def_ro("wg_duration", &og::schedule_t::wg_duration)
      .def_ro("order", &og::schedule_t::order)
      .def(
          "start",
          [](const og::schedule_t& s, const og::wg_node_t& n) { return s.start.at(n); },
          "node"_a)
      .def("finish", &og::schedule_t::finish, "node"_a)
      .def("makespan", &og::schedule_t::makespan)
      .def("summary", &og::schedule_t::summary);

  nb::class_<og::runtime_t>(g, "runtime_t", "A scheduling policy over integer timesteps.")
      .def("schedule", &og::runtime_t::schedule, "graph"_a);

  nb::class_<og::breadth_first_runtime_t, og::runtime_t>(
      g, "breadth_first_runtime_t", "No overlap: one operation at a time. The baseline.")
      .def(nb::init<std::optional<int>, origami::graphs::index_t>(),
           "lanes"_a       = nb::none(),
           "wg_duration"_a = 1);
  nb::class_<og::asap_runtime_t, og::runtime_t>(
      g, "asap_runtime_t", "Producer-greedy: dispatch every ready atom.")
      .def(nb::init<std::optional<int>, origami::graphs::index_t>(),
           "lanes"_a       = nb::none(),
           "wg_duration"_a = 1);
  nb::class_<og::depth_first_runtime_t, og::runtime_t>(
      g,
      "depth_first_runtime_t",
      "Chain-greedy: follow a dependency chain before starting another.")
      .def(nb::init<std::optional<int>, origami::graphs::index_t>(),
           "lanes"_a       = nb::none(),
           "wg_duration"_a = 1);

  // ─── cost models ────────────────────────────────────────────────────

  nb::enum_<og::cost_kind_t>(g, "cost_kind_t", "Which arm prices an operation.")
      .value("comm", og::cost_kind_t::comm)
      .value("gemm", og::cost_kind_t::gemm)
      .value("roofline", og::cost_kind_t::roofline)
      .value("custom", og::cost_kind_t::custom);

  nb::class_<og::roofline_hardware_t>(
      g, "roofline_hardware_t", "Headline rates for the roofline arm; MI300X-ish by default.")
      .def(nb::init<>())
      .def_rw("peak_flops", &og::roofline_hardware_t::peak_flops)
      .def_rw("hbm_bw", &og::roofline_hardware_t::hbm_bw)
      .def_rw("link_bw", &og::roofline_hardware_t::link_bw)
      .def_rw("link_latency", &og::roofline_hardware_t::link_latency)
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

  g.def("roofline_seconds",
        &og::roofline_seconds,
        "spec"_a,
        "hardware"_a = og::roofline_hardware_t{},
        "The largest of the compute, HBM and link roofs.");
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
      "comm_seconds",
      [](const og::comm_spec_t& spec, const oc::system_t& system, std::optional<int> active_cus) {
        return og::comm_seconds(spec, system, active_cus);
      },
      "spec"_a,
      "system"_a,
      "active_cus"_a = nb::none(),
      "Per-workgroup-tile latency from origami.comm's calibrated model, in seconds.");

  nb::class_<og::op_cost_t>(g, "op_cost_t", "How one operation is priced.")
      .def_static("from_roofline", &og::op_cost_t::from_roofline, "spec"_a)
      .def_static("from_comm", &og::op_cost_t::from_comm, "spec"_a)
      .def_static(
          "from_custom", &og::op_cost_t::from_custom, "fn"_a, "fn(node, active_cus) -> seconds.")
      .def("kind", &og::op_cost_t::kind);

  nb::class_<og::cost_settings_t>(g, "cost_settings_t", "Machine description shared by a table.")
      .def(nb::init<>())
      .def_rw("roofline_hardware", &og::cost_settings_t::roofline_hardware)
      .def_rw("comm_system",
              &og::cost_settings_t::comm_system,
              "Required by the comm arm; build one with origami.comm.make_system.")
      .def_rw("default_seconds", &og::cost_settings_t::default_seconds);

  nb::class_<og::cost_model_t, py_cost_model_t>(
      g,
      "cost_model_t",
      "Anything that can price a node and a hop. Subclass this to price a "
      "machine the shipped arms do not describe: override node_cost and "
      "edge_cost, and node_cost_at if contention matters.")
      .def(nb::init<>())
      .def("node_cost", &og::cost_model_t::node_cost, "node"_a)
      .def("edge_cost", &og::cost_model_t::edge_cost, "edge"_a)
      .def("node_cost_at", &og::cost_model_t::node_cost_at, "node"_a, "active_cus"_a);

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

  // ─── continuous-time runtimes ───────────────────────────────────────

  nb::class_<og::cost_schedule_t>(
      g, "cost_schedule_t", "A continuous-time schedule: real start, duration and lane per atom.")
      .def(nb::init<>())
      .def_rw("runtime", &og::cost_schedule_t::runtime)
      .def_rw("lanes", &og::cost_schedule_t::lanes)
      .def_rw("units", &og::cost_schedule_t::units)
      .def_rw("order", &og::cost_schedule_t::order)
      .def("place",
           &og::cost_schedule_t::place,
           "node"_a,
           "start"_a,
           "duration"_a,
           "lane"_a,
           "Record one atom's start, duration and lane.")
      .def(
          "start",
          [](const og::cost_schedule_t& s, const og::wg_node_t& n) { return s.start.at(n); },
          "node"_a)
      .def(
          "duration",
          [](const og::cost_schedule_t& s, const og::wg_node_t& n) { return s.duration.at(n); },
          "node"_a)
      .def(
          "lane",
          [](const og::cost_schedule_t& s, const og::wg_node_t& n) { return s.lane.at(n); },
          "node"_a)
      .def("finish", &og::cost_schedule_t::finish, "node"_a)
      .def("makespan", &og::cost_schedule_t::makespan)
      .def("num_lanes", &og::cost_schedule_t::num_lanes)
      .def("busy_by_op", &og::cost_schedule_t::busy_by_op)
      .def("utilization", &og::cost_schedule_t::utilization)
      .def("summary", &og::cost_schedule_t::summary);

  nb::class_<og::cost_runtime_options_t>(
      g, "cost_runtime_options_t", "Knobs for the lane-pool continuous runtimes.")
      .def(nb::init<>())
      .def_rw("lanes", &og::cost_runtime_options_t::lanes)
      .def_rw("scale", &og::cost_runtime_options_t::scale)
      .def_rw("units", &og::cost_runtime_options_t::units)
      .def_rw("lane_pool", &og::cost_runtime_options_t::lane_pool)
      .def_rw("serialize_wg_iters", &og::cost_runtime_options_t::serialize_wg_iters);

  nb::class_<og::xcd_options_t>(g, "xcd_options_t", "Chiplet geometry.")
      .def(nb::init<>())
      .def_rw("num_xcds", &og::xcd_options_t::num_xcds)
      .def_rw("cus_per_xcd", &og::xcd_options_t::cus_per_xcd)
      .def_rw("scale", &og::xcd_options_t::scale)
      .def_rw("units", &og::xcd_options_t::units)
      .def_rw("skip_prefix", &og::xcd_options_t::skip_prefix);

  nb::class_<og::cost_runtime_t, py_cost_runtime_t>(
      g,
      "cost_runtime_t",
      "A continuous-time scheduling policy. Subclass this to schedule the way "
      "some machine or kernel actually does: override name() and "
      "schedule(graph), building the result with cost_schedule_t.place.")
      .def(nb::init<>())
      .def("name", &og::cost_runtime_t::name)
      .def("schedule", &og::cost_runtime_t::schedule, "graph"_a);

  // Each takes a cost model or does without one, in which case it prices
  // whatever graph it is given with that graph's own.
  nb::class_<og::roofline_runtime_t, og::cost_runtime_t>(
      g, "roofline_runtime_t", "Greedy continuous time, priced once up front.")
      .def(nb::init<og::cost_runtime_options_t>(), "options"_a = og::cost_runtime_options_t{})
      .def(nb::init<const og::cost_model_t&, og::cost_runtime_options_t>(),
           "cost"_a,
           "options"_a = og::cost_runtime_options_t{},
           nb::keep_alive<1, 2>());
  nb::class_<og::event_driven_runtime_t, og::cost_runtime_t>(
      g, "event_driven_runtime_t", "As roofline, but re-priced against contention at dispatch.")
      .def(nb::init<og::cost_runtime_options_t>(), "options"_a = og::cost_runtime_options_t{})
      .def(nb::init<const og::cost_model_t&, og::cost_runtime_options_t>(),
           "cost"_a,
           "options"_a = og::cost_runtime_options_t{},
           nb::keep_alive<1, 2>());
  nb::class_<og::xcd_runtime_t, og::cost_runtime_t>(
      g, "xcd_runtime_t", "Chiplet round-robin with work stealing, in launch order.")
      .def(nb::init<og::xcd_options_t>(), "options"_a = og::xcd_options_t{})
      .def(nb::init<const og::cost_model_t&, og::xcd_options_t>(),
           "cost"_a,
           "options"_a = og::xcd_options_t{},
           nb::keep_alive<1, 2>());

  // ─── ranking ────────────────────────────────────────────────────────

  nb::enum_<og::runtime_kind_t>(g, "runtime_kind_t", "Which scheduling policy a candidate uses.")
      .value("breadth_first", og::runtime_kind_t::breadth_first)
      .value("asap", og::runtime_kind_t::asap)
      .value("depth_first", og::runtime_kind_t::depth_first)
      .value("roofline", og::runtime_kind_t::roofline)
      .value("event_driven", og::runtime_kind_t::event_driven)
      .value("xcd", og::runtime_kind_t::xcd);

  g.def("is_continuous",
        &og::is_continuous,
        "kind"_a,
        "Whether the policy schedules in real time rather than integer timesteps.");

  nb::class_<og::graph_config_t>(g, "graph_config_t", "One candidate schedule.")
      .def(nb::init<>())
      .def_rw("runtime", &og::graph_config_t::runtime)
      .def_rw("lanes", &og::graph_config_t::lanes)
      .def_rw("serialize_wg_iters", &og::graph_config_t::serialize_wg_iters)
      .def_rw("lane_pool", &og::graph_config_t::lane_pool)
      .def_rw("wg_duration", &og::graph_config_t::wg_duration)
      .def_rw("xcd", &og::graph_config_t::xcd)
      .def_rw("scale", &og::graph_config_t::scale)
      .def_rw("runtime_override",
              &og::graph_config_t::runtime_override,
              "A policy to use verbatim, ignoring every field above.")
      .def_rw("name", &og::graph_config_t::name);

  nb::class_<og::prediction_result_t>(
      g, "prediction_result_t", "A candidate's predicted cost, in the units its policy produced.")
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
        "Makespan for one graph under one candidate schedule.");

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
                          const og::cost_schedule_t&,
                          const og::trace_options_t&>(&og::chrome_trace),
        "graph"_a,
        "schedule"_a,
        "options"_a = og::trace_options_t{},
        "Chrome Trace Event JSON, loadable in chrome://tracing or Perfetto.");
  g.def("chrome_trace",
        nb::overload_cast<const og::wg_graph_t&, const og::schedule_t&, const og::trace_options_t&>(
            &og::chrome_trace),
        "graph"_a,
        "schedule"_a,
        "options"_a = og::trace_options_t{});

  g.def("write_chrome_trace",
        nb::overload_cast<const og::wg_graph_t&,
                          const og::cost_schedule_t&,
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
