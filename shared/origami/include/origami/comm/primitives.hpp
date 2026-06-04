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

// origami::comm — analytical communication cost model
//
// Communication primitives — composable operations that map to functional
// unit work. Mirrors origami_comms/model/primitives.py.
//
// Each primitive's resolve() traces the full data path through the cache
// hierarchy and returns functional_unit_work_t for one loop iteration
// (cl_per_iter cache lines).
#pragma once

#include "origami/comm/types.hpp"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace origami::comm {

// ─── op_t argument bundle ─────────────────────────────────────────
// Avoids 3 positional args at every call site and matches the
// Python signature (cl_per_iter, instrs_per_cl, elements_per_iter).
struct resolve_args_t {
  int cl_per_iter;
  int instrs_per_cl;
  int elements_per_iter;
};

// ─── Primitives ──────────────────────────────────────────────────
// Read from local HBM into registers.
struct load_t {
  constexpr functional_unit_work_t resolve(const resolve_args_t& a) const noexcept {
    functional_unit_work_t w{};
    w.vmem_read_instrs = static_cast<std::int64_t>(a.cl_per_iter) * a.instrs_per_cl;
    w.tcp_read_cl      = a.cl_per_iter;
    w.l2_read_cl       = a.cl_per_iter;
    w.mall_read_cl     = a.cl_per_iter;
    w.hbm_read_cl      = a.cl_per_iter;
    return w;
  }
};

// Write from registers to local HBM. `write_through=true` bypasses L2.
struct store_t {
  bool write_through = false;

  constexpr functional_unit_work_t resolve(const resolve_args_t& a) const noexcept {
    functional_unit_work_t w{};
    w.vmem_write_instrs = static_cast<std::int64_t>(a.cl_per_iter) * a.instrs_per_cl;
    w.tcp_write_cl      = a.cl_per_iter;
    w.l2_write_cl       = write_through ? 0 : static_cast<std::int64_t>(a.cl_per_iter);
    w.mall_write_cl     = a.cl_per_iter;
    w.hbm_write_cl      = a.cl_per_iter;
    return w;
  }
};

// Read from a remote GPU's HBM via xGMI (ingress).
struct pull_t {
  int peer = 0;

  constexpr functional_unit_work_t resolve(const resolve_args_t& a) const noexcept {
    functional_unit_work_t w{};
    w.vmem_read_instrs = static_cast<std::int64_t>(a.cl_per_iter) * a.instrs_per_cl;
    w.tcp_read_cl      = a.cl_per_iter;
    w.l2_read_cl       = a.cl_per_iter;
    w.xgmi_read_cl     = a.cl_per_iter;
    return w;
  }
};

// Write to a remote GPU's HBM via xGMI (egress).
struct push_t {
  int peer = 0;

  constexpr functional_unit_work_t resolve(const resolve_args_t& a) const noexcept {
    functional_unit_work_t w{};
    w.vmem_read_instrs = static_cast<std::int64_t>(a.cl_per_iter) * a.instrs_per_cl;
    w.tcp_read_cl      = a.cl_per_iter;
    w.l2_read_cl       = a.cl_per_iter;
    w.mall_read_cl     = a.cl_per_iter;
    w.hbm_read_cl      = a.cl_per_iter;
    w.xgmi_write_cl    = a.cl_per_iter;
    return w;
  }
};

// Element-wise reduction on data in registers.
struct reduce_t {
  reduce_op_t op = reduce_op_t::SUM;

  constexpr functional_unit_work_t resolve(const resolve_args_t& a) const noexcept {
    functional_unit_work_t w{};
    w.valu_ops = a.elements_per_iter;
    return w;
  }
};

// Notify a peer that data is ready.
struct signal_t {
  int peer = 0;

  constexpr functional_unit_work_t resolve(const resolve_args_t&) const noexcept {
    functional_unit_work_t w{};
    w.atomic_count  = 1;
    w.xgmi_write_cl = 1;
    return w;
  }
};

// Spin-wait for a peer's signal.
struct wait_t {
  int peer = 0;

  constexpr functional_unit_work_t resolve(const resolve_args_t&) const noexcept {
    functional_unit_work_t w{};
    w.atomic_count = 1;
    w.l2_read_cl   = 1;
    return w;
  }
};

// Sum type for any primitive in a work graph.
using op_t = std::variant<load_t, store_t, pull_t, push_t, reduce_t, signal_t, wait_t>;

// ─── resolve_work_graph ─────────────────────────────────────────
// Composes a list of Ops into per-iteration (`iter_work`) and sync
// (`sync_work`) functional_unit_work_t totals. signal_t/wait_t are charged
// once per iteration to `sync_work`; everything else accumulates
// into `iter_work` (the inner-loop body).
//
// Mirrors model.latency.resolve_work_graph in Python.
struct resolved_work_t {
  functional_unit_work_t iter_work;
  functional_unit_work_t sync_work;
};

inline resolved_work_t resolve_work_graph(const std::vector<op_t>& ops,
                                          const resolve_args_t& args) noexcept {
  resolved_work_t out{};
  for (const op_t& op : ops) {
    std::visit(
        [&](const auto& concrete) {
          using T                        = std::decay_t<decltype(concrete)>;
          const functional_unit_work_t w = concrete.resolve(args);
          if constexpr (std::is_same_v<T, signal_t> || std::is_same_v<T, wait_t>) {
            out.sync_work += w;
          } else {
            out.iter_work += w;
          }
        },
        op);
  }
  return out;
}

// Convenience overload (positional args matching Python signature).
inline resolved_work_t resolve_work_graph(const std::vector<op_t>& ops,
                                          int cl_per_iter,
                                          int instrs_per_cl,
                                          int elements_per_iter) noexcept {
  return resolve_work_graph(ops, resolve_args_t{cl_per_iter, instrs_per_cl, elements_per_iter});
}

}  // namespace origami::comm
