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

#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

const std::vector<std::size_t>& wg_graph_t::no_edges() {
  static const std::vector<std::size_t> kNone;
  return kNone;
}

std::vector<wg_node_t> wg_graph_t::nodes() const {
  std::vector<wg_node_t> out;
  out.reserve(num_nodes());
  for (int op = 0; op < num_operations(); ++op) {
    const int wgs   = static_cast<int>(wg_count(op));
    const int iters = static_cast<int>(iter_count(op));
    for (int wg = 0; wg < wgs; ++wg) {
      for (int it = 0; it < iters; ++it) out.push_back(wg_node_t{op, wg, it});
    }
  }
  return out;
}

std::size_t wg_graph_t::num_nodes() const {
  std::size_t total = 0;
  for (int op = 0; op < num_operations(); ++op) {
    total += static_cast<std::size_t>(wg_count(op)) * static_cast<std::size_t>(iter_count(op));
  }
  return total;
}

std::string wg_graph_t::label(const wg_node_t& node) const {
  std::string out = op_name(node.op) + "#" + std::to_string(node.wg);
  if (node.it != 0) out += "." + std::to_string(node.it);
  return out;
}

std::string wg_graph_t::summary() const {
  std::string out = "Graph: " + std::to_string(num_operations()) + " ops, " +
                    std::to_string(num_nodes()) + " workgroup-iterations, " +
                    std::to_string(edges().size()) + " fine edges";
  for (int op = 0; op < num_operations(); ++op) {
    out += "\n  op '" + op_name(op) + "': " + std::to_string(wg_count(op)) + " wgs";
    if (iter_count(op) > 1) out += " x " + std::to_string(iter_count(op)) + " iters";
  }
  return out;
}

}  // namespace origami::graphs
