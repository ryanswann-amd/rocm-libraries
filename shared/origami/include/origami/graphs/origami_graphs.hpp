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

// Umbrella include for origami::graphs (kirigami).
//
// Convenience header — pulls in every public type and function the
// workgroup-graph model exposes. Internal compilation units should prefer the
// finer-grained includes.
//
// Note that the GEMM cost arm (origami/graphs/cost_gemm.hpp) is deliberately
// NOT included here: it links roc::origami and therefore HIP, while everything
// below is pure C++17. Consumers that want it include it explicitly and link
// roc::origami-graphs-gemm.
//
#pragma once

#include "origami/graphs/analysis.hpp"
#include "origami/graphs/core.hpp"
#include "origami/graphs/cost.hpp"
#include "origami/graphs/cost_expr.hpp"
#include "origami/graphs/cost_model.hpp"
#include "origami/graphs/free_graph.hpp"
#include "origami/graphs/placement.hpp"
#include "origami/graphs/ranking.hpp"
#include "origami/graphs/runtime.hpp"
#include "origami/graphs/simulate.hpp"
#include "origami/graphs/spec.hpp"
#include "origami/graphs/symbolic.hpp"
#include "origami/graphs/trace.hpp"
#include "origami/graphs/types.hpp"
#include "origami/graphs/wg_graph.hpp"
