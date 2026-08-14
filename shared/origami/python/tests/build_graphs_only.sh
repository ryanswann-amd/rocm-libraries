#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build *only* the origami.graphs bindings, with no ROCm toolchain and no GPU.
#
# The full `origami` wheel needs HIP, because the GEMM arm links against it.
# origami::graphs does not — run_standalone.sh already proves the C++ side
# builds with nothing but g++ — and the bindings for it are in the same
# position. This script is the Python-side counterpart: it compiles
# bind_graphs.cpp and the HIP-free sources it needs into a module named
# `origami_graphs`, so the graphs bindings can be exercised on a laptop.
#
# The module is not a substitute for the wheel: the GEMM surface is bound in
# bindings.cpp alongside HIP-linked code and is absent here. Enough of
# `origami.comm` is bound below to describe a machine, because the comm cost
# model itself is HIP-free and the graphs cost table prices operations with it.
# Its collective entry points are not bound. With that in place the whole of
# test_graphs.py passes against this build, so a change to the graphs bindings
# can be tested where it is made rather than only on a ROCm machine.
#
# A shim that answers to the name `origami` is written next to the module, so
# the suite's import path works unchanged:
#
#   ./build_graphs_only.sh
#   PYTHONPATH=/tmp/origami-graphs-pymodule \
#       pytest -p origami_shim --noconftest test_graphs.py
#
# (--noconftest because the shared conftest builds hardware descriptions through
# the parts of origami this build does not have.)
#
# Usage: ./build_graphs_only.sh [output-dir]
#        PYTHON=... to pick an interpreter (it must have nanobind installed)

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:-${TMPDIR:-/tmp}/origami-graphs-pymodule}"
python="${PYTHON:-python3}"

nb_root="$("$python" -c 'import nanobind; print(nanobind.cmake_dir())')/.."
nb_dir="$(cd "$nb_root/include" && pwd)"
nb_src="$(cd "$nb_root/src" && pwd)"
nb_map="$(cd "$nb_root/ext/robin_map/include" && pwd)"
py_inc="$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
ext="$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

mkdir -p "$out/stub/origami"
cat > "$out/stub/origami/origami_export.h" <<'STUB'
#ifndef ORIGAMI_EXPORT_H
#define ORIGAMI_EXPORT_H
#define ORIGAMI_EXPORT
#define ORIGAMI_NO_EXPORT
#define ORIGAMI_DEPRECATED
#define ORIGAMI_DEPRECATED_EXPORT
#define ORIGAMI_DEPRECATED_NO_EXPORT
#endif
STUB

# bind_graphs.cpp defines bind_graphs(module_&); bindings.cpp would supply the
# module itself, but it also binds the HIP-linked GEMM surface, so a short entry
# point stands in for it.
#
# It also binds the handful of origami::comm types needed to build a system_t.
# bind_graphs.cpp already binds comm_spec_t and comm_cycles, so the comm arm is
# reachable from Python except for its machine description, which lives in
# bindings.cpp beside HIP-linked code. The model itself is HIP-free and is
# already linked in here, so what is missing is only the constructors. They are
# bound under the name `comm` so a caller writes origami.comm.make_system
# against either build. This is a duplicate of bindings.cpp's declarations and
# deliberately lives with the test build rather than in shipped source: the two
# are never compiled together, and nanobind would reject the second
# registration of the same C++ type if they were.
cat > "$out/module.cpp" <<'MAIN'
#include <nanobind/nanobind.h>

#include "origami/comm/hardware.hpp"
#include "origami/comm/primitives.hpp"
#include "origami/types.hpp"

void bind_graphs(nanobind::module_& m);

namespace nb = nanobind;
namespace oc = origami::comm;

NB_MODULE(origami_graphs, m) {
  bind_graphs(m);

  nb::enum_<origami::architecture_t>(m, "architecture_t")
      .value("gfx90a", origami::architecture_t::gfx90a)
      .value("gfx942", origami::architecture_t::gfx942)
      .value("gfx950", origami::architecture_t::gfx950);

  auto comm = m.def_submodule("comm", "Enough of origami.comm to describe a machine.");

  nb::enum_<oc::primitive_t>(comm, "primitive_t", "The collective operation.")
      .value("all_gather", oc::primitive_t::all_gather)
      .value("reduce_scatter", oc::primitive_t::reduce_scatter)
      .value("broadcast", oc::primitive_t::broadcast)
      .value("all_reduce", oc::primitive_t::all_reduce)
      .value("all_to_all", oc::primitive_t::all_to_all)
      .export_values();

  nb::class_<oc::hardware_t>(comm, "hardware_t", "Per-GPU compute and memory ceilings.")
      .def_ro("num_cu", &oc::hardware_t::num_cu)
      .def_ro("num_xcd", &oc::hardware_t::num_xcd)
      .def_ro("cu_per_xcd", &oc::hardware_t::cu_per_xcd)
      .def_ro("clock_ghz", &oc::hardware_t::clock_ghz)
      .def_ro("mshr_depth_per_wave", &oc::hardware_t::mshr_depth_per_wave)
      .def_ro("waves_per_wg", &oc::hardware_t::waves_per_wg)
      .def_ro("xgmi_latency_cycles", &oc::hardware_t::xgmi_latency_cycles)
      .def_ro("cacheline_bytes", &oc::hardware_t::cacheline_bytes);

  nb::class_<oc::comm_hardware_t>(comm, "comm_hardware_t", "The xGMI mesh between GPUs.")
      .def_ro("link_bw", &oc::comm_hardware_t::link_bw)
      .def_ro("num_peer_links", &oc::comm_hardware_t::num_peer_links)
      .def_ro("clock_ghz", &oc::comm_hardware_t::clock_ghz);

  nb::class_<oc::system_t>(comm, "system_t", "A GPU plus the fabric that joins it to its peers.")
      .def_ro("gpu", &oc::system_t::gpu)
      .def_ro("fabric", &oc::system_t::fabric);

  nb::class_<oc::gpu_topology_t>(comm, "gpu_topology_t", "Per-device GPU shape.")
      .def(
          "__init__",
          [](oc::gpu_topology_t* self,
             origami::architecture_t arch,
             std::size_t num_cu,
             std::size_t num_xcd,
             std::size_t cu_per_xcd,
             std::size_t l2_capacity_bytes) {
            new (self) oc::gpu_topology_t{arch, num_cu, num_xcd, cu_per_xcd, l2_capacity_bytes};
          },
          nb::arg("arch"),
          nb::arg("num_cu"),
          nb::arg("num_xcd"),
          nb::arg("cu_per_xcd"),
          nb::arg("l2_capacity_bytes"));

  nb::class_<oc::arch_ceilings_t>(
      comm, "arch_ceilings_t", "Calibrated per-architecture comm ceilings.");

  comm.def("get_arch_ceilings",
           &oc::get_arch_ceilings,
           nb::arg("arch"),
           "Measured ceilings for an architecture.");
  comm.def("make_system",
           &oc::make_system,
           nb::arg("ceilings"),
           nb::arg("topology"),
           nb::arg("clock_ghz"),
           "Fuse calibrated ceilings, a topology and a clock into a system_t.");
}
MAIN

mapfile -t sources < <(find "$root/src/origami/graphs" -name '*.cpp' ! -name 'cost_gemm.cpp' | sort)
comm_sources=(
    "$root/src/origami/comm/latency.cpp"
    "$root/src/origami/comm/primitives.cpp"
    "$root/src/origami/types.cpp"
)

echo "building origami_graphs$ext with $python"

g++ -std=c++17 -O2 -fPIC -shared -fvisibility=hidden \
    -I "$root/include" \
    -I "$out/stub" \
    -I "$nb_dir" \
    -I "$nb_map" \
    -I "$py_inc" \
    -DNB_SHARED -DNB_BUILD \
    "$nb_src/nb_combined.cpp" \
    "${sources[@]}" \
    "${comm_sources[@]}" \
    "$root/python/src/origami/bind_graphs.cpp" \
    "$out/module.cpp" \
    -o "$out/origami_graphs$ext"

cat > "$out/origami_shim.py" <<SHIM
"""Register this build as \`origami\`, as a pytest plugin so it lands first.

The suite reaches for \`origami.graphs\` and puts python/src on the path to find
it, which would pull in the real package and fail for want of its HIP-linked
extension. Claiming the name in sys.modules at plugin load time, before
collection imports anything, is what makes the same test file run against this
build. Exposes \`graphs\` and \`prebuilt\`; the real package also binds comm and
the GEMM surface, both of which need HIP.

Written by python/tests/build_graphs_only.sh, with the source directory baked in.
"""

import importlib.util
import sys
import types

import origami_graphs

_shim = types.ModuleType("origami")
_shim.graphs = origami_graphs.graphs
_shim.comm = origami_graphs.comm
_shim.architecture_t = origami_graphs.architecture_t
sys.modules.setdefault("origami", _shim)
sys.modules.setdefault("origami.comm", origami_graphs.comm)

# prebuilt.py is pure Python over origami.graphs, so it runs against this build
# unchanged. Loaded from the source tree rather than copied, so there is one of it.
_spec = importlib.util.spec_from_file_location(
    "origami.prebuilt", "$root/python/src/origami/prebuilt.py"
)
_prebuilt = importlib.util.module_from_spec(_spec)
sys.modules["origami.prebuilt"] = _prebuilt
_spec.loader.exec_module(_prebuilt)
_shim.prebuilt = _prebuilt
SHIM

echo "wrote $out/origami_graphs$ext and origami_shim.py beside it"
