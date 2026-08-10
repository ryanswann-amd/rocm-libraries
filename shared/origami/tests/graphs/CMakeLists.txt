# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# origami::graphs (kirigami) suite.
#
# Links the compiled roc::origami-graphs library (no HIP), so this runner builds
# and runs without a ROCm toolchain. Catch2 is provided by the parent
# tests/CMakeLists.txt (find_package or FetchContent).
#
# test_cost_gemm.cpp is deliberately absent: it covers the HIP-linked bridge and
# so lives in its own runner below, which keeps this one buildable on a machine
# with no ROCm install. Splitting the runners is what makes the HIP-free claim
# testable rather than merely asserted.
#
# standalone_main.cpp is deliberately absent from the source list: it defines
# main() for the dependency-free harness backend used by run_standalone.sh, and
# would collide with Catch2::Catch2WithMain here.

add_executable(origami-graphs-tests)

target_sources(
    origami-graphs-tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test_types.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_symbolic.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_core.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_free_graph.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_analysis.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_runtime.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_cost.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_cost_runtime.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_ranking.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_trace.cpp"
)

target_include_directories(origami-graphs-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(
    origami-graphs-tests PRIVATE roc::origami-graphs Catch2::Catch2WithMain
)

catch_discover_tests(origami-graphs-tests
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# The GEMM bridge suite, which needs HIP headers and origami's analytical model.
add_executable(origami-graphs-gemm-tests)

target_sources(
    origami-graphs-gemm-tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test_cost_gemm.cpp"
)

target_include_directories(origami-graphs-gemm-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(
    origami-graphs-gemm-tests PRIVATE roc::origami-graphs-gemm Catch2::Catch2WithMain
)

catch_discover_tests(origami-graphs-gemm-tests
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
