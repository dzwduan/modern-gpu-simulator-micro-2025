cmake_minimum_required(VERSION 3.14)
project(gpu_simulator_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

enable_testing()

set(SIM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../simulator-remodeled/gpu-simulator/gpgpu-sim/src)

add_executable(run_tests
  test_dependency_path.cc
  test_pipeline_routing.cc
)

target_include_directories(run_tests PRIVATE
  ${SIM_SRC_DIR}/gpgpu-sim
  ${SIM_SRC_DIR}
)

target_link_libraries(run_tests GTest::gtest GTest::gtest_main pthread)

include(GoogleTest)
gtest_discover_tests(run_tests)
