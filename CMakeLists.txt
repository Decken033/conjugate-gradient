
cmake_minimum_required(VERSION 3.20)

project(
    DSA5208_ConjugateGradient
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared header-only interface for all implementations.
add_library(cg_common INTERFACE)

target_include_directories(cg_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(cg_common INTERFACE cxx_std_17)

# Solver targets will be added in their respective tasks:
# - cg_serial
# - cg_openmp
# - cg_mpi
# - cg_hybrid
# - cg_gpu (optional)


include(CTest)

if(BUILD_TESTING)
    add_executable(test_csr tests/test_csr.cpp)
    target_link_libraries(test_csr PRIVATE cg_common)

    add_test(NAME csr_validation COMMAND test_csr)
endif()
