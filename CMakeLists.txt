cmake_minimum_required(VERSION 3.20)

project(
    DSA5208_ConjugateGradient
    VERSION 0.1.0
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Global C++ configuration
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared include path for all project modules.
add_library(cg_common INTERFACE)
target_include_directories(cg_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(cg_common INTERFACE cxx_std_17)

# CPU OpenMP is used by the standalone OpenMP solver and Hybrid solver.
find_package(OpenMP REQUIRED COMPONENTS CXX)

# -----------------------------------------------------------------------------
# Optional MPI configuration
# -----------------------------------------------------------------------------
option(CG_ENABLE_MPI "Build MPI and Hybrid MPI+OpenMP solvers" ON)

if(CG_ENABLE_MPI)
    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        set(CG_MPI_AVAILABLE TRUE)
        message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    else()
        set(CG_MPI_AVAILABLE FALSE)
        message(WARNING "MPI was requested but not found; MPI and Hybrid targets will be skipped")
    endif()
else()
    set(CG_MPI_AVAILABLE FALSE)
    message(STATUS "MPI support disabled")
endif()

# Optional NVHPC OpenMP GPU offloading. CPU builds leave this OFF.
option(CG_ENABLE_OPENMP_GPU
    "Build the optional NVHPC OpenMP GPU solver"
    OFF
)

# -----------------------------------------------------------------------------
# Common libraries
# -----------------------------------------------------------------------------
add_library(cg_ops STATIC
    src/common/csr_ops.cpp
)
target_link_libraries(cg_ops PUBLIC cg_common)

add_library(cg_poisson STATIC
    src/common/poisson.cpp
)
target_link_libraries(cg_poisson PUBLIC cg_common)

add_library(cg_matrix_market STATIC
    src/common/matrix_market.cpp
)
target_link_libraries(cg_matrix_market PUBLIC cg_common)

# -----------------------------------------------------------------------------
# Serial solver and benchmark
# -----------------------------------------------------------------------------
add_library(cg_serial STATIC
    src/serial/cg_serial.cpp
)
target_link_libraries(cg_serial PUBLIC cg_ops)

add_library(cg_serial_benchmark STATIC
    src/serial/benchmark_serial.cpp
    src/serial/benchmark_csv.cpp
)
target_link_libraries(cg_serial_benchmark PUBLIC cg_serial)

add_executable(run_serial_benchmark
    src/serial/run_serial_benchmark.cpp
)
target_link_libraries(run_serial_benchmark PRIVATE
    cg_serial_benchmark
    cg_poisson
    cg_matrix_market
)

# -----------------------------------------------------------------------------
# Standalone OpenMP solver and benchmark
# -----------------------------------------------------------------------------
add_library(cg_openmp_ops STATIC
    src/openmp/openmp_ops.cpp
)
target_link_libraries(cg_openmp_ops PUBLIC
    cg_common
    OpenMP::OpenMP_CXX
)

add_library(cg_openmp STATIC
    src/openmp/cg_openmp.cpp
)
target_link_libraries(cg_openmp PUBLIC cg_openmp_ops)

add_library(cg_openmp_benchmark STATIC
    src/openmp/benchmark_openmp.cpp
    src/openmp/benchmark_openmp_csv.cpp
)
target_link_libraries(cg_openmp_benchmark PUBLIC cg_openmp)

add_executable(run_openmp_benchmark
    src/openmp/run_openmp_benchmark.cpp
)
target_link_libraries(run_openmp_benchmark PRIVATE
    cg_openmp_benchmark
    cg_poisson
    cg_matrix_market
    cg_ops
)

# -----------------------------------------------------------------------------
# MPI and Hybrid MPI+OpenMP modules
# -----------------------------------------------------------------------------
if(CG_MPI_AVAILABLE)

    # MPI partitioning and rank-local numerical operations.
    add_library(cg_mpi_ops STATIC
        src/mpi/mpi_partition.cpp
        src/mpi/mpi_ops.cpp
    )
    target_link_libraries(cg_mpi_ops PUBLIC
        cg_common
        MPI::MPI_CXX
    )

    add_library(cg_mpi STATIC
        src/mpi/mpi_solver.cpp
    )
    target_link_libraries(cg_mpi PUBLIC cg_mpi_ops)

    add_library(cg_mpi_benchmark STATIC
        src/mpi/benchmark_mpi.cpp
        src/mpi/benchmark_mpi_csv.cpp
    )
    target_link_libraries(cg_mpi_benchmark PUBLIC cg_mpi)

    add_executable(run_mpi_benchmark
        src/mpi/run_mpi_benchmark.cpp
    )
    target_link_libraries(run_mpi_benchmark PRIVATE
        cg_mpi_benchmark
        cg_poisson
        cg_matrix_market
    )

    # Hybrid operations combine MPI communication with OpenMP local kernels.
    # cg_mpi_ops provides MPIPartitionedMatrix/partitioning and MPI operations;
    # OpenMP::OpenMP_CXX provides the OpenMP compile and link requirements.
    add_library(cg_hybrid_ops STATIC
        src/hybrid/hybrid_ops.cpp
        src/hybrid/hybrid_workspace.cpp
    )
    target_link_libraries(cg_hybrid_ops PUBLIC
        cg_mpi_ops
        OpenMP::OpenMP_CXX
    )

    # Hybrid CG control flow calls the Hybrid operations above.
    add_library(cg_hybrid STATIC
        src/hybrid/hybrid_solver.cpp
    )
    target_link_libraries(cg_hybrid PUBLIC cg_hybrid_ops)

    # Hybrid benchmark timing and CSV output.
    add_library(cg_hybrid_benchmark STATIC
        src/hybrid/benchmark_hybrid.cpp
        src/hybrid/benchmark_hybrid_csv.cpp
    )
    target_link_libraries(cg_hybrid_benchmark PUBLIC cg_hybrid)

    # Hybrid command-line runner.
    add_executable(run_hybrid_benchmark
        src/hybrid/run_hybrid_benchmark.cpp
    )
    target_link_libraries(run_hybrid_benchmark PRIVATE
        cg_hybrid_benchmark
        cg_poisson
        cg_matrix_market
    )

    message(STATUS "MPI targets enabled: cg_mpi_ops, cg_mpi, cg_mpi_benchmark")
    message(STATUS "Hybrid targets enabled: cg_hybrid_ops, cg_hybrid, cg_hybrid_benchmark, run_hybrid_benchmark")

endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
include(CTest)

if(BUILD_TESTING)

    # Common / CSR tests.
    add_executable(test_csr
        tests/test_csr.cpp
    )
    target_link_libraries(test_csr PRIVATE cg_common)
    add_test(NAME csr_validation COMMAND test_csr)

    add_executable(test_csr_ops
        tests/test_csr_ops.cpp
    )
    target_link_libraries(test_csr_ops PRIVATE cg_ops)
    add_test(NAME csr_matvec COMMAND test_csr_ops)

    add_executable(test_matrix_market
        tests/test_matrix_market.cpp
    )
    target_link_libraries(test_matrix_market PRIVATE cg_matrix_market)
    target_compile_options(test_matrix_market PRIVATE -UNDEBUG)
    add_test(NAME matrix_market COMMAND test_matrix_market)

    # Serial tests.
    add_executable(test_serial_cg
        tests/test_serial_cg.cpp
    )
    target_link_libraries(test_serial_cg PRIVATE cg_serial cg_poisson)
    target_compile_options(test_serial_cg PRIVATE -UNDEBUG)
    add_test(NAME serial_cg COMMAND test_serial_cg)

    add_executable(test_serial_benchmark
        tests/test_serial_benchmark.cpp
    )
    target_link_libraries(test_serial_benchmark PRIVATE cg_serial_benchmark)
    target_compile_options(test_serial_benchmark PRIVATE -UNDEBUG)
    add_test(NAME serial_benchmark COMMAND test_serial_benchmark)

    # Standalone OpenMP tests.
    add_executable(test_openmp_ops
        tests/test_openmp_ops.cpp
    )
    target_link_libraries(test_openmp_ops PRIVATE cg_openmp_ops cg_ops)
    add_test(NAME openmp_spmv COMMAND test_openmp_ops)

    add_executable(test_openmp_cg
        tests/test_openmp_cg.cpp
    )
    target_link_libraries(test_openmp_cg PRIVATE
        cg_openmp
        cg_serial
        cg_ops
        cg_poisson
    )
    add_test(NAME openmp_cg COMMAND test_openmp_cg)

    # MPI and Hybrid tests are registered only when MPI is available.
    if(CG_MPI_AVAILABLE)

        add_executable(test_mpi_partition
            tests/test_mpi_partition.cpp
        )
        target_link_libraries(test_mpi_partition PRIVATE cg_mpi_ops cg_poisson)
        add_test(
            NAME mpi_partition
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_mpi_partition>
                    ${MPIEXEC_POSTFLAGS}
        )

        add_executable(test_mpi_spmv
            tests/test_mpi_spmv.cpp
        )
        target_link_libraries(test_mpi_spmv PRIVATE cg_mpi_ops cg_ops cg_poisson)
        add_test(
            NAME mpi_spmv
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_mpi_spmv>
                    ${MPIEXEC_POSTFLAGS}
        )

        add_executable(test_mpi_cg
            tests/test_mpi_cg.cpp
        )
        target_link_libraries(test_mpi_cg PRIVATE cg_mpi cg_serial cg_poisson)
        add_test(
            NAME mpi_cg
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_mpi_cg>
                    ${MPIEXEC_POSTFLAGS}
        )

        add_executable(test_hybrid_ops
            tests/test_hybrid_ops.cpp
        )
        target_link_libraries(test_hybrid_ops PRIVATE
            cg_hybrid_ops
            cg_mpi_ops
            cg_ops
            cg_poisson
        )
        add_test(
            NAME hybrid_ops
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_hybrid_ops>
                    ${MPIEXEC_POSTFLAGS}
        )

        add_executable(test_hybrid_cg
            tests/test_hybrid_cg.cpp
        )
        target_link_libraries(test_hybrid_cg PRIVATE cg_hybrid cg_poisson)
        add_test(
            NAME hybrid_cg
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_hybrid_cg>
                    ${MPIEXEC_POSTFLAGS}
        )

        set_tests_properties(
            mpi_partition mpi_spmv mpi_cg hybrid_ops hybrid_cg
            PROPERTIES PROCESSORS 4
        )
    endif()

endif()

# -----------------------------------------------------------------------------
# Optional NVHPC OpenMP GPU-offloaded solver
# -----------------------------------------------------------------------------
if(CG_ENABLE_OPENMP_GPU)

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        message(FATAL_ERROR
            "CG_ENABLE_OPENMP_GPU requires the NVHPC nvc++ compiler"
        )
    endif()

    add_library(cg_gpu STATIC
        src/gpu/cg_gpu.cpp
    )
    target_link_libraries(cg_gpu PUBLIC cg_common)
    target_compile_options(cg_gpu PRIVATE -mp=gpu -gpu=cc86)

    add_library(cg_gpu_benchmark STATIC
        src/gpu/benchmark_gpu.cpp
        src/gpu/benchmark_gpu_csv.cpp
    )
    target_link_libraries(cg_gpu_benchmark PUBLIC cg_gpu cg_ops)

    add_executable(run_gpu_benchmark
        src/gpu/run_gpu_benchmark.cpp
    )
    target_link_libraries(run_gpu_benchmark PRIVATE
        cg_gpu_benchmark
        cg_poisson
        cg_matrix_market
    )
    target_compile_options(run_gpu_benchmark PRIVATE -mp=gpu -gpu=cc86)
    target_link_options(run_gpu_benchmark PRIVATE -mp=gpu -gpu=cc86)

    if(BUILD_TESTING)
        add_executable(test_gpu_cg
            tests/test_gpu_cg.cpp
        )
        target_link_libraries(test_gpu_cg PRIVATE
            cg_gpu
            cg_serial
            cg_poisson
            cg_ops
        )
        target_compile_options(test_gpu_cg PRIVATE -mp=gpu -gpu=cc86 -UNDEBUG)
        target_link_options(test_gpu_cg PRIVATE -mp=gpu -gpu=cc86)
        add_test(NAME gpu_cg COMMAND test_gpu_cg)
        set_tests_properties(gpu_cg PROPERTIES
            LABELS "gpu"
            ENVIRONMENT "OMP_TARGET_OFFLOAD=MANDATORY"
        )
    endif()

endif()
