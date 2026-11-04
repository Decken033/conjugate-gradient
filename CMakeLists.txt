
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
# OpenMP support for CPU parallel implementations.
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Optional MPI support. Enable with -DCG_ENABLE_MPI=ON.
# MPI targets are added only when the library is found.
# Disabling MPI (or an unavailable MPI library) does not affect
# Serial or OpenMP builds.
option(CG_ENABLE_MPI "Build the distributed-memory MPI CG solver" ON)

# Optional OpenMP GPU offloading (NVIDIA HPC SDK).
# Disabled by default to preserve existing CPU and MPI builds.
option(CG_ENABLE_OPENMP_GPU
    "Build the optional NVHPC OpenMP GPU solver"
    OFF
)

if(CG_ENABLE_MPI)
    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
        set(CG_MPI_AVAILABLE TRUE)
    else()
        message(STATUS "MPI not found; MPI targets will be skipped")
        set(CG_MPI_AVAILABLE FALSE)
    endif()
else()
    set(CG_MPI_AVAILABLE FALSE)
endif()

# Solver targets will be added in their respective tasks:
# - cg_serial
# - cg_openmp
# - cg_mpi
# - cg_hybrid
# - cg_gpu (optional)


# Common numerical operations shared by solver implementations.
add_library(cg_ops STATIC
    src/common/csr_ops.cpp
)

target_link_libraries(cg_ops PUBLIC cg_common)


# Serial Conjugate Gradient solver.
# Depends on the shared CSR matrix-vector multiplication library.
add_library(cg_serial STATIC
    src/serial/cg_serial.cpp
)

target_link_libraries(cg_serial PUBLIC cg_ops)
# OpenMP numerical operations: parallel CSR SpMV and
# subsequent parallel vector/reduction kernels.
add_library(cg_openmp_ops STATIC
    src/openmp/openmp_ops.cpp
)

target_link_libraries(cg_openmp_ops PUBLIC
    cg_common
    OpenMP::OpenMP_CXX
)
# CPU OpenMP Conjugate Gradient solver.
# Reuses parallel SpMV, dot products and vector operations.
add_library(cg_openmp STATIC
    src/openmp/cg_openmp.cpp
)

target_link_libraries(cg_openmp PUBLIC
    cg_openmp_ops
)

# OpenMP CG performance benchmarking library.
# Provides repeated solver timing and CSV reporting.
add_library(cg_openmp_benchmark STATIC
    src/openmp/benchmark_openmp.cpp
    src/openmp/benchmark_openmp_csv.cpp
)

target_link_libraries(cg_openmp_benchmark PUBLIC
    cg_openmp
)
# Shared 2D/3D Poisson matrix generators.
# Reusable by serial, OpenMP, MPI and hybrid solvers.
add_library(cg_poisson STATIC
    src/common/poisson.cpp
)

# Shared Matrix Market loader for real-world sparse matrices.
# Converts coordinate-format matrices into the common CSR structure.
# Supports real general and symmetric Matrix Market matrices.
# Reusable by serial, OpenMP, MPI and hybrid implementations.
add_library(cg_matrix_market STATIC
    src/common/matrix_market.cpp
)

target_link_libraries(cg_matrix_market PUBLIC cg_common)


target_link_libraries(cg_poisson PUBLIC cg_common)

# Serial CG benchmark timing library.
# Measures repeated solver execution and records numerical metrics.
add_library(cg_serial_benchmark STATIC
    src/serial/benchmark_serial.cpp
    src/serial/benchmark_csv.cpp
)

target_link_libraries(cg_serial_benchmark PUBLIC cg_serial)

# Command-line executable for serial CG performance benchmarking.
# Uses the shared Poisson generator and benchmark timing library.
add_executable(run_serial_benchmark
    src/serial/run_serial_benchmark.cpp
)

target_link_libraries(run_serial_benchmark PRIVATE
    cg_serial_benchmark
    cg_poisson
    cg_matrix_market
)

# Command-line runner for OpenMP CG performance benchmarking.
# Supports Poisson matrices and the thermal2 Matrix Market input.
add_executable(run_openmp_benchmark
    src/openmp/run_openmp_benchmark.cpp
)

target_link_libraries(run_openmp_benchmark PRIVATE
    cg_openmp_benchmark
    cg_poisson
    cg_matrix_market
    cg_ops
)

include(CTest)

# MPI solver and benchmark library targets.
# Only added when MPI is found and enabled.
if(CG_MPI_AVAILABLE)

    # MPI numerical operations: partitioning, distributed SpMV, dot products.
    # Modular design allows reuse by hybrid MPI+OpenMP solver (Person C).
    add_library(cg_mpi_ops STATIC
        src/mpi/mpi_partition.cpp
        src/mpi/mpi_ops.cpp
    )
    target_link_libraries(cg_mpi_ops PUBLIC
        cg_common
        MPI::MPI_CXX
    )

    # MPI CG solver. Depends only on mpi_ops.
    add_library(cg_mpi STATIC
        src/mpi/mpi_solver.cpp
    )
    target_link_libraries(cg_mpi PUBLIC cg_mpi_ops)

    # MPI benchmark harness: timing, CSV output, warmup protocol.
    add_library(cg_mpi_benchmark STATIC
        src/mpi/benchmark_mpi.cpp
        src/mpi/benchmark_mpi_csv.cpp
    )
    target_link_libraries(cg_mpi_benchmark PUBLIC cg_mpi)

    # Command-line MPI CG benchmark runner.
    add_executable(run_mpi_benchmark
        src/mpi/run_mpi_benchmark.cpp
    )
    target_link_libraries(run_mpi_benchmark PRIVATE
        cg_mpi_benchmark
        cg_poisson
        cg_matrix_market
    )

    message(STATUS "MPI CG targets: cg_mpi_ops, cg_mpi, cg_mpi_benchmark, run_mpi_benchmark")

endif()

if(BUILD_TESTING)
    add_executable(test_csr tests/test_csr.cpp)
    target_link_libraries(test_csr PRIVATE cg_common)

    add_test(NAME csr_validation COMMAND test_csr)
    add_executable(test_csr_ops tests/test_csr_ops.cpp)
    target_link_libraries(test_csr_ops PRIVATE cg_ops)
    add_test(NAME csr_matvec COMMAND test_csr_ops)
    # OpenMP CSR SpMV correctness tests.
    # Compare parallel operations against the serial reference.
    add_executable(test_openmp_ops
        tests/test_openmp_ops.cpp
    )

    target_link_libraries(test_openmp_ops PRIVATE
        cg_openmp_ops
        cg_ops
    )
    add_test(
        NAME openmp_spmv
        COMMAND test_openmp_ops
    )

    # Complete OpenMP CG correctness tests.
    # Compare OpenMP CG against the serial reference solver
    # on known SPD and manufactured 2D/3D Poisson problems.
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
    # Serial CG solver correctness tests.
    add_executable(test_serial_cg tests/test_serial_cg.cpp)
    target_link_libraries(test_serial_cg PRIVATE cg_serial cg_poisson)
    target_compile_options(test_serial_cg PRIVATE -UNDEBUG)
    add_test(NAME serial_cg COMMAND test_serial_cg)
    # Serial benchmark correctness tests.
    add_executable(test_serial_benchmark
        tests/test_serial_benchmark.cpp
    )
    target_link_libraries(test_serial_benchmark
        PRIVATE cg_serial_benchmark
    )
    target_compile_options(test_serial_benchmark PRIVATE -UNDEBUG)
    add_test(NAME serial_benchmark COMMAND test_serial_benchmark)
    # Matrix Market loader correctness tests.
    # Verify symmetric expansion, duplicate entries,
    # and rejection of invalid coordinates.
    add_executable(test_matrix_market
        tests/test_matrix_market.cpp
    )

    target_link_libraries(test_matrix_market
        PRIVATE cg_matrix_market
    )

    # Keep assertions enabled in Release builds.
    target_compile_options(test_matrix_market PRIVATE -UNDEBUG)

    add_test(
        NAME matrix_market
        COMMAND test_matrix_market
    )

    # MPI tests require MPI library and mpiexec launcher.
    # Each test is run with 4 MPI ranks to verify distributed correctness.
    if(CG_MPI_AVAILABLE)
        # MPI partition correctness: row distribution and gather/scatter
        add_executable(test_mpi_partition
            tests/test_mpi_partition.cpp
        )
        target_link_libraries(test_mpi_partition PRIVATE
            cg_mpi_ops
            cg_poisson
        )
        add_test(
            NAME mpi_partition
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_mpi_partition>
                    ${MPIEXEC_POSTFLAGS}
        )

        # MPI SpMV correctness: compare distributed vs serial result
        add_executable(test_mpi_spmv
            tests/test_mpi_spmv.cpp
        )
        target_link_libraries(test_mpi_spmv PRIVATE
            cg_mpi_ops
            cg_ops
            cg_poisson
        )
        add_test(
            NAME mpi_spmv
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_mpi_spmv>
                    ${MPIEXEC_POSTFLAGS}
        )

        # MPI CG correctness: convergence and cross-rank consistency
        add_executable(test_mpi_cg
            tests/test_mpi_cg.cpp
        )
        target_link_libraries(test_mpi_cg PRIVATE
            cg_mpi
            cg_serial
            cg_poisson
        )
        add_test(
            NAME mpi_cg
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_mpi_cg>
                    ${MPIEXEC_POSTFLAGS}
        )
    endif()
endif()

# ------------------------------------------------------------
# Optional OpenMP GPU-offloaded CG solver.
# This extension is disabled by default.
# ------------------------------------------------------------
if(CG_ENABLE_OPENMP_GPU)

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        message(FATAL_ERROR
            "CG_ENABLE_OPENMP_GPU requires the NVHPC nvc++ compiler"
        )
    endif()

    # GPU Conjugate Gradient solver.
    add_library(cg_gpu STATIC
        src/gpu/cg_gpu.cpp
    )

    target_link_libraries(cg_gpu PUBLIC
        cg_common
    )

    target_compile_options(cg_gpu PRIVATE
        -mp=gpu
        -gpu=cc86
    )

    # GPU numerical correctness tests.
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

        target_compile_options(test_gpu_cg PRIVATE
            -mp=gpu
            -gpu=cc86
        )

        target_link_options(test_gpu_cg PRIVATE
            -mp=gpu
            -gpu=cc86
        )

        add_test(
            NAME gpu_cg
            COMMAND test_gpu_cg
        )

        set_tests_properties(gpu_cg PROPERTIES
            LABELS "gpu"
            ENVIRONMENT "OMP_TARGET_OFFLOAD=MANDATORY"
        )

    endif()

endif()

# OpenMP GPU benchmark timing and CSV reporting.
# Enabled only when the NVHPC GPU extension is selected.
if(CG_ENABLE_OPENMP_GPU)
    add_library(cg_gpu_benchmark STATIC
        src/gpu/benchmark_gpu.cpp
        src/gpu/benchmark_gpu_csv.cpp
    )

    target_link_libraries(cg_gpu_benchmark PUBLIC
        cg_gpu
        cg_ops
    )

    # Command-line GPU benchmark runner.
    add_executable(run_gpu_benchmark
        src/gpu/run_gpu_benchmark.cpp
    )

    target_link_libraries(run_gpu_benchmark PRIVATE
        cg_gpu_benchmark
        cg_poisson
        cg_matrix_market
    )

    # Link the executable with NVIDIA GPU OpenMP support.
    target_link_options(run_gpu_benchmark PRIVATE
        -mp=gpu
        -gpu=cc86
    )
endif()
