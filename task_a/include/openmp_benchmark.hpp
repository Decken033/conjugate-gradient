
#pragma once

#include "cg_solver.hpp"
#include "csr_matrix.hpp"
#include "openmp_ops.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace cg {

// Results from one measured OpenMP CG execution.
//
// Matrix generation, warm-up, numerical validation,
// and CSV writing are excluded from solver timing.
struct OpenMPBenchmarkRecord {
    std::string matrix_name;

    std::size_t n = 0;
    std::size_t nnz = 0;
    std::size_t run_index = 0;

    // OpenMP execution configuration.
    int threads = 1;
    OpenMPSchedule schedule = OpenMPSchedule::Static;
    int chunk_size = 0;

    // CG convergence information.
    std::size_t iterations = 0;
    bool converged = false;

    // Performance and numerical accuracy.
    double solve_seconds = 0.0;
    double true_relative_residual = 0.0;
    double relative_solution_error = 0.0;
};

// Run configurable untimed warm-ups followed by repeated timed
// OpenMP CG solves.
//
// Each run uses the same A, b, solver settings and
// zero initial guess as the Task 1 serial benchmark.
//
// Only solve_openmp_cg() is timed.

std::vector<OpenMPBenchmarkRecord> benchmark_openmp_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const std::vector<double>& exact_solution,
    const std::string& matrix_name,
    std::size_t repeats = 5,
    const CGOptions& options = {},
    const OpenMPOptions& omp_options = {},
    std::size_t warmups = 1
);

// Write the measured results to a CSV file.
// CSV writing is performed outside all timed sections.
void write_openmp_benchmark_csv(
    const std::vector<OpenMPBenchmarkRecord>& records,
    const std::string& output_path
);

} // namespace cg
