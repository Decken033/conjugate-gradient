#pragma once

#include "serial/cg_solver.hpp"
#include "common/csr_matrix.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace cg {

// Results from one measured OpenMP GPU CG execution.
//
// Matrix generation, warm-ups, independent numerical
// validation and CSV writing are excluded from timing.
//
// The measured solve_gpu_cg() call includes GPU memory
// allocation, host-device transfers and GPU computation.
struct GPUBenchmarkRecord {
    std::string matrix_name;

    std::size_t n = 0;
    std::size_t nnz = 0;
    std::size_t run_index = 0;

    std::size_t iterations = 0;
    bool converged = false;

    double solve_seconds = 0.0;
    double true_relative_residual = 0.0;
    double relative_solution_error = 0.0;
};

// Perform untimed warm-ups followed by repeated timed GPU solves.
// Every solve uses identical inputs and a zero initial guess.
std::vector<GPUBenchmarkRecord> benchmark_gpu_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const std::vector<double>& exact_solution,
    const std::string& matrix_name,
    std::size_t repeats = 5,
    const CGOptions& options = {},
    std::size_t warmups = 1
);

// Write one CSV row per measured run.
// No file I/O is included in the measured solver time.
void write_gpu_benchmark_csv(
    const std::vector<GPUBenchmarkRecord>& records,
    const std::string& output_path
);

} // namespace cg
