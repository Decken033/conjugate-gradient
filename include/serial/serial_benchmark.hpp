
#pragma once

#include "serial/cg_solver.hpp"
#include "common/csr_matrix.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace cg {

// Results from one measured serial CG execution.
// Matrix generation and validation are excluded from
// the measured solver execution time.
struct SerialBenchmarkRecord {
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

// Run one warm-up followed by repeated timed CG solves.
//
// Each measured run uses identical A, b, options and x0 = 0.
// The known exact solution is used to calculate accuracy
// outside the timed section.

std::vector<SerialBenchmarkRecord> benchmark_serial_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const std::vector<double>& exact_solution,
    const std::string& matrix_name,
    std::size_t repeats = 5,
    const CGOptions& options = {},
    std::size_t warmups = 1
);
 // Write benchmark records to a CSV file.
 // Creates or overwrites the specified output file.
 // Each measured run is written as a separate row.
 //
 // Solver execution and timing are completed before
 // this function is called. No CSV I/O is timed.
 void write_serial_benchmark_csv(
     const std::vector<SerialBenchmarkRecord>& records,
     const std::string& output_path
 );
} // namespace cg
