
/*
 * OpenMP CG benchmark timing implementation.
 *
 * Timing methodology matches Task 1 serial benchmarking:
 * - Configurable untimed warm-up solves.
 * - Repeated timed solves with identical inputs.
 * - Measure solver-only elapsed wall-clock time.
 * - Calculate solution accuracy outside the timed region.
 * - Keep individual run records for CSV reporting.
 *
 * Wall-clock timing uses C++17 std::chrono::steady_clock.
 */

#include "openmp_benchmark.hpp"
#include "openmp_solver.hpp"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cg {
namespace {

// Stable Euclidean norm, used only outside timed sections.
double vector_norm(const std::vector<double>& v) {
    double norm = 0.0;

    for (double value : v) {
        norm = std::hypot(norm, value);
    }

    return norm;
}

// Relative solution error:
// ||x - x_exact||_2 / ||x_exact||_2
//
// For a zero exact solution, return absolute error norm.
double compute_solution_error(
    const std::vector<double>& x,
    const std::vector<double>& exact_solution
) {
    if (x.size() != exact_solution.size()) {
        throw std::invalid_argument(
            "OpenMP benchmark: solution dimension mismatch"
        );
    }

    std::vector<double> difference(x.size());

    for (std::size_t i = 0; i < x.size(); ++i) {
        difference[i] = x[i] - exact_solution[i];
    }

    const double error_norm = vector_norm(difference);
    const double exact_norm = vector_norm(exact_solution);

    return exact_norm > 0.0
        ? error_norm / exact_norm
        : error_norm;
}

} // namespace


std::vector<OpenMPBenchmarkRecord> benchmark_openmp_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const std::vector<double>& exact_solution,
    const std::string& matrix_name,
    std::size_t repeats,
    const CGOptions& options,
    const OpenMPOptions& omp_options,
    std::size_t warmups
) {
    A.validate();

    if (b.size() != A.rows ||
        exact_solution.size() != A.rows) {
        throw std::invalid_argument(
            "OpenMP benchmark: input dimension mismatch"
        );
    }

    if (repeats == 0) {
        throw std::invalid_argument(
            "OpenMP benchmark: repeats must be at least 1"
        );
    }

    if (omp_options.num_threads < 0 ||
        omp_options.chunk_size < 0) {
        throw std::invalid_argument(
            "OpenMP benchmark: invalid OpenMP options"
        );
    }

    for (double value : exact_solution) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "OpenMP benchmark: nonfinite exact solution"
            );
        }
    }

    // Resolve default thread count to the runtime maximum.
    const int threads = omp_options.num_threads == 0
        ? omp_get_max_threads()
        : omp_options.num_threads;

    using Clock = std::chrono::steady_clock;


    // All warm-up solves are excluded from measurements.
    // Use identical warm-up methodology for serial and OpenMP.
    for (std::size_t warmup_index = 0;
         warmup_index < warmups;
         ++warmup_index) {

        std::cout
            << "[OpenMP CG] Matrix: " << matrix_name
            << " | threads=" << threads
            << " | Warm-up " << (warmup_index + 1)
            << "/" << warmups
            << " started..."
            << std::endl;

        const auto warmup = solve_openmp_cg(
            A, b, options, omp_options
        );

        if (!warmup.converged) {
            throw std::runtime_error(
                "OpenMP benchmark: warm-up did not converge"
            );
        }

        std::cout
            << "[OpenMP CG] Warm-up "
            << (warmup_index + 1)
            << "/" << warmups
            << " completed"
            << " | iterations=" << warmup.iterations
            << " | residual=" << warmup.relative_residual
            << std::endl;
    }
    std::vector<OpenMPBenchmarkRecord> records;
    records.reserve(repeats);

    for (std::size_t run = 0; run < repeats; ++run) {

        // Progress output is outside the timed section.
        std::cout
            << "[OpenMP CG] Measured run "
            << (run + 1) << "/" << repeats
            << " started..."
            << std::endl;

        // Only the complete OpenMP CG solver is timed.
        const auto start = Clock::now();

        auto result = solve_openmp_cg(
            A, b, options, omp_options
        );

        const auto stop = Clock::now();

        const double elapsed_seconds =
            std::chrono::duration<double>(
                stop - start
            ).count();

        // Numerical checks and record creation are untimed.
        if (!result.converged) {
            throw std::runtime_error(
                "OpenMP benchmark: measured solve did not converge"
            );
        }

        const double solution_error =
            compute_solution_error(
                result.x, exact_solution
            );

        if (!std::isfinite(solution_error) ||
            !std::isfinite(result.relative_residual)) {
            throw std::runtime_error(
                "OpenMP benchmark: nonfinite numerical result"
            );
        }

        OpenMPBenchmarkRecord record;

        record.matrix_name = matrix_name;
        record.n = A.rows;
        record.nnz = A.nnz();
        record.run_index = run + 1;

        record.threads = threads;
        record.schedule = omp_options.schedule;
        record.chunk_size = omp_options.chunk_size;

        record.iterations = result.iterations;
        record.converged = result.converged;

        record.solve_seconds = elapsed_seconds;
        record.true_relative_residual =
            result.relative_residual;
        record.relative_solution_error =
            solution_error;

        records.push_back(record);

        std::cout
            << "[OpenMP CG] Measured run "
            << (run + 1) << "/" << repeats
            << " completed"
            << " | time=" << elapsed_seconds << " s"
            << " | iterations=" << result.iterations
            << " | residual=" << result.relative_residual
            << " | solution_error=" << solution_error
            << std::endl;
    }

    return records;
}

} // namespace cg
