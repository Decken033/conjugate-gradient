
/*
 * Serial CG benchmark timing implementation.
 *
 * Timing methodology:
 * - One untimed warm-up solve.
 * - Repeated timed solves with identical inputs.
 * - Measure solver-only elapsed wall-clock time.
 * - Calculate solution accuracy outside the timed region.
 * - Keep individual run records for later CSV reporting.
 *
 * Reference:
 * Barrett et al. (1994), Templates for the Solution
 * of Linear Systems, SIAM, for CG convergence concepts.
 *
 * Wall-clock timing uses C++17 std::chrono::steady_clock.
 */

#include "serial_benchmark.hpp"

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
//     ||x - x_exact||_2 / ||x_exact||_2
//
// For a zero exact solution, return absolute error norm.
double compute_solution_error(
    const std::vector<double>& x,
    const std::vector<double>& exact_solution
) {
    if (x.size() != exact_solution.size()) {
        throw std::invalid_argument(
            "Benchmark: solution dimension mismatch"
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


std::vector<SerialBenchmarkRecord> benchmark_serial_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const std::vector<double>& exact_solution,
    const std::string& matrix_name,
    std::size_t repeats,
    const CGOptions& options,
    std::size_t warmups
) {
    A.validate();

    if (b.size() != A.rows ||
        exact_solution.size() != A.rows) {
        throw std::invalid_argument(
            "Benchmark: input dimension mismatch"
        );
    }

    if (repeats == 0) {
        throw std::invalid_argument(
            "Benchmark: repeats must be at least 1"
        );
    }

    for (double value : exact_solution) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Benchmark: nonfinite exact solution"
            );
        }
    }
    using Clock = std::chrono::steady_clock;

    // All warm-up solves are excluded from measurements.
    // Configurable warm-ups help stabilize performance
    // before collecting timing results.
    for (std::size_t warmup_index = 0;
         warmup_index < warmups;
         ++warmup_index) {

        std::cout
            << "[Serial CG] Matrix: " << matrix_name
            << " | Warm-up " << (warmup_index + 1)
            << "/" << warmups
            << " started..."
            << std::endl;

        const auto warmup = solve_serial_cg(A, b, options);

        if (!warmup.converged) {
            throw std::runtime_error(
                "Benchmark: warm-up CG solve did not converge"
            );
        }

        std::cout
            << "[Serial CG] Warm-up "
            << (warmup_index + 1)
            << "/" << warmups
            << " completed"
            << " | iterations=" << warmup.iterations
            << " | residual=" << warmup.relative_residual
            << std::endl;
    }
    std::vector<SerialBenchmarkRecord> records;
    records.reserve(repeats);

        for (std::size_t run = 0; run < repeats; ++run) {

        // Progress reporting is outside the timed region.
        std::cout
            << "[Serial CG] Measured run "
            << (run + 1) << "/" << repeats
            << " started..."
            << std::endl;

        // Benchmark only the serial CG solver.
        const auto start = Clock::now();

        auto result = solve_serial_cg(A, b, options);

        const auto stop = Clock::now();

        const double elapsed_seconds =
            std::chrono::duration<double>(
                stop - start
            ).count();

        // Everything below is outside the timed region.
        if (!result.converged) {
            throw std::runtime_error(
                "Benchmark: measured CG solve did not converge"
            );
        }

        const double solution_error =
            compute_solution_error(
                result.x,
                exact_solution
            );

        if (!std::isfinite(solution_error)) {
            throw std::runtime_error(
                "Benchmark: invalid solution error"
            );
        }

        SerialBenchmarkRecord record;

        record.matrix_name = matrix_name;
        record.n = A.rows;
        record.nnz = A.nnz();
        record.run_index = run + 1;

        record.iterations = result.iterations;
        record.converged = result.converged;

        record.solve_seconds = elapsed_seconds;
        record.true_relative_residual =
            result.relative_residual;

        record.relative_solution_error =
            solution_error;
        records.push_back(record);
        // Report results only after timing and validation.
        std::cout
            << "[Serial CG] Measured run "
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
