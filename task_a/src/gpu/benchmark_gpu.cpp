#include "gpu_benchmark.hpp"
#include "gpu_solver.hpp"
#include "csr_ops.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cg {
namespace {

double stable_norm(const std::vector<double>& v) {
    double norm = 0.0;
    for (double value : v) {
        norm = std::hypot(norm, value);
    }
    return norm;
}

struct Accuracy {
    double true_residual;
    double solution_error;
};

Accuracy validate_solution(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const std::vector<double>& exact,
    const CGResult& result
) {
    if (result.x.size() != A.rows) {
        throw std::runtime_error(
            "GPU benchmark: solution dimension mismatch"
        );
    }

    const auto Ax = csr_matvec(A, result.x);

    std::vector<double> residual(A.rows);
    std::vector<double> error(A.rows);

    for (std::size_t i = 0; i < A.rows; ++i) {
        residual[i] = b[i] - Ax[i];
        error[i] = result.x[i] - exact[i];
    }

    const double rhs_norm = stable_norm(b);
    const double exact_norm = stable_norm(exact);

    const double true_residual =
        rhs_norm > 0.0
            ? stable_norm(residual) / rhs_norm
            : stable_norm(residual);

    const double solution_error =
        exact_norm > 0.0
            ? stable_norm(error) / exact_norm
            : stable_norm(error);

    if (!std::isfinite(true_residual) ||
        !std::isfinite(solution_error)) {
        throw std::runtime_error(
            "GPU benchmark: nonfinite accuracy metric"
        );
    }

    return {true_residual, solution_error};
}

} // namespace

std::vector<GPUBenchmarkRecord> benchmark_gpu_cg(
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
            "GPU benchmark: input dimension mismatch"
        );
    }

    if (repeats == 0) {
        throw std::invalid_argument(
            "GPU benchmark: repeats must be positive"
        );
    }

    for (double value : exact_solution) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "GPU benchmark: nonfinite exact solution"
            );
        }
    }

    using Clock = std::chrono::steady_clock;

    // Warmups are excluded from performance measurements.
    for (std::size_t i = 0; i < warmups; ++i) {
        std::cout
            << "[GPU CG] " << matrix_name
            << " | Warm-up " << (i + 1)
            << "/" << warmups
            << " started..."
            << std::endl;

        const auto result = solve_gpu_cg(A, b, options);

        if (!result.converged) {
            throw std::runtime_error(
                "GPU benchmark: warm-up did not converge"
            );
        }

        const auto accuracy =
            validate_solution(A, b, exact_solution, result);

        if (accuracy.true_residual > options.relative_tolerance) {
            throw std::runtime_error(
                "GPU benchmark: warm-up residual exceeds tolerance"
            );
        }
    }

    std::vector<GPUBenchmarkRecord> records;
    records.reserve(repeats);

    for (std::size_t run = 0; run < repeats; ++run) {
        std::cout
            << "[GPU CG] " << matrix_name
            << " | Measured run " << (run + 1)
            << "/" << repeats
            << " started..."
            << std::endl;

        const auto start = Clock::now();
        const auto result = solve_gpu_cg(A, b, options);
        const auto stop = Clock::now();

        const double seconds =
            std::chrono::duration<double>(stop - start).count();

        // Independent CPU verification occurs outside the timed region.
        // The solver's own input and convergence checks remain timed.
        if (!result.converged) {
            throw std::runtime_error(
                "GPU benchmark: measured solve did not converge"
            );
        }

        const auto accuracy =
            validate_solution(A, b, exact_solution, result);

        if (accuracy.true_residual > options.relative_tolerance ||
            !std::isfinite(seconds) ||
            seconds < 0.0) {
            throw std::runtime_error(
                "GPU benchmark: measured run failed validation"
            );
        }

        GPUBenchmarkRecord record;
        record.matrix_name = matrix_name;
        record.n = A.rows;
        record.nnz = A.nnz();
        record.run_index = run + 1;
        record.iterations = result.iterations;
        record.converged = result.converged;
        record.solve_seconds = seconds;
        record.true_relative_residual = accuracy.true_residual;
        record.relative_solution_error = accuracy.solution_error;

        records.push_back(record);

        std::cout
            << "[GPU CG] Run " << record.run_index
            << " completed"
            << " | time=" << seconds << " s"
            << " | iterations=" << record.iterations
            << " | residual=" << record.true_relative_residual
            << " | solution_error=" << record.relative_solution_error
            << std::endl;
    }

    return records;
}

} // namespace cg
