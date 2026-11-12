
/*
 * CPU OpenMP Conjugate Gradient solver for sparse SPD systems.
 *
 * Based on the same classical CG recurrence and numerical
 * safeguards as the Task 1 serial implementation.
 *
 * OpenMP parallelizes SpMV, dot products and vector updates.
 * The outer CG iteration remains sequential.
 *
 * References:
 * Hestenes and Stiefel (1952).
 * Barrett et al. (1994).
 * Shewchuk (1994).
 */

#include "openmp/openmp_solver.hpp"
#include "openmp/openmp_ops.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cg {
namespace {

// Euclidean norm using the parallel dot product.
double norm2_openmp(
    const std::vector<double>& v,
    const OpenMPOptions& omp_options
) {
    return std::sqrt(dot_openmp(v, v, omp_options));
}

// Recompute the actual residual b - A*x.
// This is separate from the recursive CG residual update.
std::vector<double> true_residual_openmp(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const std::vector<double>& x,
    const OpenMPOptions& omp_options
) {
    const auto Ax = csr_matvec_openmp(A, x, omp_options);
    return subtract_openmp(b, Ax, omp_options);
}

} // namespace

CGResult solve_openmp_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const CGOptions& options,
    const OpenMPOptions& omp_options
) {
    // Validate all inputs before parallel execution.
    A.validate();

    if (b.size() != A.rows) {
        throw std::invalid_argument("CG: RHS dimension mismatch");
    }

    if (!std::isfinite(options.relative_tolerance) ||
        options.relative_tolerance <= 0.0) {
        throw std::invalid_argument(
            "CG: tolerance must be positive and finite"
        );
    }

    if (omp_options.num_threads < 0 ||
        omp_options.chunk_size < 0) {
        throw std::invalid_argument(
            "OpenMP CG: invalid OpenMP options"
        );
    }

    switch (omp_options.schedule) {
        case OpenMPSchedule::Static:
        case OpenMPSchedule::Dynamic:
        case OpenMPSchedule::Guided:
            break;
        default:
            throw std::invalid_argument(
                "OpenMP CG: unsupported scheduling policy"
            );
    }

    for (double value : b) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("CG: nonfinite RHS value");
        }
    }

    CGResult result;
    result.x.assign(A.rows, 0.0);

    // Initial guess x0 = 0, so initial residual r0 = b.
    std::vector<double> r = b;
    std::vector<double> p = r;

    const double b_norm = norm2_openmp(b, omp_options);

    if (!std::isfinite(b_norm)) {
        throw std::invalid_argument("CG: invalid RHS norm");
    }

    // Zero RHS: x = 0 is already an exact solution.
    if (b_norm == 0.0) {
        result.converged = true;
        result.relative_residual = 0.0;
        return result;
    }

    double rr = dot_openmp(r, r, omp_options);

    // Initial true relative residual is 1 for x0 = 0.
    result.relative_residual = 1.0;

    // CG iterations remain sequential due to dependencies.
    for (std::size_t k = 0; k < options.max_iterations; ++k) {

        // Parallel sparse matrix-vector multiplication.
        const auto Ap = csr_matvec_openmp(A, p, omp_options);

        // Parallel reduction: p^T A p.
        const double pAp = dot_openmp(p, Ap, omp_options);

        if (!std::isfinite(pAp) || pAp <= 0.0) {
            throw std::runtime_error(
                "CG breakdown: invalid p^T A p"
            );
        }

        const double alpha = rr / pAp;

        if (!std::isfinite(alpha)) {
            throw std::runtime_error(
                "CG breakdown: invalid alpha"
            );
        }

        // Parallel update of solution and recursive residual.
        update_solution_residual_openmp(
            result.x, r, p, Ap, alpha, omp_options
        );

        result.iterations = k + 1;

        const double rr_new = dot_openmp(r, r, omp_options);

        if (!std::isfinite(rr_new) || rr_new < 0.0) {
            throw std::runtime_error(
                "CG breakdown: invalid recursive residual"
            );
        }

        const double recursive_relative =
            std::sqrt(rr_new) / b_norm;

        if (!std::isfinite(recursive_relative)) {
            throw std::runtime_error(
                "CG breakdown: invalid relative residual"
            );
        }

        // Verify convergence against the true residual.
        // Avoid an additional SpMV on ordinary iterations.
        if (recursive_relative <= options.relative_tolerance) {

            auto verified_r = true_residual_openmp(
                A, b, result.x, omp_options
            );

            const double verified_relative =
                norm2_openmp(verified_r, omp_options) / b_norm;

            if (!std::isfinite(verified_relative)) {
                throw std::runtime_error(
                    "CG breakdown: invalid true residual"
                );
            }

            result.relative_residual = verified_relative;

            if (verified_relative <= options.relative_tolerance) {
                result.converged = true;
                return result;
            }

            // Recursive residual has drifted.
            // Restart using the verified true residual.
            r = std::move(verified_r);
            p = r;
            rr = dot_openmp(r, r, omp_options);
            continue;
        }

        // Standard CG beta recurrence.
        const double beta = rr_new / rr;

        if (!std::isfinite(beta)) {
            throw std::runtime_error(
                "CG breakdown: invalid beta"
            );
        }

        // Parallel search-direction update.
        update_direction_openmp(p, r, beta, omp_options);

        rr = rr_new;
    }

    // If the iteration limit is reached, report true residual.
    // This also handles max_iterations = 0.
    const auto final_r = true_residual_openmp(
        A, b, result.x, omp_options
    );

    result.relative_residual =
        norm2_openmp(final_r, omp_options) / b_norm;

    if (!std::isfinite(result.relative_residual)) {
        throw std::runtime_error(
            "CG breakdown: invalid final true residual"
        );
    }

    result.converged =
        result.relative_residual <= options.relative_tolerance;

    return result;
}

} // namespace cg
