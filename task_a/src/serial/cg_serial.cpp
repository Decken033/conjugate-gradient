
/*
 * Serial Conjugate Gradient solver for sparse SPD systems.
 *
 * References:
 * [1] Hestenes, M. R. and Stiefel, E. (1952).
 *     Methods of Conjugate Gradients for Solving Linear Systems.
 *     Journal of Research of the National Bureau of Standards,
 *     49(6), 409-436.
 *
 * [2] Barrett, R. et al. (1994).
 *     Templates for the Solution of Linear Systems:
 *     Building Blocks for Iterative Methods. SIAM.
 *
 * [3] Shewchuk, J. R. (1994).
 *     An Introduction to the Conjugate Gradient Method
 *     Without the Agonizing Pain.
 *
 * Classical unpreconditioned CG with residual verification.
 * The implementation is independent, based on the
 * mathematical recurrences in the references above.
 *
 * A must be symmetric positive definite. CSR validation
 * checks storage structure, not mathematical SPD.
 */

#include "cg_solver.hpp"
#include "csr_ops.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <utility>

namespace cg {
namespace {

// Serial inner product, used for r^T r and p^T A p.
double dot(const std::vector<double>& a,
           const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Euclidean norm. This simple implementation assumes the
// benchmark inputs are within a numerically safe range.
double norm2(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

// Recompute the true residual b - A*x.
// This is intentionally separate from the recursive CG update.
std::vector<double> true_residual(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const std::vector<double>& x
) {
    const auto Ax = csr_matvec(A, x);
    std::vector<double> r(b.size());

    for (std::size_t i = 0; i < b.size(); ++i) {
        r[i] = b[i] - Ax[i];
    }

    return r;
}

} // namespace

CGResult solve_serial_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const CGOptions& options
) {
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

    for (double value : b) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("CG: nonfinite RHS value");
        }
    }

    CGResult result;
    result.x.assign(A.rows, 0.0);

    // Since x0 = 0, r0 = b - A*x0 = b.
    std::vector<double> r = b;
    std::vector<double> p = r;

    const double b_norm = norm2(b);

    if (!std::isfinite(b_norm)) {
        throw std::invalid_argument("CG: invalid RHS norm");
    }

    // For b = 0, x = 0 is already an exact solution.
    if (b_norm == 0.0) {
        result.converged = true;
        result.relative_residual = 0.0;
        return result;
    }

    double rr = dot(r, r);

    // With max_iterations = 0, report the initial true residual.
    result.relative_residual = 1.0;

    // References [1]-[3]: classical CG iteration.
    for (std::size_t k = 0; k < options.max_iterations; ++k) {

        // One SpMV per normal iteration: q_k = A*p_k.
        const auto Ap = csr_matvec(A, p);
        const double pAp = dot(p, Ap);

        // Positive definiteness requires p^T A p > 0.
        if (!std::isfinite(pAp) || pAp <= 0.0) {
            throw std::runtime_error(
                "CG breakdown: invalid p^T A p"
            );
        }

        const double alpha = rr / pAp;

        if (!std::isfinite(alpha)) {
            throw std::runtime_error("CG breakdown: invalid alpha");
        }

        // x_{k+1} = x_k + alpha*p_k
        // r_{k+1} = r_k - alpha*A*p_k
        for (std::size_t i = 0; i < A.rows; ++i) {
            result.x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        result.iterations = k + 1;

        const double rr_new = dot(r, r);

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

        // Verify convergence using the actual b - A*x.
        // Avoid a second SpMV on ordinary iterations.
        if (recursive_relative <= options.relative_tolerance) {
            auto verified_r = true_residual(A, b, result.x);

            const double verified_relative =
                norm2(verified_r) / b_norm;

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

            // Recursive residual drifted from the true residual.
            // Restart the search direction from the true residual.
            // This is a robustness measure, not a classical CG step.
            r = std::move(verified_r);
            p = r;
            rr = dot(r, r);
            continue;
        }

        // Standard CG beta recurrence.
        const double beta = rr_new / rr;

        if (!std::isfinite(beta)) {
            throw std::runtime_error("CG breakdown: invalid beta");
        }

        for (std::size_t i = 0; i < A.rows; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rr = rr_new;
    }

    // Always report the true residual if the iteration limit
    // is reached, including max_iterations = 0.
    const auto final_r = true_residual(A, b, result.x);
    result.relative_residual = norm2(final_r) / b_norm;

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
