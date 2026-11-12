
/*
 * MPI Conjugate Gradient solver for distributed SPD systems.
 *
 * Implements the same mathematical recurrences as cg_serial.cpp
 * with distributed computation via MPI collectives.
 *
 * Key design decisions:
 *
 * 1. All ranks follow identical control flow.
 *    Convergence, breakdown, and restart decisions are broadcast
 *    to all ranks before taking action. This prevents deadlocks
 *    from rank-specific early exits on collective operations.
 *
 * 2. Distributed true residual verification.
 *    True residual b - A*x is computed in distributed manner
 *    using mpi_true_residual (SpMV + local subtraction + Allreduce
 *    for norm). No serial gather-and-compute on rank 0.
 *
 * 3. Floating-point reduction order.
 *    MPI_Allreduce with MPI_SUM accumulates partial dot products
 *    in a different order than serial CG. This may cause small
 *    differences in iteration count vs serial, which are expected
 *    and acceptable. Both implementations converge to the same
 *    tolerance.
 *
 * 4. Residual restart semantics.
 *    Preserved from serial CG: if the recursive residual appears
 *    converged but true residual doesn't satisfy tolerance,
 *    restart from the true residual. All ranks take identical path.
 */

#include "mpi_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cg {
namespace {

// Collective error indicator: 0 = ok, 1 = error.
// Any rank that encounters an error sets value to 1.
// All ranks receive the max, so all see the error.
void collective_check(int local_error, MPI_Comm comm,
                      const char* message)
{
    int global_error = 0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX, comm);
    if (global_error) {
        throw std::runtime_error(message);
    }
}

} // namespace

MPICGResult solve_mpi_cg(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const CGOptions& options,
    MPI_Comm comm
) {
    // Validate inputs collectively
    {
        int local_error = 0;

        if (b_local.size() != A.local_rows()) {
            local_error = 1;
        }

        if (!std::isfinite(options.relative_tolerance) ||
            options.relative_tolerance <= 0.0) {
            local_error = 1;
        }

        if (A.global_rows != A.global_cols) {
            local_error = 1;
        }

        collective_check(local_error, comm,
            "MPI CG: invalid input dimensions or options"
        );
    }

    // Check RHS values are finite
    {
        int local_error = 0;
        for (double v : b_local) {
            if (!std::isfinite(v)) {
                local_error = 1;
                break;
            }
        }
        collective_check(local_error, comm,
            "MPI CG: nonfinite RHS value"
        );
    }

    MPICGResult result;
    result.x_local.assign(A.local_rows(), 0.0);

    // Since x0 = 0, r0 = b - A*x0 = b
    std::vector<double> r_local = b_local;
    std::vector<double> p_local = r_local;

    // Compute global RHS norm (distributed)
    const double b_norm = mpi_norm2(b_local, comm);

    if (!std::isfinite(b_norm)) {
        throw std::runtime_error("MPI CG: invalid RHS norm");
    }

    // b = 0: x = 0 is exact solution
    if (b_norm == 0.0) {
        result.converged = true;
        result.relative_residual = 0.0;
        return result;
    }

    // Initial recursive residual norm squared (distributed dot)
    double rr = mpi_dot(r_local, r_local, comm);

    result.relative_residual = 1.0;

    for (std::size_t k = 0; k < options.max_iterations; ++k) {

        // q_k = A * p_k (one distributed SpMV per iteration)
        // mpi_csr_matvec performs Allgatherv internally
        std::vector<double> Ap_local = mpi_csr_matvec(A, p_local, comm);

        // p^T A p (global dot product via Allreduce)
        const double pAp = mpi_dot(p_local, Ap_local, comm);

        // Check breakdown: p^T A p must be positive
        {
            int local_bad = (!std::isfinite(pAp) || pAp <= 0.0) ? 1 : 0;
            collective_check(local_bad, comm,
                "MPI CG breakdown: invalid p^T A p"
            );
        }

        const double alpha = rr / pAp;

        {
            int local_bad = !std::isfinite(alpha) ? 1 : 0;
            collective_check(local_bad, comm,
                "MPI CG breakdown: invalid alpha"
            );
        }

        // x_{k+1} = x_k + alpha * p_k
        // r_{k+1} = r_k - alpha * A*p_k
        mpi_update_solution_residual(
            result.x_local, r_local, p_local, Ap_local, alpha
        );

        result.iterations = k + 1;

        const double rr_new = mpi_dot(r_local, r_local, comm);

        {
            int local_bad = (!std::isfinite(rr_new) || rr_new < 0.0) ? 1 : 0;
            collective_check(local_bad, comm,
                "MPI CG breakdown: invalid recursive residual"
            );
        }

        const double recursive_relative = std::sqrt(rr_new) / b_norm;

        // Check convergence using recursive residual first
        // If it looks converged, verify with the true residual
        if (recursive_relative <= options.relative_tolerance) {

            // Compute true residual: r_true = b - A*x (distributed)
            std::vector<double> r_true_local = mpi_true_residual(
                A, b_local, result.x_local, comm
            );

            // Compute global true residual norm (Allreduce)
            const double true_norm = mpi_norm2(r_true_local, comm);
            const double verified_relative = true_norm / b_norm;

            {
                int local_bad = !std::isfinite(verified_relative) ? 1 : 0;
                collective_check(local_bad, comm,
                    "MPI CG breakdown: invalid true residual"
                );
            }

            result.relative_residual = verified_relative;

            // All ranks check convergence with same globally computed value
            if (verified_relative <= options.relative_tolerance) {
                result.converged = true;
                return result;
            }

            // Recursive residual has drifted from true residual.
            // Restart from true residual (matching serial CG behavior).
            // All ranks execute this path identically.
            r_local = std::move(r_true_local);
            p_local = r_local;
            rr = mpi_dot(r_local, r_local, comm);
            continue;
        }

        const double beta = rr_new / rr;

        {
            int local_bad = !std::isfinite(beta) ? 1 : 0;
            collective_check(local_bad, comm,
                "MPI CG breakdown: invalid beta"
            );
        }

        mpi_update_direction(p_local, r_local, beta);
        rr = rr_new;
    }

    // Iteration limit reached: compute true residual for report
    std::vector<double> final_r_local = mpi_true_residual(
        A, b_local, result.x_local, comm
    );
    const double final_norm = mpi_norm2(final_r_local, comm);
    result.relative_residual = final_norm / b_norm;

    if (!std::isfinite(result.relative_residual)) {
        throw std::runtime_error(
            "MPI CG breakdown: invalid final true residual"
        );
    }

    result.converged =
        result.relative_residual <= options.relative_tolerance;

    return result;
}

} // namespace cg
