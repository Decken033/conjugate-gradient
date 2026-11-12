#include "hybrid/hybrid_solver.hpp"
#include "common/ops_profiler.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace cg {
namespace {

void collective_check(int local_error, MPI_Comm comm, const char* message) {
    ProfileScope timer(ProfileSection::CollectiveCheck);
    int global_error = 0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX, comm);
    if (global_error)
        throw std::runtime_error(message);
}

} // namespace

HybridCGResult solve_hybrid_cg(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const CGOptions& options,
    const OpenMPOptions& omp_options,
    MPI_Comm comm
) {
    int local_error = 0;
    if (b_local.size() != A.local_rows() || A.global_rows != A.global_cols ||
        !std::isfinite(options.relative_tolerance) || options.relative_tolerance <= 0.0 ||
        omp_options.num_threads < 0 || omp_options.chunk_size < 0) {
        local_error = 1;
    }
    switch (omp_options.schedule) {
        case OpenMPSchedule::Static:
        case OpenMPSchedule::Dynamic:
        case OpenMPSchedule::Guided:
            break;
        default:
            local_error = 1;
    }
    collective_check(local_error, comm, "Hybrid CG: invalid dimensions or options");

    local_error = 0;
    for (double value : b_local) {
        if (!std::isfinite(value)) {
            local_error = 1;
            break;
        }
    }
    collective_check(local_error, comm, "Hybrid CG: nonfinite RHS value");

    // V1: allocate communication metadata and SpMV buffers once per solve.
    // The workspace is reused by every iteration and true-residual check.
    HybridWorkspace workspace;
    {
        ProfileScope workspace_timer(ProfileSection::WorkspaceInit);
        workspace = make_hybrid_workspace(A, comm);
    }

    HybridCGResult result;
    result.x_local.assign(A.local_rows(), 0.0);
    std::vector<double> r_local = b_local;
    std::vector<double> p_local = r_local;

    const double b_norm = hybrid_norm2(b_local, omp_options, comm);
    collective_check(!std::isfinite(b_norm), comm, "Hybrid CG: invalid RHS norm");
    if (b_norm == 0.0) {
        result.converged = true;
        result.relative_residual = 0.0;
        return result;
    }

    double rr = hybrid_dot(r_local, r_local, omp_options, comm);
    result.relative_residual = 1.0;

    for (std::size_t k = 0; k < options.max_iterations; ++k) {
        const auto& Ap_local = hybrid_csr_matvec(A, p_local, workspace, omp_options, comm);
        const double pAp = hybrid_dot(p_local, Ap_local, omp_options, comm);
        collective_check((!std::isfinite(pAp) || pAp <= 0.0), comm,
                         "Hybrid CG breakdown: invalid p^T A p");

        const double alpha = rr / pAp;
        collective_check(!std::isfinite(alpha), comm,
                         "Hybrid CG breakdown: invalid alpha");

        hybrid_update_solution_residual(
            result.x_local, r_local, p_local, Ap_local, alpha, omp_options);
        result.iterations = k + 1;

        const double rr_new = hybrid_dot(r_local, r_local, omp_options, comm);
        collective_check((!std::isfinite(rr_new) || rr_new < 0.0), comm,
                         "Hybrid CG breakdown: invalid recursive residual");

        const double recursive_relative = std::sqrt(rr_new) / b_norm;
        collective_check(!std::isfinite(recursive_relative), comm,
                         "Hybrid CG breakdown: invalid relative residual");

        if (recursive_relative <= options.relative_tolerance) {
            auto true_r_local = hybrid_true_residual(
                A, b_local, result.x_local, workspace, omp_options, comm);
            const double verified_relative =
                hybrid_norm2(true_r_local, omp_options, comm) / b_norm;
            collective_check(!std::isfinite(verified_relative), comm,
                             "Hybrid CG breakdown: invalid true residual");
            result.relative_residual = verified_relative;

            if (verified_relative <= options.relative_tolerance) {
                result.converged = true;
                return result;
            }

            // Restart from the explicitly recomputed residual.
            r_local = std::move(true_r_local);
            p_local = r_local;
            rr = hybrid_dot(r_local, r_local, omp_options, comm);
            continue;
        }

        const double beta = rr_new / rr;
        collective_check(!std::isfinite(beta), comm,
                         "Hybrid CG breakdown: invalid beta");
        hybrid_update_direction(p_local, r_local, beta, omp_options);
        rr = rr_new;
    }

    const auto final_r_local = hybrid_true_residual(
        A, b_local, result.x_local, workspace, omp_options, comm);
    result.relative_residual =
        hybrid_norm2(final_r_local, omp_options, comm) / b_norm;
    collective_check(!std::isfinite(result.relative_residual), comm,
                     "Hybrid CG breakdown: invalid final true residual");
    result.converged = result.relative_residual <= options.relative_tolerance;
    return result;
}

} // namespace cg
