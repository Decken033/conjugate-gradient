#include "hybrid/hybrid_ops.hpp"
#include "common/ops_profiler.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <omp.h>

namespace cg {
namespace {

int resolve_threads(const OpenMPOptions& options) {
    if (options.num_threads < 0 || options.chunk_size < 0) {
        throw std::invalid_argument("Hybrid OpenMP: negative thread count or chunk size");
    }
    return options.num_threads == 0 ? omp_get_max_threads() : options.num_threads;
}

void validate_schedule(OpenMPSchedule schedule) {
    switch (schedule) {
        case OpenMPSchedule::Static:
        case OpenMPSchedule::Dynamic:
        case OpenMPSchedule::Guided:
            return;
        default:
            throw std::invalid_argument("Hybrid OpenMP: unsupported scheduling policy");
    }
}

} // namespace

const std::vector<double>& hybrid_csr_matvec(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& x_local,
    HybridWorkspace& workspace,
    const OpenMPOptions& omp_options,
    MPI_Comm comm
) {
    const int threads = resolve_threads(omp_options);
    validate_schedule(omp_options.schedule);

    int num_ranks = 0;
    MPI_Comm_size(comm, &num_ranks);

    if (A.global_rows != A.global_cols) {
        throw std::invalid_argument("Hybrid SpMV: CG requires a square matrix");
    }
    if (x_local.size() != A.local_rows()) {
        throw std::invalid_argument("Hybrid SpMV: x_local size does not match local rows");
    }
    if (A.partition_offsets.size() != static_cast<std::size_t>(num_ranks + 1) ||
        workspace.recvcounts.size() != static_cast<std::size_t>(num_ranks) ||
        workspace.displs.size() != static_cast<std::size_t>(num_ranks) ||
        workspace.x_global.size() != A.global_rows ||
        workspace.spmv_output.size() != A.local_rows()) {
        throw std::invalid_argument("Hybrid SpMV: workspace does not match matrix partition");
    }

    // Reuse the same global-vector buffer on every CG iteration.
    // MPI communication remains outside all OpenMP parallel regions.
    {
        ProfileScope allgatherv_timer(ProfileSection::SpmvAllgatherv);
        MPI_Allgatherv(
            x_local.data(), static_cast<int>(x_local.size()), MPI_DOUBLE,
            workspace.x_global.data(), workspace.recvcounts.data(),
            workspace.displs.data(), MPI_DOUBLE, comm
        );
    }

    const auto& mat = A.local_matrix;
    auto& y_local = workspace.spmv_output;
    const auto& x_global = workspace.x_global;
    const int chunk = omp_options.chunk_size;
    ProfileScope spmv_timer(ProfileSection::SpmvCompute);

    // Every local row is overwritten, so no per-call zero initialization is needed.
    switch (omp_options.schedule) {
        case OpenMPSchedule::Static:
            if (chunk == 0) {
                #pragma omp parallel for schedule(static) num_threads(threads)
                for (std::size_t i = 0; i < A.local_rows(); ++i) {
                    double sum = 0.0;
                    for (std::size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j)
                        sum += mat.values[j] * x_global[mat.col_idx[j]];
                    y_local[i] = sum;
                }
            } else {
                #pragma omp parallel for schedule(static, chunk) num_threads(threads)
                for (std::size_t i = 0; i < A.local_rows(); ++i) {
                    double sum = 0.0;
                    for (std::size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j)
                        sum += mat.values[j] * x_global[mat.col_idx[j]];
                    y_local[i] = sum;
                }
            }
            break;
        case OpenMPSchedule::Dynamic:
            if (chunk == 0) {
                #pragma omp parallel for schedule(dynamic) num_threads(threads)
                for (std::size_t i = 0; i < A.local_rows(); ++i) {
                    double sum = 0.0;
                    for (std::size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j)
                        sum += mat.values[j] * x_global[mat.col_idx[j]];
                    y_local[i] = sum;
                }
            } else {
                #pragma omp parallel for schedule(dynamic, chunk) num_threads(threads)
                for (std::size_t i = 0; i < A.local_rows(); ++i) {
                    double sum = 0.0;
                    for (std::size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j)
                        sum += mat.values[j] * x_global[mat.col_idx[j]];
                    y_local[i] = sum;
                }
            }
            break;
        case OpenMPSchedule::Guided:
            if (chunk == 0) {
                #pragma omp parallel for schedule(guided) num_threads(threads)
                for (std::size_t i = 0; i < A.local_rows(); ++i) {
                    double sum = 0.0;
                    for (std::size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j)
                        sum += mat.values[j] * x_global[mat.col_idx[j]];
                    y_local[i] = sum;
                }
            } else {
                #pragma omp parallel for schedule(guided, chunk) num_threads(threads)
                for (std::size_t i = 0; i < A.local_rows(); ++i) {
                    double sum = 0.0;
                    for (std::size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j)
                        sum += mat.values[j] * x_global[mat.col_idx[j]];
                    y_local[i] = sum;
                }
            }
            break;
    }
    return y_local;
}

std::vector<double> hybrid_csr_matvec(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& x_local,
    const OpenMPOptions& omp_options,
    MPI_Comm comm
) {
    auto workspace = make_hybrid_workspace(A, comm);
    const auto& result = hybrid_csr_matvec(A, x_local, workspace, omp_options, comm);
    return result;
}

double hybrid_dot(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    const OpenMPOptions& omp_options,
    MPI_Comm comm
) {
    const int threads = resolve_threads(omp_options);
    if (a_local.size() != b_local.size())
        throw std::invalid_argument("Hybrid dot: local vector size mismatch");

    double local_sum = 0.0;
    {
        ProfileScope dot_timer(ProfileSection::DotCompute);
        #pragma omp parallel for schedule(static) num_threads(threads) reduction(+:local_sum)
        for (std::size_t i = 0; i < a_local.size(); ++i)
            local_sum += a_local[i] * b_local[i];
    }

    // OpenMP reduction is complete before the master/calling thread enters MPI.
    double global_sum = 0.0;
    {
        ProfileScope allreduce_timer(ProfileSection::Allreduce);
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    }
    return global_sum;
}

double hybrid_norm2(
    const std::vector<double>& v_local,
    const OpenMPOptions& omp_options,
    MPI_Comm comm
) {
    return std::sqrt(hybrid_dot(v_local, v_local, omp_options, comm));
}

void hybrid_axpy(
    std::vector<double>& y_local,
    const std::vector<double>& x_local,
    double alpha,
    const OpenMPOptions& omp_options
) {
    const int threads = resolve_threads(omp_options);
    if (y_local.size() != x_local.size())
        throw std::invalid_argument("Hybrid AXPY: vector size mismatch");
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < y_local.size(); ++i)
        y_local[i] += alpha * x_local[i];
}

void hybrid_update_solution_residual(
    std::vector<double>& x_local,
    std::vector<double>& r_local,
    const std::vector<double>& p_local,
    const std::vector<double>& Ap_local,
    double alpha,
    const OpenMPOptions& omp_options
) {
    const int threads = resolve_threads(omp_options);
    const std::size_t n = x_local.size();
    if (r_local.size() != n || p_local.size() != n || Ap_local.size() != n)
        throw std::invalid_argument("Hybrid solution/residual update: vector size mismatch");
    ProfileScope update_timer(ProfileSection::SolutionResidualUpdate);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < n; ++i) {
        x_local[i] += alpha * p_local[i];
        r_local[i] -= alpha * Ap_local[i];
    }
}

void hybrid_update_direction(
    std::vector<double>& p_local,
    const std::vector<double>& r_local,
    double beta,
    const OpenMPOptions& omp_options
) {
    const int threads = resolve_threads(omp_options);
    if (p_local.size() != r_local.size())
        throw std::invalid_argument("Hybrid direction update: vector size mismatch");
    ProfileScope direction_timer(ProfileSection::DirectionUpdate);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < p_local.size(); ++i)
        p_local[i] = r_local[i] + beta * p_local[i];
}

std::vector<double> hybrid_true_residual(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const std::vector<double>& x_local,
    HybridWorkspace& workspace,
    const OpenMPOptions& omp_options,
    MPI_Comm comm
) {
    if (b_local.size() != A.local_rows())
        throw std::invalid_argument("Hybrid true residual: RHS size mismatch");
    const auto& Ax_local = hybrid_csr_matvec(A, x_local, workspace, omp_options, comm);
    const int threads = resolve_threads(omp_options);
    std::vector<double> residual(b_local.size(), 0.0);
    ProfileScope residual_timer(ProfileSection::ResidualSubtract);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < b_local.size(); ++i)
        residual[i] = b_local[i] - Ax_local[i];
    return residual;
}

std::vector<double> hybrid_true_residual(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const std::vector<double>& x_local,
    const OpenMPOptions& omp_options,
    MPI_Comm comm
) {
    auto workspace = make_hybrid_workspace(A, comm);
    return hybrid_true_residual(A, b_local, x_local, workspace, omp_options, comm);
}

} // namespace cg
