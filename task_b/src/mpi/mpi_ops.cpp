
#include "mpi_ops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cstddef>

namespace cg {

std::vector<double> mpi_csr_matvec(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& x_local,
    MPI_Comm comm
) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    // For square matrices (CG requirement), global_rows == global_cols
    // and vectors are partitioned by rows.
    if (A.global_rows != A.global_cols) {
        throw std::invalid_argument(
            "MPI SpMV: only square matrices are supported (CG requirement)"
        );
    }

    if (x_local.size() != A.local_rows()) {
        throw std::invalid_argument(
            "MPI SpMV: x_local size doesn't match local rows"
        );
    }

    // Gather full x vector on all ranks using Allgatherv.
    //
    // Communication cost: O(N) per rank.
    // Memory cost: O(N) per rank.
    //
    // This is correct for arbitrary sparsity patterns.
    // Future optimization: sparse halo exchange when only
    // a subset of x entries is needed per rank.
    //
    // Build displacement and count arrays
    std::vector<int> recvcounts(num_ranks);
    std::vector<int> displs(num_ranks);

    for (int r = 0; r < num_ranks; ++r) {
        const std::size_t r_rows =
            A.partition_offsets[r + 1] - A.partition_offsets[r];
        recvcounts[r] = static_cast<int>(r_rows);
        if (A.partition_offsets[r] > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "MPI SpMV: partition offset exceeds int limit for Allgatherv"
            );
        }
        displs[r] = static_cast<int>(A.partition_offsets[r]);
    }

    // Replicate full x on all ranks
    // For square matrices: global_rows == global_cols
    std::vector<double> x_global(A.global_rows);

    MPI_Allgatherv(
        x_local.data(), static_cast<int>(x_local.size()), MPI_DOUBLE,
        x_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
        comm
    );

    // Compute local portion of y = A * x
    // Uses global column indices from local CSR
    const std::size_t n_local = A.local_rows();
    std::vector<double> y_local(n_local, 0.0);

    const auto& mat = A.local_matrix;

    for (std::size_t i = 0; i < n_local; ++i) {
        double sum = 0.0;
        const std::size_t row_begin = mat.row_ptr[i];
        const std::size_t row_end = mat.row_ptr[i + 1];

        for (std::size_t j = row_begin; j < row_end; ++j) {
            // col_idx[j] is a global column index
            sum += mat.values[j] * x_global[mat.col_idx[j]];
        }

        y_local[i] = sum;
    }

    return y_local;
}

double mpi_dot(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm
) {
    if (a_local.size() != b_local.size()) {
        throw std::invalid_argument(
            "MPI dot: local vector size mismatch"
        );
    }

    // Local partial sum
    double local_sum = 0.0;
    for (std::size_t i = 0; i < a_local.size(); ++i) {
        local_sum += a_local[i] * b_local[i];
    }

    // Global sum via Allreduce (all ranks receive the result)
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    return global_sum;
}

double mpi_norm2(
    const std::vector<double>& v_local,
    MPI_Comm comm
) {
    return std::sqrt(mpi_dot(v_local, v_local, comm));
}

void mpi_axpy(
    std::vector<double>& y_local,
    const std::vector<double>& x_local,
    double alpha
) {
    if (y_local.size() != x_local.size()) {
        throw std::invalid_argument(
            "mpi_axpy: vector size mismatch"
        );
    }
    for (std::size_t i = 0; i < y_local.size(); ++i) {
        y_local[i] += alpha * x_local[i];
    }
}

void mpi_update_solution_residual(
    std::vector<double>& x_local,
    std::vector<double>& r_local,
    const std::vector<double>& p_local,
    const std::vector<double>& Ap_local,
    double alpha
) {
    if (x_local.size() != r_local.size() ||
        x_local.size() != p_local.size() ||
        x_local.size() != Ap_local.size()) {
        throw std::invalid_argument(
            "mpi_update_solution_residual: vector size mismatch"
        );
    }

    for (std::size_t i = 0; i < x_local.size(); ++i) {
        x_local[i] += alpha * p_local[i];
        r_local[i] -= alpha * Ap_local[i];
    }
}

void mpi_update_direction(
    std::vector<double>& p_local,
    const std::vector<double>& r_local,
    double beta
) {
    if (p_local.size() != r_local.size()) {
        throw std::invalid_argument(
            "mpi_update_direction: vector size mismatch"
        );
    }

    for (std::size_t i = 0; i < p_local.size(); ++i) {
        p_local[i] = r_local[i] + beta * p_local[i];
    }
}

std::vector<double> mpi_true_residual(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const std::vector<double>& x_local,
    MPI_Comm comm
) {
    // Compute Ax in distributed manner
    std::vector<double> Ax_local = mpi_csr_matvec(A, x_local, comm);

    // Compute r_true_local = b_local - Ax_local
    std::vector<double> r_local(b_local.size());
    for (std::size_t i = 0; i < b_local.size(); ++i) {
        r_local[i] = b_local[i] - Ax_local[i];
    }

    return r_local;
}

} // namespace cg
