
#pragma once

#include "mpi_partition.hpp"
#include "csr_matrix.hpp"

#include <mpi.h>
#include <vector>

namespace cg {

// Distributed sparse matrix-vector multiplication: y_local = A * x
//
// Input:
//   A: partitioned matrix (each rank owns subset of rows)
//   x_local: this rank's portion of the input vector
//
// Output:
//   y_local: this rank's portion of the output (size = A.local_rows())
//
// Communication:
//   Performs MPI_Allgatherv to replicate full x vector on all ranks,
//   then computes local SpMV using standard CSR algorithm.
//
// Design rationale:
//   Simple correctness-first implementation. Each rank needs access
//   to all x elements because A's sparsity pattern is arbitrary.
//   Future optimization: sparse halo exchange for structured matrices.
//
// Memory: O(N) per rank for gathered x vector.
// Communication: O(N) per rank (allgather of full vector).
//
// Collective: all ranks must call.
std::vector<double> mpi_csr_matvec(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& x_local,
    MPI_Comm comm = MPI_COMM_WORLD
);

// Distributed dot product: sum = a_local^T * b_local
//
// Each rank computes partial sum from its local portion,
// then MPI_Allreduce sums across all ranks.
//
// Note: Floating-point summation order differs from serial,
// potentially causing minor numerical differences in CG convergence.
//
// Collective: all ranks must call with matching local sizes.
double mpi_dot(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    MPI_Comm comm = MPI_COMM_WORLD
);

// Distributed Euclidean norm: ||v||_2
//
// Computes sqrt(sum of local v_i^2 across all ranks).
// Uses mpi_dot internally.
//
// Collective: all ranks must call.
double mpi_norm2(
    const std::vector<double>& v_local,
    MPI_Comm comm = MPI_COMM_WORLD
);

// Local vector operation: y = y + alpha * x
// No MPI communication; operates on local portions only.
void mpi_axpy(
    std::vector<double>& y_local,
    const std::vector<double>& x_local,
    double alpha
);

// Local CG solution and residual update:
//   x_local = x_local + alpha * p_local
//   r_local = r_local - alpha * Ap_local
//
// No MPI communication; operates on local portions only.
void mpi_update_solution_residual(
    std::vector<double>& x_local,
    std::vector<double>& r_local,
    const std::vector<double>& p_local,
    const std::vector<double>& Ap_local,
    double alpha
);

// Local CG direction update:
//   p_local = r_local + beta * p_local
//
// No MPI communication; operates on local portions only.
void mpi_update_direction(
    std::vector<double>& p_local,
    const std::vector<double>& r_local,
    double beta
);

// Compute distributed true residual: r_true_local = b_local - A * x
//
// Uses mpi_csr_matvec internally to compute A*x in distributed manner,
// then subtracts from b_local.
//
// Returns local portion of true residual.
//
// Collective: all ranks must call.
std::vector<double> mpi_true_residual(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const std::vector<double>& x_local,
    MPI_Comm comm = MPI_COMM_WORLD
);

} // namespace cg
