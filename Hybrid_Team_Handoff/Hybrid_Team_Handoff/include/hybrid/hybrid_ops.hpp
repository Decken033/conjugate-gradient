#pragma once

#include "mpi/mpi_partition.hpp"
#include "openmp/openmp_ops.hpp"

#include <mpi.h>
#include <vector>

namespace cg {

// Hybrid distributed SpMV: each rank gathers the full input vector with MPI,
// then uses OpenMP to compute its locally owned CSR rows.
std::vector<double> hybrid_csr_matvec(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& x_local,
    const OpenMPOptions& omp_options = {},
    MPI_Comm comm = MPI_COMM_WORLD
);

// OpenMP reduction within each rank, followed by MPI_Allreduce across ranks.
double hybrid_dot(
    const std::vector<double>& a_local,
    const std::vector<double>& b_local,
    const OpenMPOptions& omp_options = {},
    MPI_Comm comm = MPI_COMM_WORLD
);

double hybrid_norm2(
    const std::vector<double>& v_local,
    const OpenMPOptions& omp_options = {},
    MPI_Comm comm = MPI_COMM_WORLD
);

// Rank-local elementwise operations, parallelized with OpenMP.
void hybrid_axpy(
    std::vector<double>& y_local,
    const std::vector<double>& x_local,
    double alpha,
    const OpenMPOptions& omp_options = {}
);

void hybrid_update_solution_residual(
    std::vector<double>& x_local,
    std::vector<double>& r_local,
    const std::vector<double>& p_local,
    const std::vector<double>& Ap_local,
    double alpha,
    const OpenMPOptions& omp_options = {}
);

void hybrid_update_direction(
    std::vector<double>& p_local,
    const std::vector<double>& r_local,
    double beta,
    const OpenMPOptions& omp_options = {}
);

std::vector<double> hybrid_true_residual(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const std::vector<double>& x_local,
    const OpenMPOptions& omp_options = {},
    MPI_Comm comm = MPI_COMM_WORLD
);

} // namespace cg
