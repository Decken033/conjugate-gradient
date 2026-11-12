
#pragma once

#include "cg_solver.hpp"
#include "mpi_partition.hpp"
#include "mpi_ops.hpp"

#include <mpi.h>
#include <vector>

namespace cg {

// Results from a distributed MPI CG solve.
struct MPICGResult {
    // Local portion of the solution vector.
    // Concatenating x_local from all ranks (in rank order)
    // gives the full global solution.
    std::vector<double> x_local;

    bool converged = false;
    std::size_t iterations = 0;

    // True relative residual: ||b - Ax||_2 / ||b||_2
    // Computed in distributed fashion using MPI_Allreduce.
    double relative_residual = 0.0;
};

// Solve Ax = b using distributed MPI Conjugate Gradient.
//
// Uses the same mathematical algorithm as solve_serial_cg:
//   - Initial guess x0 = 0
//   - Double-precision arithmetic
//   - True relative residual convergence criterion
//   - Residual drift verification with restart
//   - Numerical breakdown checks
//
// Key differences from serial CG:
//   - SpMV uses MPI_Allgatherv + local computation
//   - Dot products use MPI_Allreduce (SUM)
//   - Convergence/breakdown decisions broadcast to all ranks
//   - True residual computed in distributed manner
//
// Input:
//   A:       Distributed matrix (from partition_matrix)
//   b_local: This rank's portion of the RHS vector
//   options: Tolerance and iteration limit (same as serial)
//   comm:    MPI communicator
//
// Collective: all ranks must call with consistent inputs.
//
// Throws:
//   - If A dimensions are inconsistent
//   - If b_local size doesn't match A.local_rows()
//   - On numerical breakdown (all ranks throw)
//
// Small differences in iteration count vs serial CG are expected
// due to floating-point reduction order in MPI_Allreduce.
MPICGResult solve_mpi_cg(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const CGOptions& options = {},
    MPI_Comm comm = MPI_COMM_WORLD
);

} // namespace cg
