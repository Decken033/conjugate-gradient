#pragma once

#include "serial/cg_solver.hpp"
#include "hybrid/hybrid_ops.hpp"

#include <mpi.h>

namespace cg {

struct HybridCGResult {
    // This rank's contiguous portion of the global solution.
    std::vector<double> x_local;
    bool converged = false;
    std::size_t iterations = 0;
    double relative_residual = 0.0; // true ||b-Ax||_2 / ||b||_2
};

// Distributed-memory MPI + shared-memory OpenMP Conjugate Gradient.
// All ranks in comm must call this function in the same collective order.
// MPI calls are made by the calling thread; OpenMP regions complete before
// each MPI collective (compatible with MPI_THREAD_FUNNELED).
HybridCGResult solve_hybrid_cg(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const CGOptions& options = {},
    const OpenMPOptions& omp_options = {},
    MPI_Comm comm = MPI_COMM_WORLD
);

} // namespace cg
