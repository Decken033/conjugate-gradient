#pragma once

#include "mpi/mpi_partition.hpp"

#include <mpi.h>
#include <vector>

namespace cg {

// Reusable buffers for repeated Hybrid SpMV calls during one CG solve.
// The workspace is rank-local and must be created collectively-compatible
// with the same matrix partition and communicator used by the solver.
//
// V1 changes memory lifetime only: it does not change the Allgatherv
// algorithm, CSR traversal order, OpenMP schedule, or CG mathematics.
struct HybridWorkspace {
    std::vector<int> recvcounts;
    std::vector<int> displs;
    std::vector<double> x_global;
    std::vector<double> spmv_output;
};

// Allocate and initialize communication metadata and buffers once.
HybridWorkspace make_hybrid_workspace(
    const MPIPartitionedMatrix& A,
    MPI_Comm comm = MPI_COMM_WORLD
);

} // namespace cg
