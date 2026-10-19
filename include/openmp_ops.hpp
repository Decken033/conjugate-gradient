
#pragma once

#include "csr_matrix.hpp"

#include <vector>

namespace cg {

// Scheduling policies for OpenMP parallel loops.
enum class OpenMPSchedule {
    Static,
    Dynamic,
    Guided
};

// Configuration shared by OpenMP operations and the CG solver.
struct OpenMPOptions {
    // 0: use the OpenMP runtime default thread count.
    // Positive values: request the specified number of threads.
    int num_threads = 0;

    // Default: static scheduling.
    OpenMPSchedule schedule = OpenMPSchedule::Static;

    // 0: use the scheduling policy's default chunk size.
    // Positive values: specify the chunk size.
    int chunk_size = 0;
};

// Compute y = A*x using OpenMP row-parallel CSR SpMV.
//
// A and x are read-only during computation.
// Each row of y is written by exactly one thread.
//
// The matrix must have valid CSR storage.
// The input vector must match the matrix dimensions.
std::vector<double> csr_matvec_openmp(
    const CSRMatrix& A,
    const std::vector<double>& x,
    const OpenMPOptions& options = {}
);

} // namespace cg
