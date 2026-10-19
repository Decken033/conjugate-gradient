
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

 // Compute the dot product a^T*b using an OpenMP reduction.
 //
 // Requires a.size() == b.size().
 // Floating-point reduction order may differ from serial,
 // producing small numerical differences.
 double dot_openmp(
     const std::vector<double>& a,
     const std::vector<double>& b,
     const OpenMPOptions& options = {}
 );

//
// Parallel CG vector operations.
// Each thread updates independent vector elements.
//

// Update the solution and residual:
// x[i] = x[i] + alpha * p[i]
// r[i] = r[i] - alpha * Ap[i]
void update_solution_residual_openmp(
    std::vector<double>& x,
    std::vector<double>& r,
    const std::vector<double>& p,
    const std::vector<double>& Ap,
    double alpha,
    const OpenMPOptions& options = {}
);

// Update the search direction:
// p[i] = r[i] + beta * p[i]
void update_direction_openmp(
    std::vector<double>& p,
    const std::vector<double>& r,
    double beta,
    const OpenMPOptions& options = {}
);

// Compute the true residual:
// residual[i] = b[i] - Ax[i]
std::vector<double> subtract_openmp(
    const std::vector<double>& b,
    const std::vector<double>& Ax,
    const OpenMPOptions& options = {}
);

} // namespace cg
