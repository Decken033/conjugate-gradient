
#pragma once

#include "serial/cg_solver.hpp"
#include "openmp/openmp_ops.hpp"

namespace cg {

// Solve Ax = b using CPU OpenMP Conjugate Gradient.
//
// Requires A to be symmetric positive definite.
// Uses the same numerical conventions as solve_serial_cg:
//   - Initial guess x0 = 0
//   - Double-precision arithmetic
//   - Relative residual convergence criterion
//   - True residual verification
//   - Numerical breakdown checks
//
// OpenMP parallelizes the computational operations;
// CG iteration dependencies remain sequential.
CGResult solve_openmp_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const CGOptions& options = {},
    const OpenMPOptions& omp_options = {}
);

} // namespace cg
