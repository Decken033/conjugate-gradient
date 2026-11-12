#pragma once

#include "serial/cg_solver.hpp"

namespace cg {

// Solve Ax = b using Conjugate Gradient with OpenMP GPU offloading.
//
// Requirements:
// - A must be symmetric positive definite.
// - Initial guess x0 = 0.
// - GPU execution is required; CPU fallback is not permitted.
//
// Uses the existing CGOptions and CGResult interfaces.
// The true relative residual is verified before reporting convergence.
//
// Implementation: src/gpu/cg_gpu.cpp
CGResult solve_gpu_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const CGOptions& options = {}
);

} // namespace cg
