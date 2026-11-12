// Defines solver options, result structure and function declaration

#pragma once

#include "common/csr_matrix.hpp"

#include <cstddef>
#include <vector>

namespace cg {

struct CGOptions {
    double relative_tolerance = 1e-8;
    std::size_t max_iterations = 10000;
};

struct CGResult {
    std::vector<double> x;

    bool converged = false;
    std::size_t iterations = 0;

    // True relative residual: ||b - Ax||_2 / ||b||_2
    double relative_residual = 0.0;
};

// Solve Ax = b using serial Conjugate Gradient, requires A to be symmetric positive definite. Uses x0 = 0.
CGResult solve_serial_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const CGOptions& options = {}
);

} // namespace cg
