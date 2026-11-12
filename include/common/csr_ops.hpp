// Sparse matrix-vector multiplication (SpMV)

#pragma once

#include "common/csr_matrix.hpp"

#include <vector>

namespace cg {

// Compute y = A*x for a matrix stored in CSR format.
//
// Reference:
// Saad, Y. (2003). Iterative Methods for Sparse Linear Systems,
// 2nd edition, SIAM. See the discussion of sparse matrix storage
// formats and sparse matrix-vector products.
//
// This is a serial implementation. Parallel implementations may
// provide their own SpMV kernels.

std::vector<double> csr_matvec(
    const CSRMatrix& A,
    const std::vector<double>& x
);

} // namespace cg
