
// Sparse matrix-vector multiplication (SpMV)

#include "common/csr_ops.hpp"

#include <stdexcept>

namespace cg {

std::vector<double> csr_matvec(
    const CSRMatrix& A,
    const std::vector<double>& x
) {
    // Validate CSR structure and input vector dimensions.
    A.validate();

    if (x.size() != A.cols) {
        throw std::invalid_argument(
            "CSR matrix-vector multiplication: dimension mismatch"
        );
    }

    std::vector<double> y(A.rows, 0.0);

    // Each CSR row stores a contiguous range of nonzero entries:
    // [row_ptr[i], row_ptr[i+1]).
    //
    // Reference: Saad (2003), sparse matrix storage and SpMV.
    for (std::size_t i = 0; i < A.rows; ++i) {
        double sum = 0.0;

        for (std::size_t k = A.row_ptr[i];
             k < A.row_ptr[i + 1];
             ++k) {
            sum += A.values[k] * x[A.col_idx[k]];
        }

        y[i] = sum;
    }

    return y;
}

} // namespace cg
