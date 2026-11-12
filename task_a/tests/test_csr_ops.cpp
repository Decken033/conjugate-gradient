
/*
 * CSR matrix-vector multiplication correctness test.
 *
 * Reference:
 * Saad, Y. (2003). Iterative Methods for Sparse Linear Systems,
 * 2nd edition, SIAM.
 *
 * Tests a known sparse matrix-vector product and rejects
 * incompatible vector dimensions.
 */

#include "csr_ops.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    // A = [ 4 -1  0 ]
    //     [-1  4 -1 ]
    //     [ 0 -1  4 ]
    //
    // x = [1, 2, 3]
    // Expected Ax = [2, 4, 10]

    cg::CSRMatrix A;
    A.rows = 3;
    A.cols = 3;
    A.row_ptr = {0, 2, 5, 7};
    A.col_idx = {0, 1, 0, 1, 2, 1, 2};
    A.values = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0};

    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> expected = {2.0, 4.0, 10.0};

    const auto y = cg::csr_matvec(A, x);

    if (y.size() != expected.size()) {
        std::cerr << "FAIL: output size mismatch\n";
        return 1;
    }

    for (std::size_t i = 0; i < y.size(); ++i) {
        if (std::abs(y[i] - expected[i]) > 1e-12) {
            std::cerr << "FAIL: incorrect SpMV result at index "
                      << i << '\n';
            return 1;
        }
    }

    bool rejected = false;

    try {
        cg::csr_matvec(A, {1.0, 2.0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected) {
        std::cerr << "FAIL: dimension mismatch not rejected\n";
        return 1;
    }

    std::cout << "CSR SpMV test: PASS\n";
    return 0;
}
