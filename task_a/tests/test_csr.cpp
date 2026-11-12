
#include "csr_matrix.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
    cg::CSRMatrix A;

    A.rows = 2;
    A.cols = 2;
    A.row_ptr = {0, 2, 4};
    A.col_idx = {0, 1, 0, 1};
    A.values = {4.0, -1.0, -1.0, 4.0};

    A.validate();

    assert(A.nnz() == 4);

    bool rejected = false;

    A.col_idx[0] = 5;

    try {
        A.validate();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    assert(rejected);

    std::cout << "CSR interface test: PASS\n";
    return 0;
}
