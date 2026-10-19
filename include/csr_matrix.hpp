
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cg {

struct CSRMatrix {
    using index_type = std::size_t;

    index_type rows = 0;
    index_type cols = 0;

    // row_ptr.size() == rows + 1
    // col_idx.size() == values.size() == nnz
    std::vector<index_type> row_ptr;
    std::vector<index_type> col_idx;
    std::vector<double> values;

    index_type nnz() const noexcept {
        return values.size();
    }

    void validate() const {
        if (rows != cols) {
            throw std::invalid_argument(
                "CG requires a square matrix");
        }

        if (row_ptr.size() != rows + 1) {
            throw std::invalid_argument(
                "Invalid CSR row_ptr size");
        }

        if (col_idx.size() != values.size()) {
            throw std::invalid_argument(
                "CSR column/value size mismatch");
        }

        if (row_ptr.empty() || row_ptr.front() != 0 ||
            row_ptr.back() != values.size()) {
            throw std::invalid_argument(
                "Invalid CSR row boundaries");
        }

        for (index_type i = 0; i < rows; ++i) {
            if (row_ptr[i] > row_ptr[i + 1]) {
                throw std::invalid_argument(
                    "CSR row_ptr must be nondecreasing");
            }
        }

        for (index_type col : col_idx) {
            if (col >= cols) {
                throw std::invalid_argument(
                    "CSR column index out of bounds");
            }
        }
    }
};

}  // namespace cg
