
#include "matrix_market.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::string test_path = "test_matrix_market_fixture.mtx";

void write_fixture(const std::string& contents) {
    std::ofstream output(test_path);

    if (!output) {
        throw std::runtime_error("Cannot create test fixture");
    }

    output << contents;

    if (!output) {
        throw std::runtime_error("Cannot write test fixture");
    }
}

void test_symmetric_matrix() {
    write_fixture(
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "% Small symmetric test matrix\n"
        "3 3 4\n"
        "1 1 4.0\n"
        "2 1 -1.0\n"
        "2 2 5.0\n"
        "3 3 6.0\n"
    );

    const cg::CSRMatrix A = cg::load_matrix_market(test_path);

    assert(A.rows == 3);
    assert(A.cols == 3);
    assert(A.nnz() == 5);

    assert((A.row_ptr ==
            std::vector<std::size_t>{0, 2, 4, 5}));

    assert((A.col_idx ==
            std::vector<std::size_t>{0, 1, 0, 1, 2}));

    assert((A.values ==
            std::vector<double>{4.0, -1.0, -1.0, 5.0, 6.0}));
}

void test_general_matrix_with_duplicates() {
    write_fixture(
        "%%MatrixMarket matrix coordinate real general\n"
        "2 2 4\n"
        "1 1 3.0\n"
        "1 2 1.0\n"
        "1 2 2.0\n"
        "2 2 4.0\n"
    );

    const cg::CSRMatrix A = cg::load_matrix_market(test_path);

    assert(A.rows == 2);
    assert(A.cols == 2);
    assert(A.nnz() == 3);

    assert((A.row_ptr ==
            std::vector<std::size_t>{0, 2, 3}));

    assert((A.col_idx ==
            std::vector<std::size_t>{0, 1, 1}));

    assert((A.values ==
            std::vector<double>{3.0, 3.0, 4.0}));
}

void test_invalid_coordinates() {
    write_fixture(
        "%%MatrixMarket matrix coordinate real general\n"
        "2 2 1\n"
        "3 1 5.0\n"
    );

    bool rejected = false;

    try {
        (void)cg::load_matrix_market(test_path);
    } catch (const std::runtime_error&) {
        rejected = true;
    }

    assert(rejected);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        test_symmetric_matrix();
        test_general_matrix_with_duplicates();
        test_invalid_coordinates();
        // Optional integration test with the full thermal2 dataset.
        // Run manually by passing the Matrix Market file path.
        // The default CTest run uses only the small fixtures.
        if (argc == 2) {
            const cg::CSRMatrix A =
                cg::load_matrix_market(argv[1]);

            std::cout << "Loaded real-world matrix:\n";
            std::cout << "Rows: " << A.rows << '\n';
            std::cout << "Cols: " << A.cols << '\n';
            std::cout << "NNZ: " << A.nnz() << '\n';

            if (A.rows != 1228045 ||
                A.cols != 1228045 ||
                A.nnz() != 8580313) {
                throw std::runtime_error(
                    "thermal2 dimensions or NNZ mismatch");
            }

            std::cout
                << "thermal2 CSR validation passed.\n";
        }
        std::remove(test_path.c_str());

        std::cout
            << "Matrix Market loader tests passed.\n";

        return 0;
    } catch (...) {
        std::remove(test_path.c_str());
        throw;
    }
}
