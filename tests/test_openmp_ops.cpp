
/*
 * OpenMP CSR SpMV correctness tests.
 *
 * Compares parallel SpMV against the serial reference.
 *
 * Coverage:
 * - Known 3x3 sparse matrix with expected output
 * - Irregular CSR matrix, including an empty row
 * - OpenMP thread counts: 1 and 2
 * - Static, dynamic and guided scheduling
 * - Default and explicit scheduling chunk sizes
 * - Input validation and invalid configurations
 */

#include "csr_ops.hpp"
#include "openmp_ops.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_vectors(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const char* test_name
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            std::string(test_name) + ": output size mismatch"
        );
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double tolerance =
            1e-12 * (1.0 + std::abs(expected[i]));

        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > tolerance) {
            std::cerr << "FAIL: " << test_name
                      << ", index=" << i
                      << ", expected=" << expected[i]
                      << ", actual=" << actual[i] << '\n';

            throw std::runtime_error(
                "OpenMP SpMV numerical mismatch"
            );
        }
    }
}

template <typename Function>
void expect_invalid_argument(
    Function function,
    const char* test_name
) {
    bool rejected = false;

    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected) {
        throw std::runtime_error(
            std::string(test_name) +
            ": expected invalid_argument"
        );
    }
}

void test_matrix(
    const cg::CSRMatrix& A,
    const std::vector<double>& x,
    const char* test_name
) {
    const auto expected = cg::csr_matvec(A, x);

    const cg::OpenMPSchedule schedules[] = {
        cg::OpenMPSchedule::Static,
        cg::OpenMPSchedule::Dynamic,
        cg::OpenMPSchedule::Guided
    };

    for (int threads : {1, 2}) {
        for (const auto schedule : schedules) {
            for (int chunk : {0, 1, 2, 3}) {
                cg::OpenMPOptions options;
                options.num_threads = threads;
                options.schedule = schedule;
                options.chunk_size = chunk;

                const auto actual =
                    cg::csr_matvec_openmp(A, x, options);

                check_vectors(actual, expected, test_name);
            }
        }
    }
}

void test_known_matrix() {
    // A = [ 4 -1  0 ]
    //     [-1  4 -1 ]
    //     [ 0 -1  4 ]
    //
    // x = [1, 2, 3]
    // Ax = [2, 4, 10]

    cg::CSRMatrix A;
    A.rows = 3;
    A.cols = 3;
    A.row_ptr = {0, 2, 5, 7};
    A.col_idx = {0, 1, 0, 1, 2, 1, 2};
    A.values = {4.0, -1.0, -1.0, 4.0,
                -1.0, -1.0, 4.0};

    const std::vector<double> x = {1.0, 2.0, 3.0};

    check_vectors(
        cg::csr_matvec_openmp(A, x, {1}),
        {2.0, 4.0, 10.0},
        "known 3x3 matrix"
    );

    test_matrix(A, x, "known matrix scheduling");
}

void test_irregular_matrix() {
    // Six rows with lengths 1, 4, 0, 2, 3 and 6.
    // Includes an empty row and uneven work distribution.

    cg::CSRMatrix A;
    A.rows = 6;
    A.cols = 6;
    A.row_ptr = {0, 1, 5, 5, 7, 10, 16};

    A.col_idx = {
        0,
        0, 1, 3, 5,
        1, 4,
        0, 2, 5,
        0, 1, 2, 3, 4, 5
    };

    A.values = {
        2.0,
        1.0, 4.0, -2.0, 3.0,
        -1.0, 2.0,
        5.0, -2.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    };

    const std::vector<double> x =
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    test_matrix(A, x, "irregular CSR matrix");
}

void test_invalid_inputs() {
    cg::CSRMatrix A;
    A.rows = 3;
    A.cols = 3;
    A.row_ptr = {0, 1, 2, 3};
    A.col_idx = {0, 1, 2};
    A.values = {1.0, 1.0, 1.0};

    const std::vector<double> x = {1.0, 2.0, 3.0};

    expect_invalid_argument(
        [&]() {
            cg::csr_matvec_openmp(A, {1.0, 2.0});
        },
        "dimension mismatch"
    );

    cg::OpenMPOptions options;

    options.num_threads = -1;
    expect_invalid_argument(
        [&]() {
            cg::csr_matvec_openmp(A, x, options);
        },
        "negative thread count"
    );

    options.num_threads = 1;
    options.chunk_size = -1;
    expect_invalid_argument(
        [&]() {
            cg::csr_matvec_openmp(A, x, options);
        },
        "negative chunk size"
    );

    options.chunk_size = 0;
    options.schedule = static_cast<cg::OpenMPSchedule>(999);

    expect_invalid_argument(
        [&]() {
            cg::csr_matvec_openmp(A, x, options);
        },
        "unsupported scheduling policy"
    );

    cg::CSRMatrix invalid = A;
    invalid.row_ptr = {0, 2, 1, 3};

    expect_invalid_argument(
        [&]() {
            cg::csr_matvec_openmp(invalid, x);
        },
        "invalid CSR structure"
    );
}

} // namespace

int main() {
    try {
        test_known_matrix();
        test_irregular_matrix();
        test_invalid_inputs();

        std::cout << "OpenMP CSR SpMV tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OpenMP CSR SpMV tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
