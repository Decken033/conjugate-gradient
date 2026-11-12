
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

void test_dot_product() {
    for (int threads : {1, 2}) {
        cg::OpenMPOptions options;
        options.num_threads = threads;

        // Positive values: 1*4 + 2*5 + 3*6 = 32.
        const double positive = cg::dot_openmp(
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            options
        );

        if (std::abs(positive - 32.0) > 1e-12) {
            throw std::runtime_error(
                "OpenMP dot product: positive values mismatch"
            );
        }

        // Mixed signs: 2 - 6 - 15 - 28 = -47.
        const double signed_result = cg::dot_openmp(
            {1.0, -2.0, 3.0, -4.0},
            {2.0, 3.0, -5.0, 7.0},
            options
        );

        if (std::abs(signed_result + 47.0) > 1e-12) {
            throw std::runtime_error(
                "OpenMP dot product: signed values mismatch"
            );
        }

        // Empty vectors should produce zero.
        const double empty = cg::dot_openmp(
            {}, {}, options
        );

        if (empty != 0.0) {
            throw std::runtime_error(
                "OpenMP dot product: empty vector mismatch"
            );
        }
    }

    // Reject mismatched vector lengths.
    expect_invalid_argument(
        []() {
            cg::dot_openmp({1.0, 2.0}, {3.0});
        },
        "dot product dimension mismatch"
    );

    // Reject invalid OpenMP options.
    cg::OpenMPOptions options;
    options.num_threads = -1;

    expect_invalid_argument(
        [&]() {
            cg::dot_openmp({1.0}, {2.0}, options);
        },
        "dot product negative thread count"
    );

    options.num_threads = 1;
    options.chunk_size = -1;

    expect_invalid_argument(
        [&]() {
            cg::dot_openmp({1.0}, {2.0}, options);
        },
        "dot product negative chunk size"
    );
}

void test_vector_operations() {
    for (int threads : {1, 2}) {
        cg::OpenMPOptions options;
        options.num_threads = threads;

        // Test the combined CG solution and residual update.
        std::vector<double> x = {1.0, 2.0, 3.0};
        std::vector<double> r = {4.0, 5.0, 6.0};
        const std::vector<double> p = {2.0, 4.0, 6.0};
        const std::vector<double> Ap = {1.0, -2.0, 3.0};

        cg::update_solution_residual_openmp(
            x, r, p, Ap, 0.5, options
        );

        check_vectors(
            x, {2.0, 4.0, 6.0},
            "OpenMP solution update"
        );

        check_vectors(
            r, {3.5, 6.0, 4.5},
            "OpenMP residual update"
        );

        // Test the CG search-direction recurrence.
        std::vector<double> direction = {2.0, 4.0, 6.0};

        cg::update_direction_openmp(
            direction, r, 0.25, options
        );

        check_vectors(
            direction, {4.0, 7.0, 6.0},
            "OpenMP direction update"
        );

        // Test true-residual vector subtraction.
        const auto residual = cg::subtract_openmp(
            {5.0, 7.0, 9.0},
            {1.0, 2.0, 3.0},
            options
        );

        check_vectors(
            residual, {4.0, 5.0, 6.0},
            "OpenMP vector subtraction"
        );

        // Empty vectors should be handled correctly.
        std::vector<double> empty_x;
        std::vector<double> empty_r;
        const std::vector<double> empty;

        cg::update_solution_residual_openmp(
            empty_x, empty_r, empty, empty, 0.5, options
        );

        cg::update_direction_openmp(
            empty_x, empty, 0.5, options
        );

        if (!cg::subtract_openmp(empty, empty, options).empty()) {
            throw std::runtime_error(
                "OpenMP empty subtraction mismatch"
            );
        }
    }

    // Invalid vector dimensions must be rejected.
    expect_invalid_argument(
        []() {
            std::vector<double> x = {1.0, 2.0};
            std::vector<double> r = {3.0, 4.0};
            cg::update_solution_residual_openmp(
                x, r, {1.0}, {2.0, 3.0}, 0.5
            );
        },
        "solution/residual dimension mismatch"
    );

    expect_invalid_argument(
        []() {
            std::vector<double> p = {1.0, 2.0};
            cg::update_direction_openmp(
                p, {3.0}, 0.5
            );
        },
        "direction dimension mismatch"
    );

    expect_invalid_argument(
        []() {
            cg::subtract_openmp(
                {1.0, 2.0}, {3.0}
            );
        },
        "subtraction dimension mismatch"
    );

    // Invalid OpenMP configurations must be rejected.
    cg::OpenMPOptions bad_options;
    bad_options.num_threads = -1;

    expect_invalid_argument(
        [&]() {
            std::vector<double> x = {1.0};
            std::vector<double> r = {2.0};
            cg::update_solution_residual_openmp(
                x, r, {3.0}, {4.0}, 0.5, bad_options
            );
        },
        "solution/residual invalid threads"
    );

    expect_invalid_argument(
        [&]() {
            std::vector<double> p = {1.0};
            cg::update_direction_openmp(
                p, {2.0}, 0.5, bad_options
            );
        },
        "direction invalid threads"
    );

    expect_invalid_argument(
        [&]() {
            cg::subtract_openmp(
                {1.0}, {2.0}, bad_options
            );
        },
        "subtraction invalid threads"
    );
}

} // namespace

int main() {
    try {
        test_known_matrix();
        test_irregular_matrix();
        test_invalid_inputs();
        test_dot_product();
        test_vector_operations();

        std::cout << "OpenMP numerical operations tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OpenMP numerical operations tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
