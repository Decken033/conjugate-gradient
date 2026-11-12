
#include "serial/cg_solver.hpp"
#include "common/csr_ops.hpp"
#include "common/poisson.hpp"


#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace {

using cg::CSRMatrix;

// Symmetric positive definite matrix:
// [ 4  1 ]
// [ 1  3 ]
CSRMatrix make_spd_matrix() {
    CSRMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.row_ptr = {0, 2, 4};
    A.col_idx = {0, 1, 0, 1};
    A.values = {4.0, 1.0, 1.0, 3.0};
    return A;
}

bool near(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

template <typename Exception, typename Function>
void expect_exception(Function fn) {
    bool thrown = false;

    try {
        fn();
    } catch (const Exception&) {
        thrown = true;
    }

    assert(thrown);
}

void test_known_solution() {
    const auto A = make_spd_matrix();

    // Known exact solution x = [1, 2].
    const std::vector<double> expected = {1.0, 2.0};
    const auto b = cg::csr_matvec(A, expected);

    const auto result = cg::solve_serial_cg(A, b);

    assert(result.converged);
    assert(result.iterations > 0);
    assert(result.iterations <= 2);

    assert(near(result.x[0], 1.0));
    assert(near(result.x[1], 2.0));

    assert(result.relative_residual <= 1e-8);
}

void test_zero_rhs() {
    const auto A = make_spd_matrix();
    const std::vector<double> b = {0.0, 0.0};

    const auto result = cg::solve_serial_cg(A, b);

    assert(result.converged);
    assert(result.iterations == 0);
    assert(result.relative_residual == 0.0);

    for (double value : result.x) {
        assert(value == 0.0);
    }
}

void test_dimension_mismatch() {
    const auto A = make_spd_matrix();
    const std::vector<double> b = {1.0};

    expect_exception<std::invalid_argument>([&]() {
        cg::solve_serial_cg(A, b);
    });
}

void test_invalid_tolerance() {
    const auto A = make_spd_matrix();
    const std::vector<double> b = {1.0, 2.0};

    cg::CGOptions options;
    options.relative_tolerance = 0.0;

    expect_exception<std::invalid_argument>([&]() {
        cg::solve_serial_cg(A, b, options);
    });
}

void test_zero_iteration_limit() {
    const auto A = make_spd_matrix();
    const std::vector<double> b = {1.0, 2.0};

    cg::CGOptions options;
    options.max_iterations = 0;

    const auto result = cg::solve_serial_cg(A, b, options);

    assert(!result.converged);
    assert(result.iterations == 0);
    assert(near(result.relative_residual, 1.0));
}

void test_iteration_limit() {
    const auto A = make_spd_matrix();
    const std::vector<double> b = {1.0, 2.0};

    cg::CGOptions options;
    options.max_iterations = 1;
    options.relative_tolerance = 1e-12;

    const auto result = cg::solve_serial_cg(A, b, options);

    assert(!result.converged);
    assert(result.iterations == 1);
    assert(result.relative_residual > options.relative_tolerance);
}

void test_breakdown() {
    // Zero matrix is not SPD: p^T A p = 0.
    CSRMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.row_ptr = {0, 0, 0};

    const std::vector<double> b = {1.0, 1.0};

    expect_exception<std::runtime_error>([&]() {
        cg::solve_serial_cg(A, b);
    });
}

void test_true_residual() {
    const auto A = make_spd_matrix();
    const std::vector<double> b = {5.0, 7.0};

    const auto result = cg::solve_serial_cg(A, b);

    const auto Ax = cg::csr_matvec(A, result.x);

    double residual_squared = 0.0;
    double b_squared = 0.0;

    for (std::size_t i = 0; i < b.size(); ++i) {
        const double difference = b[i] - Ax[i];
        residual_squared += difference * difference;
        b_squared += b[i] * b[i];
    }

    const double independently_computed =
        std::sqrt(residual_squared / b_squared);

    assert(near(result.relative_residual,
                independently_computed, 1e-12));
}


/*
 * Poisson numerical validation for serial CG.
 *
 * The 2D five-point and 3D seven-point finite-difference
 * Laplacians with homogeneous Dirichlet boundaries are SPD.
 *
 * Reference:
 * Barrett et al. (1994), Templates for the Solution of
 * Linear Systems, SIAM: CG and sparse linear systems.
 *
 * Validation uses the manufactured solution x* = 1,
 * with b = A*x*. This allows independent verification
 * of both the residual and solution error.
 */

// Independently calculate numerical accuracy.
// This calculation does not rely on the solver's reported
// residual or its internal recursive residual vector.
void validate_poisson_solution(
    const CSRMatrix& A,
    const char* label
) {
    const std::vector<double> expected(A.rows, 1.0);
    const auto b = cg::csr_matvec(A, expected);

    cg::CGOptions options;
    options.relative_tolerance = 1e-8;
    options.max_iterations = 10000;

    const auto result = cg::solve_serial_cg(A, b, options);
    const auto Ax = cg::csr_matvec(A, result.x);

    double residual_squared = 0.0;
    double rhs_squared = 0.0;
    double error_squared = 0.0;

    for (std::size_t i = 0; i < A.rows; ++i) {
        const double r = b[i] - Ax[i];
        const double error = result.x[i] - expected[i];

        residual_squared += r * r;
        rhs_squared += b[i] * b[i];
        error_squared += error * error;
    }

    const double true_relative_residual =
        std::sqrt(residual_squared / rhs_squared);

    const double relative_solution_error =
        std::sqrt(error_squared / static_cast<double>(A.rows));

    std::cout
        << label
        << " N=" << A.rows
        << " NNZ=" << A.nnz()
        << " iterations=" << result.iterations
        << " converged=" << result.converged
        << " true_residual=" << true_relative_residual
        << " solution_error=" << relative_solution_error
        << '\n';

    assert(result.converged);
    assert(result.iterations > 0);
    assert(result.iterations <= options.max_iterations);

    assert(std::isfinite(true_relative_residual));
    assert(std::isfinite(relative_solution_error));

    assert(true_relative_residual <= options.relative_tolerance);
    assert(relative_solution_error <= 1e-6);

    // Solver-reported residual must match independent calculation.
    assert(near(result.relative_residual,
                true_relative_residual, 1e-12));
}

void test_poisson_2d() {
    const auto A = cg::make_poisson_2d(64);
    assert(A.rows == 4096);
    assert(A.nnz() == 20224);
    validate_poisson_solution(A, "Poisson2D");
}

void test_poisson_3d() {
    const auto A = cg::make_poisson_3d(16);
    assert(A.rows == 4096);
    assert(A.nnz() == 27136);
    validate_poisson_solution(A, "Poisson3D");
}

} // namespace

int main() {
    test_known_solution();
    test_zero_rhs();
    test_dimension_mismatch();
    test_invalid_tolerance();
    test_zero_iteration_limit();
    test_iteration_limit();
    test_breakdown();
    test_true_residual();
    test_poisson_2d();
    test_poisson_3d();

    return 0;
}
