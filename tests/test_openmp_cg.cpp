
/*
 * Correctness tests for the complete CPU OpenMP CG solver.
 *
 * Compares serial and OpenMP CG using identical SPD matrices,
 * right-hand sides, tolerances and iteration limits.
 *
 * Tests run with 1 and 2 OpenMP threads.
 *
 * Explicit runtime checks are used instead of assert(),
 * so checks remain active in Release builds.
 */

#include "serial/cg_solver.hpp"
#include "common/csr_ops.hpp"
#include "openmp/openmp_solver.hpp"
#include "common/poisson.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void expect_exception(Function fn, const std::string& label) {
    bool caught = false;

    try {
        fn();
    } catch (const Exception&) {
        caught = true;
    }

    require(caught, label + ": expected exception not thrown");
}

// Compute ||b - Ax||_2 / ||b||_2 independently using
// the serial CSR matrix-vector multiplication.
double independent_true_residual(
    const cg::CSRMatrix& A,
    const std::vector<double>& b,
    const std::vector<double>& x
) {
    const auto Ax = cg::csr_matvec(A, x);

    double residual_squared = 0.0;
    double rhs_squared = 0.0;

    for (std::size_t i = 0; i < b.size(); ++i) {
        const double difference = b[i] - Ax[i];
        residual_squared += difference * difference;
        rhs_squared += b[i] * b[i];
    }

    if (rhs_squared == 0.0) {
        return std::sqrt(residual_squared);
    }

    return std::sqrt(residual_squared / rhs_squared);
}

// Compute ||actual - expected||_2 / ||expected||_2.
double relative_solution_error(
    const std::vector<double>& actual,
    const std::vector<double>& expected
) {
    require(
        actual.size() == expected.size(),
        "Solution vector size mismatch"
    );

    double error_squared = 0.0;
    double expected_squared = 0.0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = actual[i] - expected[i];
        error_squared += error * error;
        expected_squared += expected[i] * expected[i];
    }

    const double denominator =
        std::sqrt(expected_squared);

    return denominator > 0.0
        ? std::sqrt(error_squared) / denominator
        : std::sqrt(error_squared);
}

cg::CSRMatrix make_spd_matrix() {
    cg::CSRMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.row_ptr = {0, 2, 4};
    A.col_idx = {0, 1, 0, 1};
    A.values = {4.0, 1.0, 1.0, 3.0};
    return A;
}

// Compare both solvers using a manufactured solution.
// This checks accuracy independently of solver-reported
// convergence and residual values.
void compare_solvers(
    const cg::CSRMatrix& A,
    const std::string& label
) {
    const std::vector<double> expected(A.rows, 1.0);
    const auto b = cg::csr_matvec(A, expected);

    cg::CGOptions options;
    options.relative_tolerance = 1e-8;
    options.max_iterations = 10000;

    const auto serial = cg::solve_serial_cg(A, b, options);

    require(serial.converged, label + ": serial did not converge");

    for (int threads : {1, 2}) {
        cg::OpenMPOptions omp_options;
        omp_options.num_threads = threads;

        const auto parallel = cg::solve_openmp_cg(
            A, b, options, omp_options
        );

        const double true_residual =
            independent_true_residual(A, b, parallel.x);

        const double solution_error =
            relative_solution_error(parallel.x, expected);

        const double serial_difference =
            relative_solution_error(parallel.x, serial.x);

        std::cout
            << label
            << " threads=" << threads
            << " N=" << A.rows
            << " NNZ=" << A.nnz()
            << " serial_iterations=" << serial.iterations
            << " openmp_iterations=" << parallel.iterations
            << " true_residual=" << true_residual
            << " solution_error=" << solution_error
            << " serial_difference=" << serial_difference
            << '\n';

        require(
            parallel.converged,
            label + ": OpenMP did not converge"
        );

        require(
            parallel.iterations > 0 &&
            parallel.iterations <= options.max_iterations,
            label + ": invalid OpenMP iteration count"
        );

        require(
            std::isfinite(true_residual) &&
            true_residual <= options.relative_tolerance,
            label + ": OpenMP true residual too large"
        );

        require(
            std::isfinite(solution_error) &&
            solution_error <= 1e-6,
            label + ": OpenMP solution inaccurate"
        );

        require(
            std::isfinite(serial_difference) &&
            serial_difference <= 1e-6,
            label + ": serial/OpenMP solutions differ"
        );

        require(
            std::abs(parallel.relative_residual -
                     true_residual) <= 1e-12,
            label + ": reported residual mismatch"
        );
    }
}

void test_known_solution() {
    const auto A = make_spd_matrix();
    const std::vector<double> expected = {1.0, 2.0};
    const auto b = cg::csr_matvec(A, expected);

    for (int threads : {1, 2}) {
        cg::OpenMPOptions omp_options;
        omp_options.num_threads = threads;

        const auto result = cg::solve_openmp_cg(
            A, b, {}, omp_options
        );

        require(result.converged, "Known SPD: no convergence");
        require(result.iterations > 0 &&
                result.iterations <= 2,
                "Known SPD: unexpected iteration count");

        require(
            relative_solution_error(result.x, expected) <= 1e-10,
            "Known SPD: wrong solution"
        );
    }
}

void test_edge_cases() {
    const auto A = make_spd_matrix();

    for (int threads : {1, 2}) {
        cg::OpenMPOptions omp_options;
        omp_options.num_threads = threads;

        // Zero RHS must converge immediately.
        const auto zero = cg::solve_openmp_cg(
            A, {0.0, 0.0}, {}, omp_options
        );

        require(
            zero.converged &&
            zero.iterations == 0 &&
            zero.relative_residual == 0.0,
            "Zero RHS handling failed"
        );

        require(
            relative_solution_error(
                zero.x, {0.0, 0.0}
            ) == 0.0,
            "Zero RHS solution is nonzero"
        );

        // With no iterations allowed, a nonzero RHS
        // must remain unsolved.
        cg::CGOptions options;
        options.max_iterations = 0;

        const auto limited = cg::solve_openmp_cg(
            A, {1.0, 2.0}, options, omp_options
        );

        require(
            !limited.converged &&
            limited.iterations == 0 &&
            std::abs(limited.relative_residual - 1.0) <= 1e-12,
            "Zero iteration limit handling failed"
        );
    }

    expect_exception<std::invalid_argument>(
        [&]() {
            cg::solve_openmp_cg(A, {1.0});
        },
        "RHS dimension mismatch"
    );

    cg::CGOptions bad_tolerance;
    bad_tolerance.relative_tolerance = 0.0;

    expect_exception<std::invalid_argument>(
        [&]() {
            cg::solve_openmp_cg(
                A, {1.0, 2.0}, bad_tolerance
            );
        },
        "Invalid tolerance"
    );

    cg::OpenMPOptions bad_threads;
    bad_threads.num_threads = -1;

    expect_exception<std::invalid_argument>(
        [&]() {
            cg::solve_openmp_cg(
                A, {1.0, 2.0}, {}, bad_threads
            );
        },
        "Invalid OpenMP thread count"
    );

    cg::OpenMPOptions bad_schedule;
    bad_schedule.schedule =
        static_cast<cg::OpenMPSchedule>(999);

    expect_exception<std::invalid_argument>(
        [&]() {
            cg::solve_openmp_cg(
                A, {1.0, 2.0}, {}, bad_schedule
            );
        },
        "Invalid OpenMP schedule"
    );

    // The zero matrix is not SPD and must cause breakdown.
    cg::CSRMatrix zero_matrix;
    zero_matrix.rows = 2;
    zero_matrix.cols = 2;
    zero_matrix.row_ptr = {0, 0, 0};

    expect_exception<std::runtime_error>(
        [&]() {
            cg::solve_openmp_cg(
                zero_matrix, {1.0, 1.0}
            );
        },
        "CG breakdown"
    );
}

} // namespace

int main() {
    try {
        test_known_solution();

        compare_solvers(
            cg::make_poisson_2d(64),
            "Poisson2D"
        );

        compare_solvers(
            cg::make_poisson_3d(16),
            "Poisson3D"
        );

        test_edge_cases();

        std::cout << "OpenMP CG correctness tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "OpenMP CG correctness tests: FAIL: "
            << error.what() << '\n';
        return 1;
    }
}
