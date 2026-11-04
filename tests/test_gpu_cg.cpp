#include "gpu_solver.hpp"
#include "cg_solver.hpp"
#include "csr_ops.hpp"
#include "poisson.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void validate_gpu_poisson(
    const cg::CSRMatrix& A,
    const char* label
) {
    const std::vector<double> expected(A.rows, 1.0);
    const auto b = cg::csr_matvec(A, expected);

    cg::CGOptions options;
    options.relative_tolerance = 1e-8;
    options.max_iterations = 10000;

    const auto gpu = cg::solve_gpu_cg(A, b, options);
    const auto cpu = cg::solve_serial_cg(A, b, options);

    if (!gpu.converged || !cpu.converged) {
        throw std::runtime_error(
            "Poisson CG did not converge"
        );
    }

    const auto Ax = cg::csr_matvec(A, gpu.x);

    double residual_squared = 0.0;
    double rhs_squared = 0.0;
    double error_squared = 0.0;
    double difference_squared = 0.0;

    for (std::size_t i = 0; i < A.rows; ++i) {
        const double residual = b[i] - Ax[i];
        const double error = gpu.x[i] - expected[i];
        const double difference = gpu.x[i] - cpu.x[i];

        residual_squared += residual * residual;
        rhs_squared += b[i] * b[i];
        error_squared += error * error;
        difference_squared += difference * difference;
    }

    const double true_relative_residual =
        std::sqrt(residual_squared / rhs_squared);

    const double solution_error =
        std::sqrt(error_squared / static_cast<double>(A.rows));

    const double cpu_gpu_difference =
        std::sqrt(difference_squared / static_cast<double>(A.rows));

    std::cout
        << label
        << " N=" << A.rows
        << " NNZ=" << A.nnz()
        << " GPU_iterations=" << gpu.iterations
        << " CPU_iterations=" << cpu.iterations
        << " true_residual=" << true_relative_residual
        << " solution_error=" << solution_error
        << " CPU_GPU_difference=" << cpu_gpu_difference
        << '\n';

    if (!std::isfinite(true_relative_residual) ||
        !std::isfinite(solution_error) ||
        !std::isfinite(cpu_gpu_difference) ||
        true_relative_residual > options.relative_tolerance ||
        solution_error > 1e-6 ||
        cpu_gpu_difference > 1e-5 ||
        std::abs(gpu.relative_residual - true_relative_residual)
            > 1e-10) {
        throw std::runtime_error(
            "GPU Poisson numerical validation failed"
        );
    }
}
int main() {
    try {
        cg::CSRMatrix A;
        A.rows = 2;
        A.cols = 2;
        A.row_ptr = {0, 2, 4};
        A.col_idx = {0, 1, 0, 1};
        A.values = {4.0, 1.0, 1.0, 3.0};

        const std::vector<double> b = {1.0, 2.0};

        cg::CGOptions options;
        options.relative_tolerance = 1e-8;
        options.max_iterations = 100;

        const auto result = cg::solve_gpu_cg(A, b, options);

        const double expected_x0 = 1.0 / 11.0;
        const double expected_x1 = 7.0 / 11.0;

        if (result.x.size() != 2) {
            throw std::runtime_error("Wrong solution dimension");
        }

        const double r0 =
            b[0] - (4.0 * result.x[0] + result.x[1]);

        const double r1 =
            b[1] - (result.x[0] + 3.0 * result.x[1]);

        const double relative_residual =
            std::sqrt(r0 * r0 + r1 * r1) / std::sqrt(5.0);

        std::cout << "GPU converged: " << result.converged << '\n';
        std::cout << "Iterations: " << result.iterations << '\n';
        std::cout << "x[0]: " << result.x[0] << '\n';
        std::cout << "x[1]: " << result.x[1] << '\n';
        std::cout << "Independent residual: "
                  << relative_residual << '\n';

        const bool pass =
            result.converged &&
            std::abs(result.x[0] - expected_x0) < 1e-8 &&
            std::abs(result.x[1] - expected_x1) < 1e-8 &&
            relative_residual <= options.relative_tolerance;

        if (pass) {
            validate_gpu_poisson(
                cg::make_poisson_2d(16), "Poisson2D_16"
            );
            validate_gpu_poisson(
                cg::make_poisson_2d(64), "Poisson2D_64"
            );
            validate_gpu_poisson(
                cg::make_poisson_3d(8), "Poisson3D_8"
            );
        }

        std::cout << (pass ? "PASS" : "FAIL") << '\n';
        return pass ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
