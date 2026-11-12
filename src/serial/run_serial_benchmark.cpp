
/*
 * Command-line runner for serial CG benchmarking.
 *
 * Usage:
 *   run_serial_benchmark <2d|3d> <grid_size> <repeats> <output.csv>
 *   run_serial_benchmark thermal2 <matrix.mtx> <repeats> <output.csv>
 *
 * Examples:
 *   run_serial_benchmark 2d 64 5 results/serial_2d_64.csv
 *   run_serial_benchmark thermal2 data/suitesparse/thermal2/thermal2.mtx 1 results/serial_thermal2.csv
 *
 * Example:
 *   run_serial_benchmark 2d 64 5 results/serial_2d_64.csv
 *
 * Matrix generation and CSV writing are excluded from
 * solver timing. The benchmark implementation performs
 * one untimed warm-up before the measured repetitions.
 */

#include "common/poisson.hpp"
#include "common/matrix_market.hpp"
#include "common/csr_ops.hpp"
#include "serial/serial_benchmark.hpp"

#include <cctype>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parse_positive_size(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("Empty numeric argument");
    }

    for (unsigned char c : text) {
        if (!std::isdigit(c)) {
            throw std::invalid_argument(
                "Expected a positive integer: " + text
            );
        }
    }

    const auto value = std::stoull(text);

    if (value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            "Grid size and repeats must be positive and valid"
        );
    }

    return static_cast<std::size_t>(value);
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 5 && argc != 6) {
        std::cerr
            << "Usage: " << argv[0]
            << " <2d|3d|thermal2> <grid_size|matrix.mtx>"
            << " <repeats> <output.csv> [warmups]\n";
        return 1;
    }

    try {
        const std::string matrix_type = argv[1];
        const std::string matrix_argument = argv[2];
        const std::size_t repeats =
            parse_positive_size(argv[3]);
        const std::string output_path = argv[4];
        // Optional warm-up count; retain one warm-up by default.
        const std::size_t warmups =
            argc == 6 ? parse_positive_size(argv[5]) : 1;
        cg::CSRMatrix A;
        std::string matrix_name;

        if (matrix_type == "2d" || matrix_type == "3d") {
            const std::size_t grid_size =
                parse_positive_size(matrix_argument);

            if (matrix_type == "2d") {
                A = cg::make_poisson_2d(grid_size);
            } else {
                A = cg::make_poisson_3d(grid_size);
            }

            matrix_name =
                "poisson_" + matrix_type + "_" +
                std::to_string(grid_size);

        } else if (matrix_type == "thermal2") {
            // Load the original SuiteSparse thermal2 matrix.
            // Matrix loading is outside the timed CG section.
            A = cg::load_matrix_market(matrix_argument);
            matrix_name = "thermal2";

        } else {
            throw std::invalid_argument(
                "Matrix type must be 2d, 3d or thermal2"
            );
        }

        // Manufactured exact solution x* = 1.
        // RHS is generated consistently as b = A*x*.
        const std::vector<double> exact_solution(
            A.rows, 1.0
        );
        const auto b = cg::csr_matvec(A, exact_solution);

        cg::CGOptions options;
        options.relative_tolerance = 1e-8;
        options.max_iterations = 10000;

        const auto records = cg::benchmark_serial_cg(
            A,
            b,
            exact_solution,
            matrix_name,
            repeats,
            options,
            warmups
        );

        // Writing CSV occurs after all timed runs.
        cg::write_serial_benchmark_csv(
            records, output_path
        );

        std::cout
            << "Matrix: " << matrix_name << '\n'
            << "N: " << A.rows << '\n'
            << "NNZ: " << A.nnz() << '\n'
            << "Measured runs: " << records.size() << '\n'
            << "Output: " << output_path << '\n';

        for (const auto& record : records) {
            std::cout
                << "Run " << record.run_index
                << ": " << record.solve_seconds << " s"
                << ", iterations=" << record.iterations
                << ", residual="
                << record.true_relative_residual
                << ", solution_error="
                << record.relative_solution_error
                << '\n';
        }

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "Benchmark failed: "
            << error.what() << '\n';
        return 1;
    }
}
