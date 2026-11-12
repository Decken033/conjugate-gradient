#include "gpu/gpu_benchmark.hpp"
#include "common/poisson.hpp"
#include "common/matrix_market.hpp"
#include "common/csr_ops.hpp"

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
            A = cg::load_matrix_market(matrix_argument);
            matrix_name = "thermal2";

        } else {
            throw std::invalid_argument(
                "Matrix type must be 2d, 3d or thermal2"
            );
        }

        // Manufactured exact solution x* = 1.
        const std::vector<double> exact_solution(
            A.rows, 1.0
        );
        const auto b = cg::csr_matvec(A, exact_solution);

        cg::CGOptions options;
        options.relative_tolerance = 1e-8;
        options.max_iterations = 10000;

        const auto records = cg::benchmark_gpu_cg(
            A,
            b,
            exact_solution,
            matrix_name,
            repeats,
            options,
            warmups
        );

        // CSV writing is outside all timed solves.
        cg::write_gpu_benchmark_csv(
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
            << "GPU benchmark failed: "
            << error.what() << '\n';
        return 1;
    }
}
