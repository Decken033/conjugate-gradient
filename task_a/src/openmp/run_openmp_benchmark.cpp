
/*
 * Command-line runner for CPU OpenMP CG benchmarking.
 *
 * Usage:
 * run_openmp_benchmark <2d|3d|thermal2>
 *   <grid_size|matrix.mtx>
 *   <threads> <static|dynamic|guided>
 *   <chunk_size> <repeats> <output.csv> [warmups]
 *
 * Examples:
 * run_openmp_benchmark 2d 64 2 static 0 5 results/openmp_2d_64.csv
 * run_openmp_benchmark 3d 16 2 guided 4 5 results/openmp_3d_16.csv
 * run_openmp_benchmark thermal2 data/suitesparse/thermal2/thermal2.mtx 2 static 0 5 results/openmp_thermal2.csv
 *
 * Matrix construction, RHS generation, warm-up,
 * validation and CSV output are excluded from
 * the measured solver execution time.
 */

#include "poisson.hpp"
#include "matrix_market.hpp"
#include "csr_ops.hpp"
#include "openmp_benchmark.hpp"

#include <cctype>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Parse an unsigned integer, permitting zero.
std::size_t parse_nonnegative_size(
    const std::string& text
) {
    if (text.empty()) {
        throw std::invalid_argument("Empty numeric argument");
    }

    for (unsigned char c : text) {
        if (!std::isdigit(c)) {
            throw std::invalid_argument(
                "Expected a nonnegative integer: " + text
            );
        }
    }

    const auto value = std::stoull(text);

    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            "Numeric argument exceeds size_t range"
        );
    }

    return static_cast<std::size_t>(value);
}

// Parse positive integers for grid size, threads and repeats.
std::size_t parse_positive_size(
    const std::string& text
) {
    const auto value = parse_nonnegative_size(text);

    if (value == 0) {
        throw std::invalid_argument(
            "Expected a positive integer: " + text
        );
    }

    return value;
}

// OpenMP API requires an int for thread counts/chunk sizes.
int parse_openmp_integer(
    const std::string& text,
    bool allow_zero
) {
    const auto value = allow_zero
        ? parse_nonnegative_size(text)
        : parse_positive_size(text);

    if (value >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "OpenMP numeric argument exceeds int range"
        );
    }

    return static_cast<int>(value);
}

// Translate a user-facing schedule into our enum.
cg::OpenMPSchedule parse_schedule(
    const std::string& text
) {
    if (text == "static") {
        return cg::OpenMPSchedule::Static;
    }

    if (text == "dynamic") {
        return cg::OpenMPSchedule::Dynamic;
    }

    if (text == "guided") {
        return cg::OpenMPSchedule::Guided;
    }

    throw std::invalid_argument(
        "Schedule must be static, dynamic or guided"
    );
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 8 && argc != 9) {
        std::cerr
            << "Usage: " << argv[0]
            << " <2d|3d|thermal2>"
            << " <grid_size|matrix.mtx>"
            << " <threads> <static|dynamic|guided>"
            << " <chunk_size> <repeats> <output.csv> [warmups]\n";

        return 1;
    }

    try {
        const std::string matrix_type = argv[1];
        const std::string matrix_argument = argv[2];

        const int threads =
            parse_openmp_integer(argv[3], false);

        const auto schedule = parse_schedule(argv[4]);

        // Chunk 0 means use the scheduling default.
        const int chunk_size =
            parse_openmp_integer(argv[5], true);

        const std::size_t repeats =
            parse_positive_size(argv[6]);

        const std::string output_path = argv[7];
        // Optional warm-up count; default to one for backward compatibility.
        const std::size_t warmups =
            argc == 9 ? parse_positive_size(argv[8]) : 1;

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

        // Same manufactured solution as Task 1.
        const std::vector<double> exact_solution(
            A.rows, 1.0
        );

        const auto b = cg::csr_matvec(A, exact_solution);

        cg::CGOptions options;
        options.relative_tolerance = 1e-8;
        options.max_iterations = 10000;

        cg::OpenMPOptions omp_options;
        omp_options.num_threads = threads;
        omp_options.schedule = schedule;
        omp_options.chunk_size = chunk_size;

        const auto records = cg::benchmark_openmp_cg(
            A,
            b,
            exact_solution,
            matrix_name,
            repeats,
            options,
            omp_options,
            warmups
        );

        // CSV I/O is outside all measured runs.
        cg::write_openmp_benchmark_csv(
            records, output_path
        );

        std::cout
            << "Matrix: " << matrix_name << '\n'
            << "N: " << A.rows << '\n'
            << "NNZ: " << A.nnz() << '\n'
            << "Threads: " << threads << '\n'
            << "Chunk size: " << chunk_size << '\n'
            << "Warm-up runs: " << warmups << '\n'
            << "Measured runs: " << records.size() << '\n'
            << "Output: " << output_path << '\n';

        for (const auto& record : records) {
            std::cout
                << "Run " << record.run_index
                << ": " << record.solve_seconds << " s"
                << ", iterations=" << record.iterations
                << ", residual=" << record.true_relative_residual
                << ", solution_error=" << record.relative_solution_error
                << '\n';
        }

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "OpenMP benchmark failed: "
            << error.what() << '\n';

        return 1;
    }
}
