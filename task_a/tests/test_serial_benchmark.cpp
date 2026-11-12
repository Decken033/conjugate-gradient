
#include "serial_benchmark.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstdio>    // std::remove
#include <fstream>   // std::ifstream
#include <string>

namespace {

cg::CSRMatrix make_test_matrix() {
    // SPD matrix:
    // [4 1]
    // [1 3]
    cg::CSRMatrix A;
    A.rows = 2;
    A.cols = 2;
    A.row_ptr = {0, 2, 4};
    A.col_idx = {0, 1, 0, 1};
    A.values = {4.0, 1.0, 1.0, 3.0};
    return A;
}

void test_five_measured_runs() {
    const auto A = make_test_matrix();

    // Exact solution [1, 2]; b = A*x = [6, 7].
    const std::vector<double> x_exact = {1.0, 2.0};
    const std::vector<double> b = {6.0, 7.0};

    const auto records = cg::benchmark_serial_cg(
        A, b, x_exact, "small_spd", 5
    );

    assert(records.size() == 5);

    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];

        assert(record.matrix_name == "small_spd");
        assert(record.n == 2);
        assert(record.nnz == 4);
        assert(record.run_index == i + 1);

        assert(record.converged);
        assert(record.iterations > 0);
        assert(record.iterations <= 2);

        assert(std::isfinite(record.solve_seconds));
        assert(record.solve_seconds >= 0.0);

        assert(record.true_relative_residual <= 1e-8);
        assert(record.relative_solution_error <= 1e-8);
    }
}

void test_invalid_repeat_count() {
    const auto A = make_test_matrix();

    const std::vector<double> x_exact = {1.0, 2.0};
    const std::vector<double> b = {6.0, 7.0};

    bool threw = false;

    try {
        cg::benchmark_serial_cg(
            A, b, x_exact, "small_spd", 0
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_solution_size() {
    const auto A = make_test_matrix();

    const std::vector<double> b = {6.0, 7.0};
    const std::vector<double> invalid_solution = {1.0};

    bool threw = false;

    try {
        cg::benchmark_serial_cg(
            A, b, invalid_solution, "small_spd", 5
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_csv_output() {
    const auto A = make_test_matrix();

    const std::vector<double> x_exact = {1.0, 2.0};
    const std::vector<double> b = {6.0, 7.0};

    // Run the benchmark and obtain five measured records.
    const auto records = cg::benchmark_serial_cg(
        A, b, x_exact, "small_spd", 5
    );

    const std::string csv_path = "test_serial_benchmark.csv";

    // Write all benchmark records to CSV.
    cg::write_serial_benchmark_csv(records, csv_path);

    // Read back the generated CSV and verify its contents.
    std::ifstream input(csv_path);
    assert(input.is_open());

    std::string line;

    // Verify the CSV header.
    assert(std::getline(input, line));

    const std::string expected_header =
        "solver,matrix_name,n,nnz,ranks,threads,"
        "run_index,iterations,converged,solve_seconds,"
        "true_relative_residual,relative_solution_error";

    assert(line == expected_header);

    // Verify that exactly five data rows were written.
    std::size_t row_count = 0;

    while (std::getline(input, line)) {
        assert(!line.empty());

        // Verify the fixed metadata fields.
        const std::string expected_prefix =
            "serial,small_spd,2,4,1,1," +
            std::to_string(row_count + 1) + ",";

        assert(line.rfind(expected_prefix, 0) == 0);

        ++row_count;
    }

    assert(row_count == 5);

    input.close();

    // Remove the temporary test output.
    assert(std::remove(csv_path.c_str()) == 0);
}
} // namespace

int main() {
    test_five_measured_runs();
    test_invalid_repeat_count();
    test_invalid_solution_size();
    test_csv_output();

    return 0;
}
