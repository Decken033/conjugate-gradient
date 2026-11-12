
/*
 * CSV output for serial CG benchmark results.
 *
 * Each row represents one measured solver execution.
 *
 * Matrix generation, solver warm-up, numerical validation,
 * and CSV writing are excluded from solver timing.
 *
 * The column layout is designed for future comparison
 * with OpenMP, MPI, and hybrid CG implementations.
 */

#include "serial_benchmark.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace cg {
namespace {

// Escape text fields according to standard CSV quoting rules.
// Embedded double quotes are represented by two double quotes.
std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";

    for (char c : value) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped += c;
        }
    }

    escaped += '"';
    return escaped;
}

} // namespace

void write_serial_benchmark_csv(
    const std::vector<SerialBenchmarkRecord>& records,
    const std::string& output_path
) {
    if (output_path.empty()) {
        throw std::invalid_argument(
            "Benchmark CSV: output path cannot be empty"
        );
    }

    std::ofstream output(output_path, std::ios::out | std::ios::trunc);

    if (!output.is_open()) {
        throw std::runtime_error(
            "Benchmark CSV: cannot open output file: " + output_path
        );
    }

    output << "solver,matrix_name,n,nnz,ranks,threads,"
              "run_index,iterations,converged,solve_seconds,"
              "true_relative_residual,relative_solution_error\n";

    // Preserve double precision for numerical analysis.
    output << std::setprecision(17);

    for (const auto& record : records) {
        output
            << "serial,"
            << csv_escape(record.matrix_name) << ','
            << record.n << ','
            << record.nnz << ','
            << 1 << ','                // MPI ranks: serial = 1
            << 1 << ','                // CPU threads: serial = 1
            << record.run_index << ','
            << record.iterations << ','
            << (record.converged ? 1 : 0) << ','
            << record.solve_seconds << ','
            << record.true_relative_residual << ','
            << record.relative_solution_error
            << '\n';
    }

    output.flush();

    if (!output) {
        throw std::runtime_error(
            "Benchmark CSV: failed to write output file: " + output_path
        );
    }
}

} // namespace cg
