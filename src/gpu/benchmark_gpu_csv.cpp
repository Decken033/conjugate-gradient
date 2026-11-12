#include "gpu/gpu_benchmark.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace cg {
namespace {

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

void write_gpu_benchmark_csv(
    const std::vector<GPUBenchmarkRecord>& records,
    const std::string& output_path
) {
    if (output_path.empty()) {
        throw std::invalid_argument(
            "GPU benchmark CSV: output path cannot be empty"
        );
    }

    std::ofstream output(
        output_path,
        std::ios::out | std::ios::trunc
    );

    if (!output.is_open()) {
        throw std::runtime_error(
            "GPU benchmark CSV: cannot open output file: " +
            output_path
        );
    }

    output << "solver,matrix_name,n,nnz,ranks,threads,"
              "run_index,iterations,converged,solve_seconds,"
              "true_relative_residual,relative_solution_error\n";

    output << std::setprecision(17);

    for (const auto& record : records) {
        output
            << "openmp_gpu,"
            << csv_escape(record.matrix_name) << ','
            << record.n << ','
            << record.nnz << ','
            << 1 << ','
            << 1 << ','
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
            "GPU benchmark CSV: failed to write output file: " +
            output_path
        );
    }
}

} // namespace cg
