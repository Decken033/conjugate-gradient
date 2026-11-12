
/*
 * CSV output for OpenMP CG benchmark results.
 *
 * The first 12 columns match the Task 1 serial benchmark.
 * OpenMP-specific scheduling metadata is appended.
 *
 * Each row represents one measured solver execution.
 * CSV writing is performed outside all timed regions.
 */

#include "openmp/openmp_benchmark.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg {
namespace {

// Escape CSV fields containing commas, quotes or newlines.
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

// Convert OpenMP scheduling policy to readable CSV text.
std::string schedule_name(OpenMPSchedule schedule) {
    switch (schedule) {
        case OpenMPSchedule::Static:
            return "static";

        case OpenMPSchedule::Dynamic:
            return "dynamic";

        case OpenMPSchedule::Guided:
            return "guided";

        default:
            throw std::invalid_argument(
                "OpenMP benchmark CSV: unsupported schedule"
            );
    }
}

} // namespace

void write_openmp_benchmark_csv(
    const std::vector<OpenMPBenchmarkRecord>& records,
    const std::string& output_path
) {
    if (output_path.empty()) {
        throw std::invalid_argument(
            "OpenMP benchmark CSV: output path cannot be empty"
        );
    }

    std::ofstream output(
        output_path,
        std::ios::out | std::ios::trunc
    );

    if (!output.is_open()) {
        throw std::runtime_error(
            "OpenMP benchmark CSV: cannot open file: " +
            output_path
        );
    }

    // Common columns match Task 1 serial benchmark.
    // Scheduling metadata is appended for OpenMP analysis.
    output
        << "solver,matrix_name,n,nnz,ranks,threads,"
        << "run_index,iterations,converged,solve_seconds,"
        << "true_relative_residual,relative_solution_error,"
        << "schedule,chunk_size\n";

    output << std::setprecision(17);

    for (const auto& record : records) {
        output
            << "openmp,"
            << csv_escape(record.matrix_name) << ','
            << record.n << ','
            << record.nnz << ','
            << 1 << ','  // One process, no MPI.
            << record.threads << ','
            << record.run_index << ','
            << record.iterations << ','
            << (record.converged ? 1 : 0) << ','
            << record.solve_seconds << ','
            << record.true_relative_residual << ','
            << record.relative_solution_error << ','
            << schedule_name(record.schedule) << ','
            << record.chunk_size
            << '\n';
    }

    output.flush();

    if (!output) {
        throw std::runtime_error(
            "OpenMP benchmark CSV: failed to write file: " +
            output_path
        );
    }
}

} // namespace cg
