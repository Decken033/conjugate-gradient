
/*
 * CSV output for MPI CG benchmark results.
 *
 * Uses the same 12-column schema as serial and OpenMP benchmarks
 * for direct comparability:
 *
 * solver,matrix_name,n,nnz,ranks,threads,run_index,iterations,
 * converged,solve_seconds,true_relative_residual,relative_solution_error
 *
 * For pure MPI implementation:
 *   solver = "mpi"
 *   threads = 1 (no OpenMP threading in pure MPI)
 *
 * solve_seconds reports the MAXIMUM elapsed time across all MPI ranks,
 * obtained via MPI_Reduce(MPI_MAX).
 *
 * Only rank 0 should call this function.
 */

#include "mpi/mpi_benchmark.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace cg {
namespace {

// CSV escaping for text fields
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

void write_mpi_benchmark_csv(
    const std::vector<MPIBenchmarkRecord>& records,
    const std::string& output_path,
    int rank
) {
    // Only rank 0 writes CSV
    if (rank != 0) {
        return;
    }

    if (output_path.empty()) {
        throw std::invalid_argument(
            "MPI benchmark CSV: output path cannot be empty"
        );
    }

    std::ofstream output(output_path, std::ios::out | std::ios::trunc);

    if (!output.is_open()) {
        throw std::runtime_error(
            "MPI benchmark CSV: cannot open output file: " + output_path
        );
    }

    // Write header (same schema as serial/OpenMP)
    output << "solver,matrix_name,n,nnz,ranks,threads,"
              "run_index,iterations,converged,solve_seconds,"
              "true_relative_residual,relative_solution_error\n";

    // 17-digit precision for numerical analysis
    output << std::setprecision(17);

    for (const auto& record : records) {
        output
            << "mpi,"
            << csv_escape(record.matrix_name) << ','
            << record.n << ','
            << record.nnz << ','
            << record.ranks << ','
            << 1 << ','  // threads: pure MPI = 1
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
            "MPI benchmark CSV: failed to write output file: " + output_path
        );
    }
}

} // namespace cg
