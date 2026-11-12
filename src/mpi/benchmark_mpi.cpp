
#include "mpi_benchmark.hpp"
#include "mpi_solver.hpp"
#include "mpi_ops.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cg {
namespace {

void collective_benchmark_check(bool invalid, MPI_Comm comm,
                                const char* message) {
    const int local_error = invalid ? 1 : 0;
    int global_error = 0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX, comm);
    if (global_error) {
        throw std::invalid_argument(message);
    }
}

// Validate before warmups/timing: no rank may throw or enter a different
// number of solver collectives while another rank continues benchmarking.
void validate_benchmark_inputs(
    const MPIPartitionedMatrix& A, const std::vector<double>& b_local,
    const std::vector<double>& exact_solution_local, std::size_t repeats,
    const CGOptions& options, std::size_t warmups, int rank, int num_ranks,
    MPI_Comm comm) {
    bool invalid = repeats == 0 || A.global_rows != A.global_cols ||
        A.row_start > A.row_end || A.row_end > A.global_rows ||
        b_local.size() != A.local_rows() ||
        exact_solution_local.size() != A.local_rows() ||
        !std::isfinite(options.relative_tolerance) ||
        options.relative_tolerance <= 0.0 ||
        A.local_matrix.rows != A.local_rows() ||
        A.local_matrix.cols != A.global_cols ||
        A.partition_offsets.size() != static_cast<std::size_t>(num_ranks) + 1;

    if (A.partition_offsets.size() == static_cast<std::size_t>(num_ranks) + 1) {
        invalid = invalid || A.partition_offsets.front() != 0 ||
            A.partition_offsets.back() != A.global_rows ||
            A.row_start != A.partition_offsets[rank] ||
            A.row_end != A.partition_offsets[rank + 1];
        for (int r = 0; r < num_ranks; ++r) {
            const auto begin = A.partition_offsets[r];
            const auto end = A.partition_offsets[r + 1];
            // mpi_csr_matvec uses int counts and displacements.
            invalid = invalid || begin > end || end > A.global_rows ||
                begin > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                end - begin > static_cast<std::size_t>(std::numeric_limits<int>::max());
        }
    }
    try {
        A.local_matrix.validate_structure();
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    for (double value : b_local) {
        invalid = invalid || !std::isfinite(value);
    }
    for (double value : exact_solution_local) {
        invalid = invalid || !std::isfinite(value);
    }
    for (double value : A.local_matrix.values) {
        invalid = invalid || !std::isfinite(value);
    }
    collective_benchmark_check(invalid, comm,
        "MPI benchmark: invalid input dimensions, values, partition or options");

    // Explicit conversion avoids assuming that size_t is unsigned long
    // (which is not true on LLP64 platforms).
    static_assert(sizeof(std::size_t) <= sizeof(unsigned long long),
                  "MPI benchmark requires size_t to fit unsigned long long");
    const unsigned long long settings[] = {
        static_cast<unsigned long long>(repeats),
        static_cast<unsigned long long>(warmups),
        static_cast<unsigned long long>(options.max_iterations),
        static_cast<unsigned long long>(A.global_rows),
        static_cast<unsigned long long>(A.global_cols)
    };
    unsigned long long min_settings[5] = {}, max_settings[5] = {};
    MPI_Allreduce(settings, min_settings, 5, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    MPI_Allreduce(settings, max_settings, 5, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    invalid = false;
    for (int i = 0; i < 5; ++i) {
        invalid = invalid || min_settings[i] != max_settings[i];
    }
    double min_tolerance = 0.0, max_tolerance = 0.0;
    MPI_Allreduce(&options.relative_tolerance, &min_tolerance, 1,
                  MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&options.relative_tolerance, &max_tolerance, 1,
                  MPI_DOUBLE, MPI_MAX, comm);
    invalid = invalid || min_tolerance != max_tolerance;
    collective_benchmark_check(invalid, comm,
        "MPI benchmark: repeats, warmups, solver options or dimensions differ across ranks");

    // Each rank can have a locally valid but globally inconsistent partition.
    // Compare all boundaries against rank zero without relying on size_t's MPI type.
    for (std::size_t offset : A.partition_offsets) {
        unsigned long long reference = static_cast<unsigned long long>(offset);
        MPI_Bcast(&reference, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
        invalid = invalid || reference != static_cast<unsigned long long>(offset);
    }
    collective_benchmark_check(invalid, comm,
        "MPI benchmark: partition offsets differ across ranks");
}

} // namespace

std::vector<MPIBenchmarkRecord> benchmark_mpi_cg(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const std::vector<double>& exact_solution_local,
    const std::string& matrix_name,
    std::size_t repeats,
    const CGOptions& options,
    std::size_t warmups,
    MPI_Comm comm
) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    validate_benchmark_inputs(A, b_local, exact_solution_local, repeats,
                              options, warmups, rank, num_ranks, comm);

    // Compute global NNZ for CSV
    const unsigned long long local_nnz =
        static_cast<unsigned long long>(A.local_matrix.nnz());
    unsigned long long global_nnz = 0;
    MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, comm);

    // Run untimed warmups (all ranks participate)
    for (std::size_t w = 0; w < warmups; ++w) {
        if (rank == 0) {
            std::cout << "[warmup " << w + 1 << "/" << warmups
                      << "] " << matrix_name << "\n" << std::flush;
        }

        auto warmup_result = solve_mpi_cg(A, b_local, options, comm);

        if (rank == 0 && !warmup_result.converged) {
            std::cerr << "WARNING: warmup " << w + 1
                      << " did not converge\n";
        }
    }

    // Timed benchmark runs
    std::vector<MPIBenchmarkRecord> records;

    for (std::size_t run = 0; run < repeats; ++run) {
        if (rank == 0) {
            std::cout << "[run " << run + 1 << "/" << repeats
                      << "] " << matrix_name << "\n" << std::flush;
        }

        // Synchronize all ranks before timing
        MPI_Barrier(comm);

        const double t_start = MPI_Wtime();
        auto cg_result = solve_mpi_cg(A, b_local, options, comm);
        const double t_end = MPI_Wtime();

        const double local_elapsed = t_end - t_start;

        // Reduce: report MAX time across all ranks
        // This captures the slowest participant (true wall-clock time)
        double max_elapsed = 0.0;
        MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE,
                   MPI_MAX, 0, comm);

        // Compute solution error outside timed region
        // (all ranks compute local portion, Allreduce for global norm)
        double relative_solution_error = 0.0;

        // Empty partitions contribute zero but must join the same collective.
        double local_sq_err = 0.0;
        double local_sq_exact = 0.0;
        for (std::size_t i = 0; i < exact_solution_local.size(); ++i) {
            const double diff =
                cg_result.x_local[i] - exact_solution_local[i];
            local_sq_err += diff * diff;
            local_sq_exact += exact_solution_local[i] *
                              exact_solution_local[i];
        }

        const double sums[2] = {local_sq_err, local_sq_exact};
        double global_sums[2] = {0.0, 0.0};
        MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, comm);

        if (global_sums[1] > 0.0) {
            relative_solution_error = std::sqrt(global_sums[0] / global_sums[1]);
        }

        // Only rank 0 records results
        if (rank == 0) {
            MPIBenchmarkRecord rec;
            rec.matrix_name = matrix_name;
            rec.n = A.global_rows;
            rec.nnz = global_nnz;
            rec.ranks = num_ranks;
            rec.run_index = run;
            rec.iterations = cg_result.iterations;
            rec.converged = cg_result.converged;
            rec.solve_seconds = max_elapsed;
            rec.true_relative_residual = cg_result.relative_residual;
            rec.relative_solution_error = relative_solution_error;

            records.push_back(rec);
        }
    }

    return records;
}

} // namespace cg
