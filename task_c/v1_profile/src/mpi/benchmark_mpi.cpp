
#include "mpi/mpi_benchmark.hpp"
#include "mpi/mpi_solver.hpp"
#include "mpi/mpi_ops.hpp"
#include "common/ops_profiler.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cg {

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

    // Validate inputs
    if (repeats == 0) {
        throw std::invalid_argument(
            "MPI benchmark: repeats must be > 0"
        );
    }

    if (b_local.size() != A.local_rows()) {
        throw std::invalid_argument(
            "MPI benchmark: b_local size doesn't match partition"
        );
    }

    if (exact_solution_local.size() != A.local_rows()) {
        throw std::invalid_argument(
            "MPI benchmark: exact_solution_local size doesn't match"
        );
    }

    // Compute global NNZ for CSV
    std::size_t local_nnz = A.local_matrix.nnz();
    std::size_t global_nnz = 0;
    MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_UNSIGNED_LONG,
                  MPI_SUM, comm);

    process_ops_profile.reset();

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
        MPICGResult cg_result;
        {
            ProfileSession profile_session(process_ops_profile);
            ProfileScope solver_timer(ProfileSection::SolverTotal);
            cg_result = solve_mpi_cg(A, b_local, options, comm);
        }
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

        if (!exact_solution_local.empty()) {
            // Local squared error: ||x_local - x*_local||^2
            double local_sq_err = 0.0;
            double local_sq_exact = 0.0;

            for (std::size_t i = 0; i < exact_solution_local.size(); ++i) {
                const double diff =
                    cg_result.x_local[i] - exact_solution_local[i];
                local_sq_err += diff * diff;
                local_sq_exact += exact_solution_local[i] *
                                  exact_solution_local[i];
            }

            double global_sq_err = 0.0, global_sq_exact = 0.0;
            double sums[2] = {local_sq_err, local_sq_exact};
            double global_sums[2] = {0.0, 0.0};

            MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE,
                          MPI_SUM, comm);

            global_sq_err = global_sums[0];
            global_sq_exact = global_sums[1];

            if (global_sq_exact > 0.0) {
                relative_solution_error =
                    std::sqrt(global_sq_err / global_sq_exact);
            }
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
