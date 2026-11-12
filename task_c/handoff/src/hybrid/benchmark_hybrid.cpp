#include "hybrid/hybrid_benchmark.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cg {

std::vector<HybridBenchmarkRecord> benchmark_hybrid_cg(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const std::vector<double>& exact_solution_local,
    const std::string& matrix_name,
    std::size_t repeats,
    const CGOptions& options,
    const OpenMPOptions& omp_options,
    std::size_t warmups,
    MPI_Comm comm
) {
    int rank = 0, ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    int local_error = (repeats == 0 || b_local.size() != A.local_rows() ||
                       exact_solution_local.size() != A.local_rows()) ? 1 : 0;
    int global_error = 0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX, comm);
    if (global_error)
        throw std::invalid_argument("Hybrid benchmark: invalid repeat count or local vector dimensions");

    unsigned long long local_nnz = static_cast<unsigned long long>(A.local_matrix.nnz());
    unsigned long long global_nnz = 0;
    MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);

    for (std::size_t w = 0; w < warmups; ++w) {
        const auto warm = solve_hybrid_cg(A, b_local, options, omp_options, comm);
        if (!warm.converged)
            throw std::runtime_error("Hybrid benchmark: warm-up solve did not converge");
        if (rank == 0)
            std::cout << "[Hybrid warm-up " << (w + 1) << "/" << warmups << "] iterations="
                      << warm.iterations << " residual=" << warm.relative_residual << '\n';
    }

    std::vector<HybridBenchmarkRecord> records;
    if (rank == 0) records.reserve(repeats);

    for (std::size_t run = 0; run < repeats; ++run) {
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        const auto result = solve_hybrid_cg(A, b_local, options, omp_options, comm);
        const double stop = MPI_Wtime();

        const double local_elapsed = stop - start;
        double max_elapsed = 0.0;
        MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (!result.converged)
            throw std::runtime_error("Hybrid benchmark: measured solve did not converge");

        // Compute global relative solution error outside the timed region.
        double local_error_sq = 0.0;
        double local_exact_sq = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:local_error_sq,local_exact_sq) num_threads(omp_options.num_threads > 0 ? omp_options.num_threads : 1)
        for (std::size_t i = 0; i < exact_solution_local.size(); ++i) {
            const double diff = result.x_local[i] - exact_solution_local[i];
            local_error_sq += diff * diff;
            local_exact_sq += exact_solution_local[i] * exact_solution_local[i];
        }
        double local_sums[2] = {local_error_sq, local_exact_sq};
        double global_sums[2] = {0.0, 0.0};
        MPI_Allreduce(local_sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, comm);
        const double solution_error = global_sums[1] > 0.0
            ? std::sqrt(global_sums[0] / global_sums[1])
            : std::sqrt(global_sums[0]);

        if (rank == 0) {
            HybridBenchmarkRecord rec;
            rec.matrix_name = matrix_name;
            rec.n = A.global_rows;
            rec.nnz = static_cast<std::size_t>(global_nnz);
            rec.ranks = ranks;
            rec.threads = omp_options.num_threads > 0 ? omp_options.num_threads : 1;
            rec.schedule = omp_options.schedule;
            rec.chunk_size = omp_options.chunk_size;
            rec.run_index = run + 1;
            rec.iterations = result.iterations;
            rec.converged = result.converged;
            rec.solve_seconds = max_elapsed;
            rec.true_relative_residual = result.relative_residual;
            rec.relative_solution_error = solution_error;
            records.push_back(rec);
            std::cout << "[Hybrid run " << (run + 1) << "/" << repeats << "] time="
                      << max_elapsed << " s iterations=" << result.iterations
                      << " residual=" << result.relative_residual
                      << " solution_error=" << solution_error << '\n';
        }
    }
    return records;
}

} // namespace cg
