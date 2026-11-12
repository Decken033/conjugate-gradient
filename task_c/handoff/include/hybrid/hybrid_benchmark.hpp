#pragma once

#include "hybrid/hybrid_solver.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace cg {

struct HybridBenchmarkRecord {
    std::string matrix_name;
    std::size_t n = 0;
    std::size_t nnz = 0;
    int ranks = 1;
    int threads = 1;
    OpenMPSchedule schedule = OpenMPSchedule::Static;
    int chunk_size = 0;
    std::size_t run_index = 0;
    std::size_t iterations = 0;
    bool converged = false;
    double solve_seconds = 0.0; // maximum elapsed time across MPI ranks
    double true_relative_residual = 0.0;
    double relative_solution_error = 0.0;
};

// All ranks participate. Rank 0 receives records; other ranks return empty.
std::vector<HybridBenchmarkRecord> benchmark_hybrid_cg(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const std::vector<double>& exact_solution_local,
    const std::string& matrix_name,
    std::size_t repeats = 5,
    const CGOptions& options = {},
    const OpenMPOptions& omp_options = {},
    std::size_t warmups = 1,
    MPI_Comm comm = MPI_COMM_WORLD
);

void write_hybrid_benchmark_csv(
    const std::vector<HybridBenchmarkRecord>& records,
    const std::string& output_path,
    int rank
);

} // namespace cg
