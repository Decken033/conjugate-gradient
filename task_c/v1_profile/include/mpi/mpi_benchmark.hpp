
#pragma once

#include "serial/cg_solver.hpp"
#include "mpi/mpi_partition.hpp"

#include <mpi.h>
#include <cstddef>
#include <string>
#include <vector>

namespace cg {

// Results from one measured MPI CG execution.
//
// solve_seconds reports the MAXIMUM elapsed solver time across
// all ranks, obtained with MPI_Reduce(MPI_MAX).
//
// This ensures timing captures the slowest participant, representing
// the true wall-clock time the parallel solver took.
struct MPIBenchmarkRecord {
    std::string matrix_name;

    std::size_t n = 0;
    std::size_t nnz = 0;
    int ranks = 1;       // MPI communicator size
    std::size_t run_index = 0;

    std::size_t iterations = 0;
    bool converged = false;

    // Maximum elapsed solver time across all ranks.
    // Timing: MPI_Barrier -> MPI_Wtime -> solve -> MPI_Wtime
    // Excludes: distribution, RHS construction, warmups,
    //           validation, CSV export, console output.
    double solve_seconds = 0.0;

    double true_relative_residual = 0.0;
    double relative_solution_error = 0.0;
};

// Run configurable warmups and repeated timed MPI CG solves.
//
// All ranks participate in every warmup and measured run.
// Only rank 0 records benchmark results for CSV output.
//
// Timing uses:
//   MPI_Barrier(comm);  // synchronize before timing
//   t_start = MPI_Wtime();
//   solve_mpi_cg(A, b_local, options, comm);
//   t_end = MPI_Wtime();
//   MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
//
// Parameters:
//   A:                    Distributed matrix
//   b_local:              This rank's RHS portion
//   exact_solution_local: This rank's portion of the known solution x* = 1
//                         (used for relative solution error, computed outside timing)
//   matrix_name:          Label for CSV output
//   repeats:              Number of timed solver executions
//   options:              CG solver options
//   warmups:              Untimed warmup executions before measurement
//   comm:                 MPI communicator
//
// Returns benchmark records on rank 0 only;
// other ranks return empty vector.
//
// Collective: all ranks must call.
std::vector<MPIBenchmarkRecord> benchmark_mpi_cg(
    const MPIPartitionedMatrix& A,
    const std::vector<double>& b_local,
    const std::vector<double>& exact_solution_local,
    const std::string& matrix_name,
    std::size_t repeats = 5,
    const CGOptions& options = {},
    std::size_t warmups = 1,
    MPI_Comm comm = MPI_COMM_WORLD
);

// Write MPI benchmark results to CSV file.
//
// Must be called by rank 0 only (or with rank parameter check).
// Uses the same 12-column schema as serial and OpenMP benchmarks:
//   solver,matrix_name,n,nnz,ranks,threads,run_index,iterations,
//   converged,solve_seconds,true_relative_residual,relative_solution_error
//
// For pure MPI: solver="mpi", threads=1
// This allows direct comparison with serial/OpenMP benchmarks.
void write_mpi_benchmark_csv(
    const std::vector<MPIBenchmarkRecord>& records,
    const std::string& output_path,
    int rank
);

} // namespace cg
