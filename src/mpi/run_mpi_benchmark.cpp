
/*
 * Command-line MPI CG benchmark runner.
 *
 * Usage:
 *   mpirun -n <P> ./run_mpi_benchmark \
 *     <2d|3d|thermal2> <size|matrix.mtx> \
 *     <repeats> <output.csv> [warmups]
 *
 * Matrix type arguments:
 *   2d <n>         2D Poisson on n×n grid (N = n^2 unknowns)
 *   3d <n>         3D Poisson on n×n×n grid (N = n^3 unknowns)
 *   thermal2 <f>   Matrix Market file from SuiteSparse
 *
 * Examples:
 *   mpirun -n 4  ./run_mpi_benchmark 2d 512 10 mpi_p4_2d512.csv 4
 *   mpirun -n 16 ./run_mpi_benchmark 2d 1024 10 mpi_p16_2d1024.csv 4
 *   mpirun -n 8  ./run_mpi_benchmark thermal2 data/thermal2.mtx 5 mpi_thermal2.csv 2
 *
 * Notes:
 *   - Rank count is determined from MPI communicator (not a CLI argument)
 *   - Only rank 0 generates/loads the matrix, then distributes
 *   - Only rank 0 writes the CSV output
 *   - Timing uses MPI_Wtime and MPI_Reduce(MPI_MAX)
 */

#include "mpi_benchmark.hpp"
#include "mpi_partition.hpp"
#include "mpi_solver.hpp"
#include "csr_matrix.hpp"
#include "cg_solver.hpp"
#include "poisson.hpp"
#include "matrix_market.hpp"

#include <mpi.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage(const char* program_name) {
    std::cerr
        << "Usage: mpirun -n <P> " << program_name
        << " <2d|3d|thermal2> <n|matrix.mtx>"
        << " <repeats> <output.csv> [warmups]\n"
        << "\n"
        << "  2d <n>          2D Poisson, grid n x n (N = n^2)\n"
        << "  3d <n>          3D Poisson, grid n x n x n (N = n^3)\n"
        << "  thermal2 <f>    Matrix Market file\n"
        << "\n"
        << "  repeats         Number of timed solver runs\n"
        << "  output.csv      Output file (rank 0 only)\n"
        << "  warmups         Untimed warmup runs (default: 1)\n";
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int exit_code = 0;

    try {
        if (argc < 5 || argc > 6) {
            if (rank == 0) {
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }

        const std::string matrix_type = argv[1];
        const std::string matrix_arg = argv[2];
        const std::size_t repeats = std::stoul(argv[3]);
        const std::string output_csv = argv[4];
        const std::size_t warmups = (argc == 6) ? std::stoul(argv[5]) : 1;

        if (repeats == 0) {
            if (rank == 0) {
                std::cerr << "Error: repeats must be >= 1\n";
            }
            MPI_Finalize();
            return 1;
        }

        // Rank 0: generate or load matrix
        cg::CSRMatrix global_matrix;
        std::string matrix_name;

        if (rank == 0) {
          try {
            if (matrix_type == "2d") {
                const std::size_t n = std::stoul(matrix_arg);
                std::cout << "Generating 2D Poisson " << n << "x" << n
                          << " (N=" << n * n << ")\n";
                global_matrix = cg::make_poisson_2d(n);
                matrix_name = "poisson_2d_" + matrix_arg;
            } else if (matrix_type == "3d") {
                const std::size_t n = std::stoul(matrix_arg);
                std::cout << "Generating 3D Poisson " << n << "x" << n
                          << "x" << n
                          << " (N=" << n * n * n << ")\n";
                global_matrix = cg::make_poisson_3d(n);
                matrix_name = "poisson_3d_" + matrix_arg;
            } else if (matrix_type == "thermal2") {
                std::cout << "Loading Matrix Market: " << matrix_arg << "\n";
                global_matrix = cg::load_matrix_market(matrix_arg);
                matrix_name = "thermal2";
            } else {
                std::cerr << "Error: unknown matrix type '"
                          << matrix_type << "'\n";
                print_usage(argv[0]);
                exit_code = 1;
            }

            if (exit_code == 0) {
                std::cout << "Matrix: N=" << global_matrix.rows
                          << " NNZ=" << global_matrix.nnz() << "\n";
            }
          } catch (const std::exception& ex) {
            std::cerr << "[rank 0] Matrix preparation failed: " << ex.what() << "\n";
            exit_code = 1;
          }
        }

        // Root preparation may fail before peers can receive the matrix name.
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (exit_code != 0) {
            MPI_Finalize();
            return exit_code;
        }

        // Broadcast matrix name to all ranks
        {
            std::size_t name_len = (rank == 0) ? matrix_name.size() : 0;
            MPI_Bcast(&name_len, 1, MPI_UNSIGNED_LONG, 0,
                      MPI_COMM_WORLD);
            matrix_name.resize(name_len);
            MPI_Bcast(matrix_name.data(), static_cast<int>(name_len),
                      MPI_CHAR, 0, MPI_COMM_WORLD);
        }

        // Broadcast exit code to all ranks
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (exit_code != 0) {
            MPI_Finalize();
            return exit_code;
        }

        // Distribute matrix to all ranks
        if (rank == 0) {
            std::cout << "Distributing matrix to " << num_ranks
                      << " ranks...\n";
        }

        cg::MPIPartitionedMatrix dist_A =
            cg::partition_matrix(global_matrix, MPI_COMM_WORLD);

        // Validate partition
        cg::validate_partition(dist_A, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "Partition complete. Starting benchmark...\n";
        }

        // Construct b = A * x*, x* = ones (manufactured solution)
        // All ranks construct their local b portion
        // x* is uniform ones: local portion is all-ones vector
        std::vector<double> x_star_local(dist_A.local_rows(), 1.0);

        // b_local = A_local * x* = row sum of local A rows
        // (Distributed SpMV: Allgatherv x_star, then local SpMV)
        std::vector<double> b_local =
            cg::mpi_csr_matvec(dist_A, x_star_local, MPI_COMM_WORLD);

        // Run benchmark
        cg::CGOptions cg_options;
        cg_options.relative_tolerance = 1e-8;
        cg_options.max_iterations = 10000;

        auto records = cg::benchmark_mpi_cg(
            dist_A,
            b_local,
            x_star_local,
            matrix_name,
            repeats,
            cg_options,
            warmups,
            MPI_COMM_WORLD
        );

        // Write CSV (rank 0 only)
        if (rank == 0) {
          try {
            cg::write_mpi_benchmark_csv(records, output_csv, rank);

            std::cout << "\nResults written to: " << output_csv << "\n";

            // Print summary
            std::cout << "\n=== Summary ===\n";
            std::cout << "Ranks: " << num_ranks << "\n";
            std::cout << "Runs: " << records.size() << "\n";

            std::size_t converged_count = 0;
            for (const auto& r : records) {
                if (r.converged) converged_count++;
            }
            std::cout << "Converged: " << converged_count
                      << "/" << records.size() << "\n";

            if (!records.empty()) {
                // Compute median of solve_seconds
                std::vector<double> times;
                times.reserve(records.size());
                for (const auto& r : records) {
                    times.push_back(r.solve_seconds);
                }

                std::sort(times.begin(), times.end());
                const double median = times.size() % 2 == 1
                    ? times[times.size() / 2]
                    : (times[times.size() / 2 - 1] +
                       times[times.size() / 2]) / 2.0;

                std::cout << "Median solve time: " << median << " s\n";
                std::cout << "Iterations: " << records[0].iterations << "\n";
                std::cout << "True rel. residual: "
                          << records[0].true_relative_residual << "\n";
            }
          } catch (const std::exception& ex) {
            std::cerr << "[rank 0] CSV output failed: " << ex.what() << "\n";
            exit_code = 1;
          }
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    } catch (const std::exception& ex) {
        std::cerr << "[rank " << rank << "] Error: " << ex.what() << "\n";
        // Unexpected exceptions may occur on only one rank. Finalize alone
        // cannot release peers blocked in a collective; fail the communicator.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    } catch (...) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return exit_code;
}
