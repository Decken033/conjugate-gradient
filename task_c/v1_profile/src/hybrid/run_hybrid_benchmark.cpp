#include "hybrid/hybrid_benchmark.hpp"
#include "common/ops_profiler.hpp"
#include "common/matrix_market.hpp"
#include "common/poisson.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parse_size(const char* text, bool allow_zero) {
    const std::string s(text);
    if (s.empty() || !std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); }))
        throw std::invalid_argument("Expected a nonnegative integer: " + s);
    const auto value = std::stoull(s);
    if (!allow_zero && value == 0) throw std::invalid_argument("Value must be positive: " + s);
    if (value > std::numeric_limits<std::size_t>::max()) throw std::out_of_range("Integer too large: " + s);
    return static_cast<std::size_t>(value);
}

cg::OpenMPSchedule parse_schedule(const std::string& s) {
    if (s == "static") return cg::OpenMPSchedule::Static;
    if (s == "dynamic") return cg::OpenMPSchedule::Dynamic;
    if (s == "guided") return cg::OpenMPSchedule::Guided;
    throw std::invalid_argument("Schedule must be static, dynamic or guided");
}

void usage(const char* exe) {
    std::cerr << "Usage: mpirun -np <ranks> " << exe
              << " <2d|3d|thermal2> <grid_size|matrix.mtx> <threads>"
                 " <static|dynamic|guided> <chunk_size> <repeats> <output.csv> [warmups]\n";
}

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int exit_code = 0;
    std::vector<cg::HybridBenchmarkRecord> records;
    std::string matrix_name;
    std::string output_path;

    try {
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI implementation does not provide MPI_THREAD_FUNNELED");
        }
        if (argc != 8 && argc != 9) {
            if (rank == 0) usage(argv[0]);
            MPI_Finalize();
            return 1;
        }

        const std::string matrix_type = argv[1];
        const std::string matrix_arg = argv[2];
        const auto thread_count = parse_size(argv[3], false);
        const auto chunk = parse_size(argv[5], true);
        const auto repeats = parse_size(argv[6], false);
        const auto warmups = argc == 9 ? parse_size(argv[8], true) : 1;
        if (thread_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            chunk > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::out_of_range("OpenMP threads/chunk exceed int range");
        const auto schedule = parse_schedule(argv[4]);
        output_path = argv[7];

        cg::CSRMatrix global_matrix;
        int matrix_load_error = 0;
        if (rank == 0) {
            try {
                if (matrix_type == "2d" || matrix_type == "3d") {
                    const auto n = parse_size(argv[2], false);
                    global_matrix = matrix_type == "2d"
                        ? cg::make_poisson_2d(n) : cg::make_poisson_3d(n);
                    matrix_name = "poisson_" + matrix_type + "_" + matrix_arg;
                } else if (matrix_type == "thermal2") {
                    global_matrix = cg::load_matrix_market(matrix_arg);
                    matrix_name = "thermal2";
                } else {
                    throw std::invalid_argument("Matrix type must be 2d, 3d or thermal2");
                }
            } catch (const std::exception& e) {
                std::cerr << "Matrix setup failed: " << e.what() << '\n';
                matrix_load_error = 1;
            }
        }
        MPI_Bcast(&matrix_load_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (matrix_load_error) {
            MPI_Finalize();
            return 1;
        }

        // Broadcast root-side matrix loading status/name before collective partitioning.
        unsigned long long name_len = rank == 0 ? static_cast<unsigned long long>(matrix_name.size()) : 0ULL;
        MPI_Bcast(&name_len, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        matrix_name.resize(static_cast<std::size_t>(name_len));
        MPI_Bcast(matrix_name.data(), static_cast<int>(name_len), MPI_CHAR, 0, MPI_COMM_WORLD);

        auto dist_A = cg::partition_matrix(global_matrix, MPI_COMM_WORLD);
        cg::validate_partition(dist_A, MPI_COMM_WORLD);

        cg::OpenMPOptions omp_options;
        omp_options.num_threads = static_cast<int>(thread_count);
        omp_options.schedule = schedule;
        omp_options.chunk_size = static_cast<int>(chunk);

        // Manufactured solution x*=1; each rank owns its local segment.
        const std::vector<double> exact_local(dist_A.local_rows(), 1.0);
        const auto b_local = cg::hybrid_csr_matvec(dist_A, exact_local, omp_options, MPI_COMM_WORLD);

        cg::CGOptions cg_options;
        cg_options.relative_tolerance = 1e-8;
        cg_options.max_iterations = 10000;

        records = cg::benchmark_hybrid_cg(
            dist_A, b_local, exact_local, matrix_name, repeats,
            cg_options, omp_options, warmups, MPI_COMM_WORLD);

        cg::write_ops_profile_csv(cg::process_ops_profile, output_path + ".profile.rank" + std::to_string(rank) + ".csv", "hybrid", rank, ranks, static_cast<int>(thread_count));

        if (rank == 0) {
            cg::write_hybrid_benchmark_csv(records, output_path, rank);
            std::cout << "Hybrid benchmark written to " << output_path
                      << " | ranks=" << ranks
                      << " | threads/rank=" << thread_count
                      << " | total MPI+OpenMP workers=" << ranks * static_cast<int>(thread_count)
                      << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " hybrid benchmark error: " << e.what() << '\n';
        exit_code = 1;
    }

    int global_exit = 0;
    MPI_Allreduce(&exit_code, &global_exit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_exit;
}
