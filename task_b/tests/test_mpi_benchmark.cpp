#include "mpi_benchmark.hpp"
#include "mpi_ops.hpp"
#include "poisson.hpp"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        if (ranks != 4) throw std::runtime_error("test requires four ranks");
        cg::CSRMatrix global;
        if (rank == 0) global = cg::make_poisson_2d(3);
        auto part = cg::partition_matrix(global, MPI_COMM_WORLD);
        if (rank == 3 && part.local_rows() != 0)
            throw std::runtime_error("expected an empty final rank");
        std::vector<double> exact(part.local_rows(), 1.0);
        auto b = cg::mpi_csr_matvec(part, exact, MPI_COMM_WORLD);
        cg::CGOptions options;
        options.relative_tolerance = 1e-8;
        options.max_iterations = 0;
        auto records = cg::benchmark_mpi_cg(part, b, exact, "zero_limit", 2,
                                           options, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (records.size() != 2) throw std::runtime_error("wrong record count");
            for (const auto& record : records)
                if (record.converged || record.iterations != 0 ||
                    !std::isfinite(record.relative_solution_error) ||
                    !std::isfinite(record.true_relative_residual) ||
                    std::abs(record.relative_solution_error - 1.0) > 1e-12 ||
                    std::abs(record.true_relative_residual - 1.0) > 1e-12)
                    throw std::runtime_error("incorrect nonconvergence record");
        }
        options.max_iterations = 10000;
        records = cg::benchmark_mpi_cg(part, b, exact, "recovery", 1,
                                      options, 0, MPI_COMM_WORLD);
        if (rank == 0 && (records.size() != 1 || !records[0].converged ||
                         !std::isfinite(records[0].true_relative_residual) ||
                         records[0].true_relative_residual > 1e-8))
            throw std::runtime_error("valid solve after zero-limit solve failed");
        if (rank == 0) std::cout << "Empty-rank benchmark checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
