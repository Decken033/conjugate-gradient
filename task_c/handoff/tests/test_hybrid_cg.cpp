#include "hybrid/hybrid_solver.hpp"
#include "common/poisson.hpp"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int failed = 0;
    try {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI_THREAD_FUNNELED unavailable");

        cg::CSRMatrix global;
        if (rank == 0) global = cg::make_poisson_2d(16);
        const auto A = cg::partition_matrix(global, MPI_COMM_WORLD);
        cg::validate_partition(A, MPI_COMM_WORLD);

        cg::OpenMPOptions omp;
        omp.num_threads = 2;
        omp.schedule = cg::OpenMPSchedule::Static;
        const std::vector<double> exact_local(A.local_rows(), 1.0);
        const auto b_local = cg::hybrid_csr_matvec(A, exact_local, omp, MPI_COMM_WORLD);

        cg::CGOptions options;
        options.relative_tolerance = 1e-8;
        options.max_iterations = 10000;
        const auto result = cg::solve_hybrid_cg(A, b_local, options, omp, MPI_COMM_WORLD);
        if (!result.converged) throw std::runtime_error("Hybrid CG did not converge");

        double local_err_sq = 0.0;
        for (double x : result.x_local) local_err_sq += (x - 1.0) * (x - 1.0);
        double global_err_sq = 0.0;
        MPI_Allreduce(&local_err_sq, &global_err_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        const double rel_error = std::sqrt(global_err_sq / static_cast<double>(A.global_rows));
        if (rel_error > 1e-7 || result.relative_residual > 1e-8)
            throw std::runtime_error("Hybrid CG accuracy check failed");

        if (rank == 0)
            std::cout << "Hybrid CG passed: ranks=" << size
                      << " threads/rank=2 iterations=" << result.iterations
                      << " residual=" << result.relative_residual
                      << " solution_error=" << rel_error << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " test failure: " << e.what() << '\n';
        failed = 1;
    }

    int global_failed = 0;
    MPI_Allreduce(&failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && !global_failed) std::cout << "All Hybrid CG tests passed!\n";
    MPI_Finalize();
    return global_failed;
}
