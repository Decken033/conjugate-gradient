#include "hybrid/hybrid_ops.hpp"
#include "mpi/mpi_ops.hpp"
#include "mpi/mpi_ops.hpp"
#include "common/csr_ops.hpp"
#include "common/poisson.hpp"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int failed = 0;
    try {
        cg::CSRMatrix global;
        if (rank == 0) global = cg::make_poisson_2d(12);
        const auto A = cg::partition_matrix(global, MPI_COMM_WORLD);
        cg::OpenMPOptions omp;
        omp.num_threads = 2;
        omp.schedule = cg::OpenMPSchedule::Dynamic;
        omp.chunk_size = 2;

        const std::vector<double> x_local(A.local_rows(), 1.0);

        // V1: exercise the reusable-workspace overload and verify that the
        // SpMV output buffer keeps the same storage across repeated calls.
        auto workspace = cg::make_hybrid_workspace(A, MPI_COMM_WORLD);
        const auto& y_workspace_first =
            cg::hybrid_csr_matvec(A, x_local, workspace, omp, MPI_COMM_WORLD);
        const double* first_output_storage = y_workspace_first.data();
        const auto& y_workspace_second =
            cg::hybrid_csr_matvec(A, x_local, workspace, omp, MPI_COMM_WORLD);
        if (y_workspace_second.data() != first_output_storage)
            throw std::runtime_error("Hybrid workspace SpMV output buffer was reallocated");

        const std::vector<double> y_hybrid(y_workspace_second.begin(), y_workspace_second.end());
        const auto y_reference = cg::mpi_csr_matvec(A, x_local, MPI_COMM_WORLD);
        double local_diff = 0.0;
        for (std::size_t i = 0; i < y_hybrid.size(); ++i)
            local_diff = std::max(local_diff, std::abs(y_hybrid[i] - y_reference[i]));
        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (global_diff > 1e-12) throw std::runtime_error("Hybrid SpMV differs from MPI SpMV");

        const double d = cg::hybrid_dot(x_local, x_local, omp, MPI_COMM_WORLD);
        if (std::abs(d - static_cast<double>(A.global_rows)) > 1e-10)
            throw std::runtime_error("Hybrid distributed dot product failed");

        if (rank == 0) std::cout << "Hybrid Ops passed: SpMV and distributed dot product\n";
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " test failure: " << e.what() << '\n';
        failed = 1;
    }
    int global_failed = 0;
    MPI_Allreduce(&failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_failed;
}
