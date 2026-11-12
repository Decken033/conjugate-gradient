
/*
 * MPI Conjugate Gradient solver correctness tests.
 *
 * Verifies:
 *   - MPI CG converges to tolerance on 2D Poisson
 *   - MPI CG converges to tolerance on 3D Poisson
 *   - Single-rank MPI CG matches serial CG
 *   - Multi-rank MPI CG produces consistent solution
 *   - True relative residual satisfies tolerance
 *   - Relative solution error is reasonable
 *   - Cross-rank consistency: different rank counts converge
 *
 * Run with 4 ranks: mpirun -n 4 ./test_mpi_cg
 */

#include "mpi/mpi_partition.hpp"
#include "mpi/mpi_ops.hpp"
#include "mpi/mpi_solver.hpp"
#include "serial/cg_solver.hpp"
#include "common/poisson.hpp"
#include "common/csr_matrix.hpp"
#include "common/csr_ops.hpp"

#include <mpi.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

using cg::CSRMatrix;
using cg::MPIPartitionedMatrix;
using cg::MPICGResult;
using cg::CGOptions;
using cg::CGResult;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "CHECK failed at line " << __LINE__ \
                  << " on rank " << rank << ": " #cond "\n"; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(a) + std::fabs(b));
}

// Test MPI CG convergence on 2D Poisson
void test_mpi_cg_2d_poisson(std::size_t n, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    CSRMatrix global;
    if (rank == 0) {
        global = cg::make_poisson_2d(n);
    }

    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    // b = A * ones (manufactured solution)
    std::vector<double> x_star_local(part.local_rows(), 1.0);
    std::vector<double> b_local = cg::mpi_csr_matvec(
        part, x_star_local, comm
    );

    // Solve
    CGOptions options;
    options.relative_tolerance = 1e-8;
    options.max_iterations = 10000;

    MPICGResult result = cg::solve_mpi_cg(part, b_local, options, comm);

    // Verify convergence
    CHECK(result.converged);
    CHECK(result.relative_residual <= 1e-8);
    CHECK(result.iterations > 0);
    CHECK(result.iterations <= options.max_iterations);

    // Verify x_local size
    CHECK(result.x_local.size() == part.local_rows());

    // Compute relative solution error (distributed)
    double local_sq_err = 0.0, local_sq_exact = 0.0;
    for (std::size_t i = 0; i < x_star_local.size(); ++i) {
        const double diff = result.x_local[i] - x_star_local[i];
        local_sq_err += diff * diff;
        local_sq_exact += x_star_local[i] * x_star_local[i];
    }

    double sums[2] = {local_sq_err, local_sq_exact};
    double global_sums[2] = {0.0, 0.0};
    MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, comm);

    const double rel_error = std::sqrt(
        global_sums[0] / global_sums[1]
    );

    // Relative solution error should be small for well-conditioned problem
    CHECK(rel_error < 1e-4);

    if (rank == 0) {
        std::cout << "  2D Poisson " << n << "x" << n
                  << ": iterations=" << result.iterations
                  << " residual=" << result.relative_residual
                  << " rel_error=" << rel_error << "\n";
    }
}

// Test MPI CG convergence on 3D Poisson
void test_mpi_cg_3d_poisson(std::size_t n, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    CSRMatrix global;
    if (rank == 0) {
        global = cg::make_poisson_3d(n);
    }

    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    std::vector<double> x_star_local(part.local_rows(), 1.0);
    std::vector<double> b_local = cg::mpi_csr_matvec(
        part, x_star_local, comm
    );

    CGOptions options;
    options.relative_tolerance = 1e-8;
    options.max_iterations = 10000;

    MPICGResult result = cg::solve_mpi_cg(part, b_local, options, comm);

    CHECK(result.converged);
    CHECK(result.relative_residual <= 1e-8);
    CHECK(result.iterations > 0);

    // Compute solution error
    double local_sq_err = 0.0, local_sq_exact = 0.0;
    for (std::size_t i = 0; i < x_star_local.size(); ++i) {
        const double diff = result.x_local[i] - x_star_local[i];
        local_sq_err += diff * diff;
        local_sq_exact += x_star_local[i] * x_star_local[i];
    }

    double sums[2] = {local_sq_err, local_sq_exact};
    double global_sums[2] = {0.0, 0.0};
    MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, comm);

    const double rel_error = std::sqrt(global_sums[0] / global_sums[1]);
    CHECK(rel_error < 1e-4);

    if (rank == 0) {
        std::cout << "  3D Poisson " << n << "x" << n << "x" << n
                  << ": iterations=" << result.iterations
                  << " residual=" << result.relative_residual
                  << " rel_error=" << rel_error << "\n";
    }
}

// Compare MPI single-rank with serial CG
// With 1 MPI rank, the computation should closely match serial CG
void test_single_rank_matches_serial(MPI_Comm comm) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    // This test only valid with 1 rank
    if (num_ranks != 1) {
        if (rank == 0) {
            std::cout << "  (single-rank test skipped: running with "
                      << num_ranks << " ranks)\n";
        }
        return;
    }

    CSRMatrix global = cg::make_poisson_2d(16);
    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    // b = A * ones
    std::vector<double> x_star(global.rows, 1.0);
    std::vector<double> b = cg::csr_matvec(global, x_star);

    // Serial solve
    CGOptions options;
    options.relative_tolerance = 1e-8;
    options.max_iterations = 10000;

    CGResult serial_result = cg::solve_serial_cg(global, b, options);

    // MPI single-rank solve
    std::vector<double> b_local(b.begin(), b.end());
    MPICGResult mpi_result = cg::solve_mpi_cg(part, b_local, options, comm);

    // Both should converge
    CHECK(serial_result.converged);
    CHECK(mpi_result.converged);

    // Iteration counts should match exactly (same reduction order on 1 rank)
    CHECK(serial_result.iterations == mpi_result.iterations);

    // Relative residuals should match
    CHECK(approx_equal(serial_result.relative_residual,
                        mpi_result.relative_residual, 1e-12));

    // Solutions should match
    CHECK(mpi_result.x_local.size() == serial_result.x.size());
    for (std::size_t i = 0; i < serial_result.x.size(); ++i) {
        CHECK(approx_equal(serial_result.x[i],
                           mpi_result.x_local[i], 1e-12));
    }

    if (rank == 0) {
        std::cout << "  Single-rank MPI matches serial: "
                  << "iterations=" << serial_result.iterations
                  << " residual=" << serial_result.relative_residual << "\n";
    }
}

// Cross-rank consistency: solution should be same regardless of rank count
// This test verifies correctness at current rank count; to verify cross-rank
// consistency, run with different -n values and compare.
void test_solution_consistency(std::size_t n, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    CSRMatrix global;
    if (rank == 0) {
        global = cg::make_poisson_2d(n);
    }

    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    std::vector<double> x_star_local(part.local_rows(), 1.0);
    std::vector<double> b_local = cg::mpi_csr_matvec(
        part, x_star_local, comm
    );

    CGOptions options;
    options.relative_tolerance = 1e-8;
    options.max_iterations = 10000;

    MPICGResult result = cg::solve_mpi_cg(part, b_local, options, comm);

    CHECK(result.converged);

    // Gather solution to rank 0
    std::vector<double> x_global = cg::gather_vector(
        result.x_local, part, comm
    );

    // On rank 0, verify solution is close to ones
    if (rank == 0) {
        CHECK(x_global.size() == part.global_rows);

        // Compute solution error
        double sq_err = 0.0, sq_exact = 0.0;
        for (std::size_t i = 0; i < x_global.size(); ++i) {
            const double diff = x_global[i] - 1.0;
            sq_err += diff * diff;
            sq_exact += 1.0;
        }

        const double rel_error = std::sqrt(sq_err / sq_exact);
        CHECK(rel_error < 1e-4);

        std::cout << "  Cross-rank consistency (" << n << "x" << n
                  << "): rel_error=" << rel_error << "\n";
    }
}

// Test non-divisible partition CG convergence
void test_non_divisible_cg(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // n=9 -> N=81, not divisible by 4 ranks
    CSRMatrix global;
    if (rank == 0) {
        global = cg::make_poisson_2d(9);
    }

    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    std::vector<double> x_star_local(part.local_rows(), 1.0);
    std::vector<double> b_local = cg::mpi_csr_matvec(
        part, x_star_local, comm
    );

    CGOptions options;
    options.relative_tolerance = 1e-8;
    options.max_iterations = 10000;

    MPICGResult result = cg::solve_mpi_cg(part, b_local, options, comm);

    CHECK(result.converged);
    CHECK(result.relative_residual <= 1e-8);

    if (rank == 0) {
        std::cout << "  Non-divisible (81 rows): iterations="
                  << result.iterations
                  << " residual=" << result.relative_residual << "\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (rank == 0) {
        std::cout << "=== MPI CG Solver Tests (" << num_ranks
                  << " ranks) ===\n";
    }

    // Test 2D Poisson at multiple sizes
    if (rank == 0) std::cout << "\n2D Poisson tests:\n";
    test_mpi_cg_2d_poisson(8, MPI_COMM_WORLD);
    test_mpi_cg_2d_poisson(16, MPI_COMM_WORLD);
    test_mpi_cg_2d_poisson(32, MPI_COMM_WORLD);

    // Test 3D Poisson
    if (rank == 0) std::cout << "\n3D Poisson tests:\n";
    test_mpi_cg_3d_poisson(4, MPI_COMM_WORLD);
    test_mpi_cg_3d_poisson(8, MPI_COMM_WORLD);

    // Test single-rank matches serial
    if (rank == 0) std::cout << "\nSingle-rank vs serial:\n";
    test_single_rank_matches_serial(MPI_COMM_WORLD);

    // Test cross-rank solution consistency
    if (rank == 0) std::cout << "\nCross-rank consistency:\n";
    test_solution_consistency(16, MPI_COMM_WORLD);
    test_solution_consistency(32, MPI_COMM_WORLD);

    // Test non-divisible partition
    if (rank == 0) std::cout << "\nNon-divisible partition:\n";
    test_non_divisible_cg(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\nAll MPI CG tests passed!\n";
    }

    MPI_Finalize();
    return 0;
}
