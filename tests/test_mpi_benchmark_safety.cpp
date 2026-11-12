// Registered MPI regression test at 1, 2 and 4 ranks.
// CTest timeout bounds collective-mismatch regressions.
#include "mpi_benchmark.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void check(bool condition, MPI_Comm comm, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        MPI_Abort(comm, 1);
    }
}

// One global identity row, owned by the last rank. In multi-rank runs,
// even the reporting rank has an empty partition.
cg::MPIPartitionedMatrix single_row_partition(int rank, int size) {
    cg::MPIPartitionedMatrix A;
    A.global_rows = A.global_cols = 1;
    A.partition_offsets.assign(static_cast<std::size_t>(size) + 1, 0);
    A.partition_offsets.back() = 1;
    A.row_start = A.partition_offsets[rank];
    A.row_end = A.partition_offsets[rank + 1];
    A.local_matrix.rows = A.local_rows();
    A.local_matrix.cols = 1;
    A.local_matrix.row_ptr = {0};
    if (A.local_rows() != 0) {
        A.local_matrix.row_ptr.push_back(1);
        A.local_matrix.col_idx = {0};
        A.local_matrix.values = {1.0};
    }
    return A;
}

template<class Call>
void expect_collective_rejection(Call call, MPI_Comm comm) {
    int rejected = 0;
    try {
        call();
    } catch (const std::invalid_argument&) {
        rejected = 1;
    }
    int all_rejected = 0;
    MPI_Allreduce(&rejected, &all_rejected, 1, MPI_INT, MPI_MIN, comm);
    check(all_rejected == 1, comm, "Expected invalid_argument on every rank");
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const auto A = single_row_partition(rank, size);
    const std::vector<double> b(A.local_rows(), 2.0);
    const std::vector<double> exact(A.local_rows(), 1.0);
    const cg::CGOptions options;

    // Deliberately choose x*=1 while the actual solution is 2. This tests
    // the nonzero global error, not just that an all-zero norm is returned.
    const auto records = cg::benchmark_mpi_cg(A, b, exact, "one-row", 2,
                                             options, 1, comm);
    if (rank == 0) {
        check(records.size() == 2, comm, "Incorrect record count");
        for (std::size_t i = 0; i < records.size(); ++i) {
            check(records[i].run_index == i && records[i].converged &&
                  records[i].nnz == 1 && records[i].n == 1 &&
                  std::fabs(records[i].relative_solution_error - 1.0) < 1e-12 &&
                  std::isfinite(records[i].solve_seconds) &&
                  records[i].solve_seconds >= 0.0,
                  comm, "Incorrect empty-partition benchmark record");
        }
    } else {
        check(records.empty(), comm, "Nonroot rank returned records");
    }

    const std::vector<double> zero(A.local_rows(), 0.0);
    const auto zero_records = cg::benchmark_mpi_cg(A, zero, zero, "zero-rhs",
                                                  1, options, 0, comm);
    if (rank == 0) {
        check(zero_records.size() == 1 && zero_records[0].converged &&
              zero_records[0].iterations == 0 &&
              zero_records[0].true_relative_residual == 0.0 &&
              zero_records[0].relative_solution_error == 0.0,
              comm, "Incorrect zero RHS benchmark result");
    }

    for (int test = 0; test < 4; ++test) {
        auto bad_A = A;
        if (rank == 0) {
            if (test == 0) bad_A.partition_offsets.clear();
            if (test == 1) bad_A.local_matrix.row_ptr.clear();
            if (test == 2) ++bad_A.row_end;
            if (test == 3) ++bad_A.global_cols;
        }
        expect_collective_rejection([&] {
            cg::benchmark_mpi_cg(bad_A, b, exact, "invalid-partition", 1,
                                 options, 0, comm);
        }, comm);
    }

    for (int test = 0; test < 6; ++test) {
        auto bad_b = b;
        auto bad_exact = exact;
        auto bad_options = options;
        std::size_t repeats = 1;
        if (rank == size - 1) {
            if (test == 0) repeats = 0;
            if (test == 1) bad_b.push_back(0.0);
            if (test == 2) bad_exact.clear();
            if (test == 3) bad_options.relative_tolerance = 0.0;
            if (test == 4) bad_b[0] = std::numeric_limits<double>::infinity();
            if (test == 5) bad_exact[0] = std::numeric_limits<double>::quiet_NaN();
        }
        expect_collective_rejection([&] {
            cg::benchmark_mpi_cg(A, bad_b, bad_exact, "invalid", repeats,
                                 bad_options, 0, comm);
        }, comm);
    }

    if (size > 1) {
        for (int test = 0; test < 4; ++test) {
            auto changed_options = options;
            std::size_t repeats = 1, warmups = 0;
            if (rank == size - 1) {
                if (test == 0) repeats = 2;
                if (test == 1) warmups = 1;
                if (test == 2) ++changed_options.max_iterations;
                if (test == 3) changed_options.relative_tolerance *= 2.0;
            }
            expect_collective_rejection([&] {
                cg::benchmark_mpi_cg(A, b, exact, "mismatch", repeats,
                                     changed_options, warmups, comm);
            }, comm);
        }
    } else {
        std::cout << "Multi-rank mismatch cases skipped (requires >=2 ranks)\n";
    }
    if (rank == 0) std::cout << "MPI benchmark safety tests passed\n";
    MPI_Finalize();
}
