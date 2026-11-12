
/*
 * MPI matrix partitioning tests.
 *
 * Verifies:
 *   - Single-rank partition equals full matrix
 *   - Multi-rank partition distributes rows correctly
 *   - Non-divisible partitions are handled
 *   - Gather/scatter round-trips correctly
 *   - Local CSR row pointers are rebased to start at 0
 *   - Global column indices are preserved
 *
 * Run with 4 ranks: mpirun -n 4 ./test_mpi_partition
 */

#include "mpi/mpi_partition.hpp"
#include "common/csr_matrix.hpp"
#include "common/poisson.hpp"

#include <mpi.h>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

using cg::CSRMatrix;
using cg::MPIPartitionedMatrix;

// Minimal assert that prints on failure
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "CHECK failed at line " << __LINE__ \
                  << " on rank " << rank << ": " #cond "\n"; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Sum local row counts and compare to global
bool total_rows_match(const MPIPartitionedMatrix& part, MPI_Comm comm) {
    std::size_t local = part.local_rows();
    std::size_t total = 0;
    MPI_Allreduce(&local, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    return total == part.global_rows;
}

// Verify each rank's partition matches the partition_offsets
bool partition_ranges_consistent(const MPIPartitionedMatrix& part,
                                 MPI_Comm comm) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    if (part.row_start != part.partition_offsets[rank]) return false;
    if (part.row_end != part.partition_offsets[rank + 1]) return false;
    if (static_cast<int>(part.partition_offsets.size()) != num_ranks + 1)
        return false;
    return true;
}

// Verify that gathering local vectors reconstructs the original
void test_gather_scatter_roundtrip(std::size_t n, MPI_Comm comm) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    CSRMatrix global;
    if (rank == 0) {
        global = cg::make_poisson_2d(n);
    }

    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    // Create a global vector on rank 0: v[i] = i
    std::vector<double> global_vec;
    if (rank == 0) {
        global_vec.resize(part.global_rows);
        for (std::size_t i = 0; i < part.global_rows; ++i) {
            global_vec[i] = static_cast<double>(i);
        }
    }

    // Scatter: rank 0 -> all ranks
    std::vector<double> local_vec = cg::scatter_vector(
        global_vec, part, comm
    );

    // Verify local portion matches expected values
    for (std::size_t i = 0; i < local_vec.size(); ++i) {
        CHECK(local_vec[i] ==
              static_cast<double>(part.row_start + i));
    }

    // Gather back: all ranks -> rank 0
    std::vector<double> gathered = cg::gather_vector(
        local_vec, part, comm
    );

    // Verify roundtrip on rank 0
    if (rank == 0) {
        CHECK(gathered.size() == global_vec.size());
        for (std::size_t i = 0; i < gathered.size(); ++i) {
            CHECK(gathered[i] == global_vec[i]);
        }
    }
}

// Verify partition of 2D Poisson with various rank counts
void test_poisson_partition(std::size_t n, MPI_Comm comm) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    CSRMatrix global;
    if (rank == 0) {
        global = cg::make_poisson_2d(n);
    }

    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    // Validate basic properties
    CHECK(part.global_rows == n * n);
    CHECK(part.global_cols == n * n);
    CHECK(part.local_rows() == part.row_end - part.row_start);
    CHECK(total_rows_match(part, comm));
    CHECK(partition_ranges_consistent(part, comm));

    // Validate local CSR structure
    part.local_matrix.validate_structure();

    // Verify row_ptr is rebased to start at 0
    CHECK(part.local_matrix.row_ptr.front() == 0);
    CHECK(part.local_matrix.row_ptr.back() ==
          part.local_matrix.nnz());

    // Verify column indices are within global bounds
    for (auto col : part.local_matrix.col_idx) {
        CHECK(col < part.global_cols);
    }

    // Verify partition_offsets are non-decreasing and cover full range
    for (int r = 0; r < num_ranks; ++r) {
        CHECK(part.partition_offsets[r] <= part.partition_offsets[r + 1]);
    }
    CHECK(part.partition_offsets[0] == 0);
    CHECK(part.partition_offsets[num_ranks] == part.global_rows);
}

// Test non-divisible partition (10 rows, variable ranks)
void test_non_divisible(MPI_Comm comm) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    // n=8 -> N=64, divisible by 4 ranks (16 each)
    // n=9 -> N=81, not divisible by 4 (21,21,21,18 or similar)
    CSRMatrix global;
    if (rank == 0) {
        global = cg::make_poisson_2d(9);  // 81 rows
    }

    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    CHECK(part.global_rows == 81);
    CHECK(total_rows_match(part, comm));
    CHECK(partition_ranges_consistent(part, comm));
    part.local_matrix.validate_structure();

    // Last rank should have fewer rows for non-divisible case
    // rows_per_rank = ceil(81/4) = 21
    // Rank 0,1,2: 21 rows each; Rank 3: 81-63 = 18 rows
    if (num_ranks == 4) {
        for (int r = 0; r < 3; ++r) {
            CHECK(part.partition_offsets[r + 1] -
                  part.partition_offsets[r] == 21);
        }
        CHECK(part.partition_offsets[4] -
              part.partition_offsets[3] == 18);
    }
}

// Single-rank test: partition should equal original matrix
void test_single_rank_properties(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    CSRMatrix global;
    if (rank == 0) {
        global = cg::make_poisson_2d(8);
    }

    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    // With 1 rank, local should equal global
    int num_ranks;
    MPI_Comm_size(comm, &num_ranks);

    if (num_ranks == 1) {
        CHECK(part.local_rows() == part.global_rows);
        CHECK(part.local_matrix.nnz() == global.nnz());
        CHECK(part.row_start == 0);
        CHECK(part.row_end == part.global_rows);
    }

    // Always valid
    part.local_matrix.validate_structure();
    CHECK(total_rows_match(part, comm));
}

// Verify local row sums match global row sums (data integrity)
void test_row_data_integrity(std::size_t n, MPI_Comm comm) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    CSRMatrix global;
    if (rank == 0) {
        global = cg::make_poisson_2d(n);
    }

    MPIPartitionedMatrix part = cg::partition_matrix(global, comm);

    // Compute local row sums
    std::vector<double> local_row_sums(part.local_rows(), 0.0);
    for (std::size_t i = 0; i < part.local_rows(); ++i) {
        const auto& mat = part.local_matrix;
        for (std::size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j) {
            local_row_sums[i] += mat.values[j];
        }
    }

    // Gather row sums to rank 0
    std::vector<double> global_row_sums = cg::gather_vector(
        local_row_sums, part, comm
    );

    // On rank 0, compare against original matrix row sums
    if (rank == 0) {
        for (std::size_t i = 0; i < global.rows; ++i) {
            double expected_sum = 0.0;
            for (std::size_t j = global.row_ptr[i];
                 j < global.row_ptr[i + 1]; ++j) {
                expected_sum += global.values[j];
            }
            CHECK(global_row_sums[i] == expected_sum);
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (rank == 0) {
        std::cout << "=== MPI Partition Tests (" << num_ranks
                  << " ranks) ===\n";
    }

    // Test 1: Single rank properties (works with any rank count)
    test_single_rank_properties(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "PASSED: single rank properties\n";

    // Test 2: 2D Poisson 16x16 (256 rows)
    test_poisson_partition(16, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "PASSED: poisson 16x16 partition\n";

    // Test 3: 2D Poisson 32x32 (1024 rows)
    test_poisson_partition(32, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "PASSED: poisson 32x32 partition\n";

    // Test 4: Non-divisible partition (81 rows)
    test_non_divisible(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "PASSED: non-divisible partition\n";

    // Test 5: Gather/scatter roundtrip
    test_gather_scatter_roundtrip(16, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "PASSED: gather/scatter roundtrip\n";

    // Test 6: Row data integrity
    test_row_data_integrity(16, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "PASSED: row data integrity\n";

    // Test 7: validate_partition should not throw
    {
        CSRMatrix global;
        if (rank == 0) {
            global = cg::make_poisson_2d(16);
        }
        MPIPartitionedMatrix part = cg::partition_matrix(global,
                                                          MPI_COMM_WORLD);
        cg::validate_partition(part, MPI_COMM_WORLD);
    }
    if (rank == 0) std::cout << "PASSED: validate_partition\n";

    if (rank == 0) {
        std::cout << "\nAll MPI partition tests passed!\n";
    }

    MPI_Finalize();
    return 0;
}
