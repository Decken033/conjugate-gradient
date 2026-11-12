
#pragma once

#include "common/csr_matrix.hpp"

#include <mpi.h>
#include <cstddef>
#include <vector>

namespace cg {

// Distributed CSR matrix using 1D row-wise partitioning.
//
// Each MPI rank owns a contiguous block of matrix rows.
// Global column indices are preserved in local storage.
//
// Design rationale:
// - Row partitioning enables independent SpMV computation per rank
// - Contiguous blocks simplify rank-to-row mapping
// - Global column indices avoid index translation during SpMV
//
// Memory: Each rank stores only its assigned rows.
struct MPIPartitionedMatrix {
    // Local CSR storage for this rank's rows.
    // col_idx uses global column indices [0, global_cols).
    CSRMatrix local_matrix;

    // Global matrix dimensions.
    std::size_t global_rows = 0;
    std::size_t global_cols = 0;

    // This rank's global row range: [row_start, row_end).
    std::size_t row_start = 0;
    std::size_t row_end = 0;

    // Partition boundaries for all ranks.
    // partition_offsets[r] = first global row owned by rank r.
    // partition_offsets[P] = global_rows.
    // Size: P+1 where P = number of MPI ranks.
    std::vector<std::size_t> partition_offsets;

    std::size_t local_rows() const noexcept {
        return row_end - row_start;
    }
};

// Distribute a global CSR matrix among MPI ranks using row-wise partitioning.
//
// Rank 0 must provide the full validated global_matrix.
// Other ranks should pass an empty/default CSRMatrix.
//
// Row partitioning:
//   rows_per_rank = ceil(N / P)
//   rank r owns rows [r * rows_per_rank, min((r+1) * rows_per_rank, N))
//
// All ranks receive a valid MPIPartitionedMatrix with:
//   - local_matrix containing their assigned CSR rows
//   - global dimensions
//   - partition_offsets for all ranks
//
// Throws if:
//   - global_matrix validation fails on rank 0
//   - MPI communication fails
//   - P > N (more ranks than rows)
//
// Type note: CSR index_type is std::size_t. On LP64 Linux (Vanda),
// std::size_t is 64-bit and is sent as MPI_UNSIGNED_LONG_LONG.
// Both types have identical binary representation on LP64 platforms.
//
// Collective: all ranks in comm must call this function.
MPIPartitionedMatrix partition_matrix(
    const CSRMatrix& global_matrix,
    MPI_Comm comm = MPI_COMM_WORLD
);

// Validate distributed matrix partition integrity.
//
// Checks:
//   - Sum of local row counts equals global_rows
//   - Partition boundaries are consistent across ranks
//   - Local CSR structure is valid
//   - Global dimensions match across ranks
//
// Throws on validation failure (collective).
//
// Collective: all ranks must call.
void validate_partition(
    const MPIPartitionedMatrix& part,
    MPI_Comm comm = MPI_COMM_WORLD
);

// Gather local vector portions to reconstruct full vector on rank 0.
//
// Input: local_vec must have size = part.local_rows() on each rank.
// Output: Full vector of size global_rows on rank 0; empty on other ranks.
//
// Note: Uses MPI_Gatherv with int displacement/count arrays.
// Maximum safe vector size: INT_MAX elements (~2^31 doubles).
// Larger global vectors require MPI large-count functions (MPI 4.0+).
//
// Collective: all ranks must call.
std::vector<double> gather_vector(
    const std::vector<double>& local_vec,
    const MPIPartitionedMatrix& part,
    MPI_Comm comm = MPI_COMM_WORLD
);

// Scatter global vector from rank 0 to all ranks.
//
// Input: global_vec must have size global_rows on rank 0; ignored on other ranks.
// Output: Local portion of size local_rows() on each rank.
//
// Note: Uses MPI_Scatterv with int displacement/count arrays.
// Maximum safe vector size: INT_MAX elements.
//
// Collective: all ranks must call.
std::vector<double> scatter_vector(
    const std::vector<double>& global_vec,
    const MPIPartitionedMatrix& part,
    MPI_Comm comm = MPI_COMM_WORLD
);

} // namespace cg
