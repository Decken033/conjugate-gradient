
#include "mpi_partition.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstddef>
#include <exception>
#include <limits>

namespace cg {
namespace {

// No rank may leave validation while peers proceed into a collective.
void agree_validation(std::exception_ptr error, MPI_Comm comm) {
    int failed = error ? 1 : 0;
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
    if (any_failed) {
        if (error) std::rethrow_exception(error);
        throw std::invalid_argument("MPI partition validation failed on another rank");
    }
}

// Compute partition boundaries for P ranks and N rows.
// Returns vector of size P+1 where entry r is the first row for rank r.
std::vector<std::size_t> compute_partition_offsets(
    std::size_t global_rows,
    int num_ranks
) {
    std::vector<std::size_t> offsets(num_ranks + 1);

    const std::size_t rows_per_rank =
        (global_rows + num_ranks - 1) / num_ranks;

    for (int r = 0; r < num_ranks; ++r) {
        offsets[r] = r * rows_per_rank;
        if (offsets[r] > global_rows) {
            offsets[r] = global_rows;
        }
    }
    offsets[num_ranks] = global_rows;

    return offsets;
}

// Extract CSR rows [row_start, row_end) from global matrix.
// Returns local CSR with rebased row_ptr starting at 0.
CSRMatrix extract_csr_rows(
    const CSRMatrix& global,
    std::size_t row_start,
    std::size_t row_end
) {
    if (row_start > row_end || row_end > global.rows) {
        throw std::invalid_argument(
            "Invalid row range for CSR extraction"
        );
    }

    const std::size_t local_rows = row_end - row_start;

    CSRMatrix local;
    local.rows = local_rows;
    local.cols = global.cols;  // Preserve global column dimension

    // Extract CSR row pointers and rebase to start at 0
    const std::size_t nnz_start = global.row_ptr[row_start];
    const std::size_t nnz_end = global.row_ptr[row_end];
    const std::size_t local_nnz = nnz_end - nnz_start;

    local.row_ptr.resize(local_rows + 1);
    for (std::size_t i = 0; i <= local_rows; ++i) {
        local.row_ptr[i] = global.row_ptr[row_start + i] - nnz_start;
    }

    // Extract column indices and values (global column indices preserved)
    local.col_idx.assign(
        global.col_idx.begin() + nnz_start,
        global.col_idx.begin() + nnz_end
    );

    local.values.assign(
        global.values.begin() + nnz_start,
        global.values.begin() + nnz_end
    );

    return local;
}

} // namespace

MPIPartitionedMatrix partition_matrix(
    const CSRMatrix& global_matrix,
    MPI_Comm comm
) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    MPIPartitionedMatrix result;

    // Rank 0: validate and broadcast global dimensions
    std::size_t dims[2] = {0, 0};

    std::exception_ptr root_error;
    if (rank == 0) {
        try {
            global_matrix.validate();

            if (global_matrix.rows == 0) {
                throw std::invalid_argument("Cannot partition empty matrix");
            }

            // CG requires square matrices (SPD must be square).
            if (global_matrix.rows != global_matrix.cols) {
                throw std::invalid_argument(
                    "MPI partition requires square matrix (CG/SPD requirement)");
            }

            if (static_cast<std::size_t>(num_ranks) > global_matrix.rows) {
                throw std::invalid_argument(
                    "More MPI ranks than matrix rows: partitioning would "
                    "create empty rank assignments");
            }

            dims[0] = global_matrix.rows;
            dims[1] = global_matrix.cols;
        } catch (...) {
            root_error = std::current_exception();
        }
    }

    int root_failed = root_error ? 1 : 0;
    MPI_Bcast(&root_failed, 1, MPI_INT, 0, comm);
    if (root_failed) {
        if (root_error) std::rethrow_exception(root_error);
        throw std::invalid_argument("Matrix validation failed on rank 0");
    }

    // Broadcast dimensions to all ranks
    // Use MPI_UNSIGNED_LONG for std::size_t on most 64-bit platforms
    // Cast explicitly to ensure compatibility
    MPI_Bcast(dims, 2, MPI_UNSIGNED_LONG, 0, comm);

    result.global_rows = dims[0];
    result.global_cols = dims[1];

    // Compute partition boundaries (same on all ranks)
    result.partition_offsets =
        compute_partition_offsets(result.global_rows, num_ranks);

    result.row_start = result.partition_offsets[rank];
    result.row_end = result.partition_offsets[rank + 1];

    const std::size_t local_rows = result.local_rows();

    // Rank 0: send CSR blocks to other ranks
    if (rank == 0) {
        // Extract own block
        result.local_matrix = extract_csr_rows(
            global_matrix,
            result.row_start,
            result.row_end
        );

        // Send blocks to other ranks
        for (int dest = 1; dest < num_ranks; ++dest) {
            const std::size_t dest_start = result.partition_offsets[dest];
            const std::size_t dest_end = result.partition_offsets[dest + 1];

            CSRMatrix dest_block = extract_csr_rows(
                global_matrix,
                dest_start,
                dest_end
            );

            // Send metadata
            std::size_t metadata[3] = {
                dest_block.rows,
                dest_block.cols,
                dest_block.nnz()
            };
            MPI_Send(metadata, 3, MPI_UNSIGNED_LONG, dest, 0, comm);

            // Send CSR arrays
            MPI_Send(dest_block.row_ptr.data(), dest_block.rows + 1,
                     MPI_UNSIGNED_LONG, dest, 1, comm);

            if (dest_block.nnz() > 0) {
                MPI_Send(dest_block.col_idx.data(), dest_block.nnz(),
                         MPI_UNSIGNED_LONG, dest, 2, comm);
                MPI_Send(dest_block.values.data(), dest_block.nnz(),
                         MPI_DOUBLE, dest, 3, comm);
            }
        }
    } else {
        // Other ranks: receive CSR block from rank 0
        std::size_t metadata[3];
        MPI_Recv(metadata, 3, MPI_UNSIGNED_LONG, 0, 0, comm,
                 MPI_STATUS_IGNORE);

        result.local_matrix.rows = metadata[0];
        result.local_matrix.cols = metadata[1];
        const std::size_t nnz = metadata[2];

        result.local_matrix.row_ptr.resize(result.local_matrix.rows + 1);
        result.local_matrix.col_idx.resize(nnz);
        result.local_matrix.values.resize(nnz);

        MPI_Recv(result.local_matrix.row_ptr.data(),
                 result.local_matrix.rows + 1,
                 MPI_UNSIGNED_LONG, 0, 1, comm, MPI_STATUS_IGNORE);

        if (nnz > 0) {
            MPI_Recv(result.local_matrix.col_idx.data(), nnz,
                     MPI_UNSIGNED_LONG, 0, 2, comm, MPI_STATUS_IGNORE);
            MPI_Recv(result.local_matrix.values.data(), nnz,
                     MPI_DOUBLE, 0, 3, comm, MPI_STATUS_IGNORE);
        }
    }

    // Validate local CSR structure collectively before returning to callers.
    std::exception_ptr local_error;
    try {
        result.local_matrix.validate_structure();
    } catch (...) {
        local_error = std::current_exception();
    }
    agree_validation(local_error, comm);

    return result;
}

void validate_partition(
    const MPIPartitionedMatrix& part,
    MPI_Comm comm
) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    std::exception_ptr local_error;
    try {
        part.local_matrix.validate_structure();
        if (part.partition_offsets.size() !=
            static_cast<std::size_t>(num_ranks + 1)) {
            throw std::invalid_argument("Partition offsets size mismatch");
        }
        if (part.row_start > part.row_end ||
            part.row_start != part.partition_offsets[rank] ||
            part.row_end != part.partition_offsets[rank + 1] ||
            part.local_matrix.rows != part.local_rows()) {
            throw std::invalid_argument(
                "Local row range doesn't match partition offsets");
        }
        if (part.partition_offsets.front() != 0 ||
            part.partition_offsets.back() != part.global_rows) {
            throw std::invalid_argument("Partition offsets don't cover global rows");
        }
        for (std::size_t i = 0; i < part.partition_offsets.size() - 1; ++i) {
            if (part.partition_offsets[i] > part.partition_offsets[i + 1]) {
                throw std::invalid_argument("Partition offsets are not nondecreasing");
            }
        }
    } catch (...) {
        local_error = std::current_exception();
    }
    agree_validation(local_error, comm);

    // Sum local row counts and verify global total
    std::size_t local_rows = part.local_rows();
    std::size_t total_rows = 0;

    MPI_Allreduce(&local_rows, &total_rows, 1, MPI_UNSIGNED_LONG,
                  MPI_SUM, comm);

    // Do not throw on a rank-specific global_rows value before the
    // remaining agreement collectives have completed.

    // Verify global dimensions are consistent across ranks
    std::size_t global_dims[2] = {part.global_rows, part.global_cols};
    std::size_t max_dims[2] = {0, 0};
    std::size_t min_dims[2] = {0, 0};

    MPI_Allreduce(global_dims, max_dims, 2, MPI_UNSIGNED_LONG,
                  MPI_MAX, comm);
    MPI_Allreduce(global_dims, min_dims, 2, MPI_UNSIGNED_LONG,
                  MPI_MIN, comm);

    std::exception_ptr consistency_error;
    if (max_dims[0] != min_dims[0] || max_dims[1] != min_dims[1]) {
        consistency_error = std::make_exception_ptr(std::runtime_error(
            "Global dimensions inconsistent across ranks"));
    } else if (total_rows != part.global_rows) {
        consistency_error = std::make_exception_ptr(std::runtime_error(
            "Sum of local rows doesn't match global row count"));
    }

    // Compare the complete layout in bounded chunks without allocating
    // validation buffers or issuing one collective per rank boundary.
    bool offsets_match = true;
    for (std::size_t first = 0; first < part.partition_offsets.size(); first += 64) {
        std::size_t reference[64];
        const int count = static_cast<int>(std::min<std::size_t>(
            64, part.partition_offsets.size() - first));
        if (rank == 0) {
            std::copy_n(part.partition_offsets.data() + first, count, reference);
        }
        MPI_Bcast(reference, count, MPI_UNSIGNED_LONG, 0, comm);
        for (int i = 0; i < count; ++i) {
            offsets_match = offsets_match &&
                reference[i] == part.partition_offsets[first + i];
        }
    }
    if (!offsets_match && !consistency_error) {
        consistency_error = std::make_exception_ptr(std::runtime_error(
            "Partition offsets inconsistent across ranks"));
    }
    agree_validation(consistency_error, comm);
}

std::vector<double> gather_vector(
    const std::vector<double>& local_vec,
    const MPIPartitionedMatrix& part,
    MPI_Comm comm
) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    validate_partition(part, comm);
    std::exception_ptr local_error;
    try {
        if (local_vec.size() != part.local_rows()) {
            throw std::invalid_argument("Local vector size doesn't match partition");
        }
        if (part.global_rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("MPI gather: vector exceeds int count limit");
        }
    } catch (...) {
        local_error = std::current_exception();
    }
    agree_validation(local_error, comm);

    // Compute receive counts and displacements for rank 0
    std::vector<int> recvcounts;
    std::vector<int> displs;

    if (rank == 0) {
        recvcounts.resize(num_ranks);
        displs.resize(num_ranks);

        for (int r = 0; r < num_ranks; ++r) {
            const std::size_t r_rows =
                part.partition_offsets[r + 1] - part.partition_offsets[r];
            recvcounts[r] = static_cast<int>(r_rows);
            displs[r] = static_cast<int>(part.partition_offsets[r]);
        }
    }

    std::vector<double> global_vec;
    if (rank == 0) {
        global_vec.resize(part.global_rows);
    }

    MPI_Gatherv(
        local_vec.data(), static_cast<int>(local_vec.size()), MPI_DOUBLE,
        global_vec.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
        0, comm
    );

    return global_vec;
}

std::vector<double> scatter_vector(
    const std::vector<double>& global_vec,
    const MPIPartitionedMatrix& part,
    MPI_Comm comm
) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    validate_partition(part, comm);
    std::exception_ptr local_error;
    try {
        if (rank == 0 && global_vec.size() != part.global_rows) {
            throw std::invalid_argument("Global vector size doesn't match partition");
        }
        if (part.global_rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("MPI scatter: vector exceeds int count limit");
        }
    } catch (...) {
        local_error = std::current_exception();
    }
    agree_validation(local_error, comm);

    // Compute send counts and displacements on rank 0
    std::vector<int> sendcounts;
    std::vector<int> displs;

    if (rank == 0) {
        sendcounts.resize(num_ranks);
        displs.resize(num_ranks);

        for (int r = 0; r < num_ranks; ++r) {
            const std::size_t r_rows =
                part.partition_offsets[r + 1] - part.partition_offsets[r];
            sendcounts[r] = static_cast<int>(r_rows);
            displs[r] = static_cast<int>(part.partition_offsets[r]);
        }
    }

    std::vector<double> local_vec(part.local_rows());

    MPI_Scatterv(
        global_vec.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_vec.data(), static_cast<int>(local_vec.size()), MPI_DOUBLE,
        0, comm
    );

    return local_vec;
}

} // namespace cg
