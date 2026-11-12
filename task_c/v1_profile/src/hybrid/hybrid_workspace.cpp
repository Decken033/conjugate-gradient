#include "hybrid/hybrid_workspace.hpp"

#include <limits>
#include <stdexcept>

namespace cg {

HybridWorkspace make_hybrid_workspace(
    const MPIPartitionedMatrix& A,
    MPI_Comm comm
) {
    int num_ranks = 0;
    MPI_Comm_size(comm, &num_ranks);

    if (A.global_rows != A.global_cols) {
        throw std::invalid_argument("Hybrid workspace: CG requires a square matrix");
    }
    if (A.partition_offsets.size() != static_cast<std::size_t>(num_ranks + 1)) {
        throw std::invalid_argument("Hybrid workspace: invalid partition offsets");
    }

    HybridWorkspace workspace;
    workspace.recvcounts.resize(static_cast<std::size_t>(num_ranks));
    workspace.displs.resize(static_cast<std::size_t>(num_ranks));

    for (int r = 0; r < num_ranks; ++r) {
        const std::size_t begin = A.partition_offsets[static_cast<std::size_t>(r)];
        const std::size_t end = A.partition_offsets[static_cast<std::size_t>(r + 1)];
        if (end < begin) {
            throw std::invalid_argument("Hybrid workspace: partition offsets are not monotonic");
        }
        const std::size_t count = end - begin;
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            begin > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "Hybrid workspace: MPI_Allgatherv count/displacement exceeds INT_MAX");
        }
        workspace.recvcounts[static_cast<std::size_t>(r)] = static_cast<int>(count);
        workspace.displs[static_cast<std::size_t>(r)] = static_cast<int>(begin);
    }

    workspace.x_global.resize(A.global_rows);
    workspace.spmv_output.resize(A.local_rows());
    return workspace;
}

} // namespace cg
