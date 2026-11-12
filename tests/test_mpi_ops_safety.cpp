#include "mpi_ops.hpp"
#include "mpi_partition.hpp"

#include <mpi.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

namespace {

void check(bool condition, const char* message, MPI_Comm comm) {
    int failed = condition ? 0 : 1;
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
    if (any_failed) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) std::cerr << message << '\n';
        MPI_Abort(comm, EXIT_FAILURE);
    }
}

template <typename Function>
void expect_collective_exception(Function function, MPI_Comm comm) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception&) {
        caught = true;
    }
    check(caught, "Every rank must receive the validation exception", comm);
}

cg::CSRMatrix identity(std::size_t n) {
    cg::CSRMatrix matrix;
    matrix.rows = matrix.cols = n;
    matrix.row_ptr.resize(n + 1);
    for (std::size_t i = 0; i < n; ++i) {
        matrix.row_ptr[i] = i;
        matrix.col_idx.push_back(i);
        matrix.values.push_back(1.0);
    }
    matrix.row_ptr[n] = n;
    return matrix;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    cg::CSRMatrix global;
    if (rank == 0) global = identity(static_cast<std::size_t>(ranks + 1));
    expect_collective_exception([&] {
        cg::CSRMatrix bad = global;
        if (rank == 0) bad.row_ptr.clear();
        cg::partition_matrix(bad, comm);
    }, comm);
    expect_collective_exception([&] {
        cg::CSRMatrix empty;
        if (rank == 0) empty = identity(0);
        cg::partition_matrix(empty, comm);
    }, comm);
    if (ranks > 1) {
        expect_collective_exception([&] {
            cg::CSRMatrix too_small;
            if (rank == 0) too_small = identity(static_cast<std::size_t>(ranks - 1));
            cg::partition_matrix(too_small, comm);
        }, comm);
    }

    const auto part = cg::partition_matrix(global, comm);
    const std::vector<double> local(part.local_rows(), 1.0);
    expect_collective_exception([&] {
        auto bad = local;
        if (rank == ranks - 1) bad.push_back(0.0);
        cg::gather_vector(bad, part, comm);
    }, comm);
    expect_collective_exception([&] {
        std::vector<double> bad;
        cg::scatter_vector(bad, part, comm);
    }, comm);
    expect_collective_exception([&] {
        auto bad = part;
        if (rank == ranks - 1) bad.partition_offsets.clear();
        cg::validate_partition(bad, comm);
    }, comm);
    expect_collective_exception([&] {
        auto bad = local;
        if (rank == ranks - 1) bad.push_back(0.0);
        cg::mpi_dot(local, bad, comm);
    }, comm);
    expect_collective_exception([&] {
        auto bad = local;
        if (rank == ranks - 1) bad.push_back(0.0);
        cg::mpi_csr_matvec(part, bad, comm);
    }, comm);
    expect_collective_exception([&] {
        auto bad = part;
        if (rank == ranks - 1) bad.local_matrix.row_ptr.clear();
        cg::mpi_csr_matvec(bad, local, comm);
    }, comm);
    expect_collective_exception([&] {
        auto bad = local;
        if (rank == ranks - 1) bad.push_back(0.0);
        cg::mpi_true_residual(part, bad, local, comm);
    }, comm);

    if (ranks == 4) {
        cg::CSRMatrix five;
        if (rank == 0) five = identity(5);
        const auto empty_tail = cg::partition_matrix(five, comm);
        check(empty_tail.partition_offsets == std::vector<std::size_t>({0, 2, 4, 5, 5}),
              "Ceil partition boundaries changed", comm);
        check(rank != 3 || (empty_tail.local_rows() == 0 &&
              empty_tail.local_matrix.row_ptr == std::vector<std::size_t>({0})),
              "Empty trailing rank must retain a valid empty CSR", comm);
        cg::validate_partition(empty_tail, comm);
        std::vector<double> full;
        if (rank == 0) full.assign(5, 2.0);
        const auto x = cg::scatter_vector(full, empty_tail, comm);
        const auto y = cg::mpi_csr_matvec(empty_tail, x, comm);
        check(y == x, "Empty-tail SpMV failed", comm);
        const auto gathered = cg::gather_vector(y, empty_tail, comm);
        check(rank != 0 || gathered == full, "Empty-tail gather failed", comm);
    }

    if (rank == 0) std::cout << "MPI collective validation tests passed\n";
    MPI_Finalize();
    return EXIT_SUCCESS;
}
