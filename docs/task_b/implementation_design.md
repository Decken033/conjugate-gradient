# Person B — MPI CG Implementation Design

## 1. Overview

This document describes the distributed-memory MPI Conjugate Gradient implementation added for DSA5208 Person B.

The implementation extends Person A's serial and OpenMP solvers with a distributed-memory CG solver using row-wise matrix partitioning and MPI collective operations.

## 2. Matrix Partitioning

### Row-wise 1D partitioning

The global matrix is divided into contiguous row blocks. For N rows and P ranks:

```
rows_per_rank = ceil(N / P)
rank r owns rows: [r * rows_per_rank, min((r+1) * rows_per_rank, N))
```

This scheme handles non-divisible N/P by assigning `ceil(N/P)` rows to all but the last rank, which receives the remainder.

Example: N=81, P=4
- Rank 0: rows 0–20 (21 rows)
- Rank 1: rows 21–41 (21 rows)
- Rank 2: rows 42–62 (21 rows)
- Rank 3: rows 63–80 (18 rows)

### Local CSR storage

Each rank stores only its assigned rows in a `CSRMatrix`:
- `row_ptr` is rebased to start at 0
- `col_idx` uses global column indices (no translation needed during SpMV)
- `rows = local_rows`, `cols = global_cols`

### Distribution protocol

1. Rank 0 generates or loads the full matrix
2. Rank 0 broadcasts global dimensions (rows, cols)
3. All ranks compute identical partition boundaries
4. Rank 0 extracts each rank's CSR block and sends via point-to-point messages
5. Other ranks receive their blocks and validate local CSR structure

Memory on rank 0 during distribution: O(N) for the full matrix plus O(N/P) temporary block.

## 3. Distributed SpMV

### Communication pattern

To compute `y = A * x` where A is row-partitioned:
- Each rank computes `y_local = A_local * x_global`
- Requires access to the full x vector

**MPI_Allgatherv** replicates x on all ranks before the local SpMV computation.

```
MPI_Allgatherv(x_local, ..., x_global, recvcounts, displs, ...)
```

Each rank then computes its local rows using the global x_global and the local CSR matrix (with global column indices).

### Communication cost analysis

- Allgatherv: each rank reconstructs an O(N) vector per call; aggregate network traffic depends on the MPI collective algorithm and topology
- Memory: O(N) per rank for x_global
- Computation: O(NNZ_local) per rank

This is correct for arbitrary sparsity patterns. For structured matrices like 2D Poisson, a sparse halo exchange would reduce communication to O(boundary_width * P), but is not implemented here.

### Limitation

MPI_Allgatherv uses `int` for counts/displacements. The maximum safe N is ~2^31. For larger matrices, MPI 4.0 large-count routines would be required.

## 4. Distributed Dot Products

```
mpi_dot(a_local, b_local) = MPI_Allreduce(local_sum, MPI_SUM)
```

All ranks contribute their local partial sums. MPI_Allreduce with MPI_SUM gives the global dot product to all ranks.

**Numerical note**: Floating-point reduction order in MPI_Allreduce differs from serial accumulation. This may cause small differences in CG convergence history compared to serial CG. Both are numerically valid; iteration count differences of ±5 or more are acceptable as long as the true relative residual criterion is met.

## 5. Distributed True Residual

True residual `r_true = b - A*x` is computed in distributed manner:

```
r_true_local = b_local - mpi_csr_matvec(A, x_local)
||r_true||_2 = sqrt(mpi_dot(r_true_local, r_true_local))
```

No gather to rank 0 is required. This is a key correctness requirement for scaling beyond rank 0.

## 6. Collective Safety

All ranks must call MPI collective operations together. The implementation ensures:

1. **All branches are collective**: Convergence, breakdown, and restart decisions are computed using global values from MPI_Allreduce
2. **Error broadcast**: `collective_check()` calls `MPI_Allreduce(MPI_MAX)` on an error flag so all ranks throw or none throw
3. **No rank-specific exits**: No rank exits the iteration loop independently; all ranks check the same globally computed convergence condition
4. **Restart path**: If recursive residual drifts from true residual, all ranks restart from the true residual (identical path)

## 7. Interface Modularity for Person C (Hybrid)

The MPI infrastructure is separated into three layers:

```
cg_mpi_ops:  mpi_partition.cpp + mpi_ops.cpp
             (partitioning, SpMV, dot, vector ops)
             
cg_mpi:      mpi_solver.cpp
             (CG iteration loop)
             
cg_mpi_benchmark: benchmark_mpi.cpp + benchmark_mpi_csv.cpp
                  (timing, CSV output)
```

Person C can:
1. Reuse `MPIPartitionedMatrix` and `partition_matrix()`
2. Reuse `mpi_dot()` and `mpi_norm2()`
3. Replace `mpi_csr_matvec()` with a hybrid version using OpenMP-parallelized local SpMV
4. Replace local vector updates with OpenMP-parallelized versions

The `solve_mpi_cg()` signature can be reused with modified `mpi_ops` functions.

## 8. CMake Integration

MPI is optional and does not affect Serial or OpenMP builds:

```cmake
option(CG_ENABLE_MPI "Build the distributed-memory MPI CG solver" ON)

if(CG_ENABLE_MPI)
    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        # Add cg_mpi_ops, cg_mpi, cg_mpi_benchmark, run_mpi_benchmark
        # Add CTest MPI test targets
    endif()
endif()
```

When MPI is not found, a status message is printed but no error is raised.

## 9. Known Limitations

1. **Allgatherv memory**: O(N) per rank. For large N (e.g., thermal2 N=1.2M), each rank allocates ~10 MB for x_global.
2. **Int displacement overflow**: N > ~2^31 would require MPI 4.0 large-count API
3. **No process placement control**: Binding is specified in PBS scripts using `--bind-to core`
4. **Single communicator**: Uses MPI_COMM_WORLD throughout; refactoring for sub-communicators is possible but not implemented
5. **Load imbalance**: For matrices with non-uniform row NNZ, some ranks do more work. Row-wise partitioning does not account for NNZ distribution.
