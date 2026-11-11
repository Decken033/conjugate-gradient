# Conjugate Gradient (CG): Mathematical Principle and Integration with the Hybrid Project

## 1. Problem statement

The Conjugate Gradient method solves a linear system

\[
Ax=b,
\]

where the standard CG method assumes that \(A\) is symmetric positive definite (SPD):

\[
A=A^T, \qquad z^T A z>0 \quad \text{for every nonzero }z.
\]

The method is especially useful for large sparse systems because it uses matrix-vector products and vector operations rather than a direct matrix factorization.

## 2. Quadratic-form interpretation

For an SPD matrix, define the quadratic objective

\[
\phi(x)=\frac12 x^T A x-b^T x.
\]

Its gradient is

\[
\nabla\phi(x)=Ax-b.
\]

The minimizer satisfies

\[
\nabla\phi(x_*)=0 \iff Ax_*=b.
\]

Thus solving the SPD linear system is equivalent to minimizing this strictly convex quadratic function. Positive definiteness makes the surface bowl-shaped and gives it a unique minimizer.

## 3. Residual and search directions

For a current estimate \(x_k\), define the residual

\[
r_k=b-Ax_k.
\]

Since \(r_k=-\nabla\phi(x_k)\), the residual points in the direction of steepest local decrease. CG does not simply follow the negative gradient repeatedly. It constructs search directions that are mutually conjugate with respect to \(A\):

\[
p_i^T A p_j=0 \quad (i\ne j).
\]

A-conjugacy is different from ordinary Euclidean orthogonality. It makes progress along a previous search direction independent of a later move along another A-conjugate direction, which is the geometric reason CG can build on earlier progress efficiently.

## 4. Standard CG iteration

Given an initial guess \(x_0\), compute

\[
r_0=b-Ax_0,\qquad p_0=r_0.
\]

For iteration \(k=0,1,\ldots\), compute:

**Step length**

\[
\alpha_k=\frac{r_k^T r_k}{p_k^T A p_k}.
\]

This is the exact one-dimensional minimizer of the quadratic objective along the current search direction.

**Solution and residual updates**

\[
x_{k+1}=x_k+\alpha_k p_k,
\]

\[
r_{k+1}=r_k-\alpha_k A p_k.
\]

**Residual norm and convergence check**

\[
\rho_{k+1}=\frac{\|r_{k+1}\|_2}{\|b\|_2}.
\]

Stop when \(\rho_{k+1}\leq\text{tolerance}\), subject to the solver's true-residual verification policy.

**Direction update**

\[
\beta_k=\frac{r_{k+1}^T r_{k+1}}{r_k^T r_k},
\qquad
p_{k+1}=r_{k+1}+\beta_k p_k.
\]

The scalar \(\beta_k\) makes the new direction A-conjugate to previous directions under the standard CG recurrence.

## 5. How the algorithm maps to this project

The project separates mathematical orchestration from numerical kernels:

- `src/hybrid/hybrid_solver.cpp`: owns the outer CG iteration, computes global scalar coefficients such as alpha and beta, checks convergence, and controls restart/true-residual verification.
- `src/hybrid/hybrid_ops.cpp`: implements the local/distributed operations used by the solver.
- `include/hybrid/hybrid_solver.hpp` and `hybrid_ops.hpp`: declare the solver and operation interfaces.
- `src/hybrid/benchmark_hybrid.cpp`: performs warm-ups and repeated timed solver runs.
- `src/hybrid/benchmark_hybrid_csv.cpp`: writes per-run metrics to CSV.
- `src/hybrid/run_hybrid_benchmark.cpp`: parses CLI arguments, creates/loads the matrix, partitions it, constructs the right-hand side, configures MPI/OpenMP, calls the benchmark, and writes results.

### 5.1 Distributed data ownership

The global CSR matrix is partitioned by rows across MPI ranks. Rank \(q\) owns its local rows and corresponding local segments of vectors such as \(b\), \(x\), \(r\), and \(p\). OpenMP threads within a rank share that rank's local matrix and vectors.

### 5.2 Hybrid SpMV

The required operation is

\[
(Ap)_{local}=A_{local}p_{global}.
\]

Each rank first calls `MPI_Allgatherv` to assemble the global input vector from all local vector segments. It then uses an OpenMP parallel loop over local CSR rows to compute its local output segment. The communication is outside the OpenMP parallel region.

### 5.3 Hybrid dot product and norm

For a distributed dot product,

\[
a^Tb=\sum_q\sum_{i\in q}a_i b_i,
\]

each rank first performs an OpenMP reduction over its local elements. The resulting rank-local scalar is combined across ranks using `MPI_Allreduce(MPI_SUM)`. The norm is computed as \(\|v\|_2=\sqrt{v^Tv}\).

### 5.4 Vector updates

Updates of \(x\), \(r\), and \(p\), and the true-residual subtraction \(r_{true}=b-Ax\), are parallelized over local vector elements with OpenMP. Each thread writes distinct indices.

### 5.5 Synchronization model

The outer CG iteration remains sequential because iteration \(k+1\) depends on global scalar values produced during iteration \(k\). MPI collectives are called outside OpenMP parallel regions. The benchmark runner requests `MPI_THREAD_FUNNELED` and checks that the MPI implementation provides at least that thread-support level.

## 6. Current implementation boundaries

The current Hybrid version reuses the existing MPI row partitioner and full-vector `MPI_Allgatherv` approach. It does not yet implement halo-only exchange, nonblocking communication overlap, persistent OpenMP parallel regions, or fused vector kernels. These are possible future optimization directions, not features claimed as implemented in this snapshot.

## 7. Source basis and terminology

The mathematical explanation follows the supplied `CG总结.docx`: quadratic forms, SPD matrices, minimization of the quadratic objective, residuals as negative gradients, A-conjugate directions, and the standard iterative solution process. The implementation mapping is based on the current Hybrid source files and project implementation guide.
