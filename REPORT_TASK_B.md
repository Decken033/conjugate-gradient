> Historical context: source paths and unified-build commands describe the original revisions. For current independent builds and experiments, use the [standalone submission README](README.md). Removed paths and archived snapshots are recorded in docs/archive/cleanup_manifest.json (repository-relative). Historical results are not verification of the standalone packages.

# DSA5208 — Person B Technical Report: Distributed-Memory MPI Conjugate Gradient

- **Scope:** MPI CG implementation, correctness verification, strong scaling, and weak scaling
- **Evidence:** Earlier Poisson and correctness jobs; thermal2 job `1495684`; recovered 3D output for job `1502451`; strong and weak outputs for jobs `1509066` and `1509067`
- **Status:** Implementation, Vanda tests, earlier scaling campaigns, and the 256-rank extension audited

## Abstract

Person B implemented a C++17, double-precision, unpreconditioned distributed-memory Conjugate Gradient (CG) solver on row-partitioned CSR matrices. Distributed sparse matrix-vector multiplication (SpMV) uses `MPI_Allgatherv`, dot products use `MPI_Allreduce`, and the solver checks the distributed true residual. The latest Vanda correctness job passed 10/10 CTest targets. An audit of 60 benchmark CSV files found all 600 timed runs converged with true relative residual at most `1e-8` and relative solution error below `1e-4`: 450 earlier records and 150 new 256-rank strong and weak scaling records. In the new fixed 1024×1024 strong job, eight ranks give the fastest mean solver time, 7.488011 s (3.8536× its one-rank baseline); the curve falls back at higher ranks. In the new weak job, 16,384 unknowns per rank are maintained, but CG iterations increase from 231 to 3,439 and full-solve weak efficiency at 256 ranks is 0.000127. The PBS outputs confirm completed runs with four warmups and ten timed solves per configuration. Ratios use each job's own one-rank MPI mean. Earlier jobs cannot be used for controlled absolute-time comparisons, and no per-collective timings establish a communication fraction or cause of the scaling decline.

## 1. Introduction and Scope

CG addresses sparse symmetric positive definite systems through SpMV, vector updates and inner products [1], [2]. Task B covers CSR row partitioning and vector scatter/gather; distributed SpMV, dot product, and local vector updates; unpreconditioned MPI CG with true-residual verification; MPI CMake targets, correctness tests, CSV benchmarking, and Vanda PBS jobs. The shared Poisson generators already exist in common project code and are reused here; this report does not claim their authorship or Person A's Serial/OpenMP results. The modular `cg_mpi_ops` library can be reused by a hybrid solver, but no hybrid implementation is claimed as a Task B deliverable.

Key files are `include/mpi_partition.hpp`, `include/mpi_ops.hpp`, `include/mpi_solver.hpp`, `include/mpi_benchmark.hpp`, `src/mpi/*.cpp`, `tests/test_mpi_{partition,spmv,cg}.cpp`, MPI sections of `CMakeLists.txt`, and the PBS scripts in `scripts/vanda/` for tests and scaling campaigns.

## 2. Conjugate Gradient and MPI Implementation

### 2.1 CSR row partitioning

CSR stores row offsets, column indices and values for nonzeros [2]. For N matrix rows and P ranks, `rows_per_rank = ceil(N/P)` and rank r owns `[min(r × rows_per_rank, N), min((r+1) × rows_per_rank, N))`. Rank 0 validates the square input, broadcasts dimensions with `MPI_Bcast`, extracts CSR row blocks, and distributes them with `MPI_Send`/`MPI_Recv`. Local `row_ptr` is rebased to zero; `col_idx` keeps global indices. The partition stores global dimensions and offsets. The vector utilities use `MPI_Scatterv` and `MPI_Gatherv` in tests; the benchmark instead constructs its RHS locally through distributed SpMV and does not gather the final solution. P > N is rejected. Distribution is outside the timed solve, but the runner retains rank 0's full input matrix throughout benchmarking, in addition to its local block.

### 2.2 Distributed operations

`mpi_csr_matvec` computes Allgatherv counts and displacements from the row offsets, replicates x on every rank, and evaluates local CSR rows. `MPI_Allgatherv` supplies each rank's varying-length segment to every rank, while `MPI_Allreduce` combines scalar contributions globally [3]. The replicated vector consumes O(N) doubles per rank; vector communication is O(N) per rank per SpMV as an order-of-magnitude description, with actual network traffic dependent on the collective. `mpi_dot` sums local products and calls `MPI_Allreduce(MPI_SUM)`; `mpi_norm2` uses this dot product. Local vector updates complete the recurrence. Allgatherv uses `int` count/displacement arguments, limiting very large cases without additional handling. Sparse halo exchange, communication overlap, and OpenMP local parallelism are not implemented in this MPI variant.

The persistent solution, RHS, residual, and direction vectors contain only local row entries. A full input vector is allocated temporarily inside each SpMV. Local arithmetic per ordinary iteration costs O(NNZ_r + N_r), where NNZ_r and N_r are the local nonzeros and rows; communication time is additional. Local CSR and solver storage cost O(NNZ_r + N_r + N + P), with the additional full input CSR on rank 0. Blocking collectives require participation and data dependencies across ranks; they are not explicit barriers. The benchmark's pre-timing `MPI_Barrier` and post-timing `MPI_Reduce(MPI_MAX)` are outside the solver timer.

### 2.3 CG solver

Standard unpreconditioned CG solves $Ax=b$ for a real symmetric positive definite (SPD) matrix: $A=A^T$ and $v^TAv>0$ for every nonzero $v$ [1], [2]. The generated Dirichlet Poisson stencils satisfy this requirement. CSR validation checks square shape and storage, not symmetry or positive definiteness of arbitrary imported inputs. The symmetric thermal2 file reconstructs a symmetric matrix, but neither its SPD property nor its condition number was independently established in this audit; successful runs for the manufactured RHS do not prove SPD.

The implementation uses $x_0=0$, $r_0=b-Ax_0=b$, and $p_0=r_0$. With $\rho_k=r_k^Tr_k$, ordinary iterations use

$$
q_k=Ap_k,\qquad \alpha_k=\frac{\rho_k}{p_k^Tq_k},\qquad
x_{k+1}=x_k+\alpha_kp_k,\qquad r_{k+1}=r_k-\alpha_kq_k,
$$

$$
\rho_{k+1}=r_{k+1}^Tr_{k+1},\qquad
\beta_k=\frac{\rho_{k+1}}{\rho_k},\qquad
p_{k+1}=r_{k+1}+\beta_kp_k.
$$

Globally reduced scalars make convergence and breakdown decisions common to all ranks; the solver does not broadcast separate decision flags. Once $\sqrt{\rho_{k+1}}/\lVert b\rVert_2$ reaches the tolerance, it forms the true residual $b-Ax_{k+1}$ with an additional distributed SpMV and accepts convergence only if $\lVert b-Ax_{k+1}\rVert_2/\lVert b\rVert_2\leq 10^{-8}$. Otherwise it replaces the residual and restarts the direction from that true residual, an implementation safeguard beyond the ordinary recurrence. A zero RHS returns the exact zero solution with reported residual zero. Nonpositive `pᵀAp` and several nonfinite scalar quantities trigger breakdown checks; a true residual is also reported at the iteration limit. Benchmarks allow at most 10,000 iterations and manufacture `x* = ones`, `b = Ax*`. The solution-error metric is $\lVert x-x^*\rVert_2/\lVert x^*\rVert_2$, distinct from the stopping residual. Double precision, true-residual verification, and restarts reduce residual-drift risk but do not guarantee accuracy for all conditioning or numerical ranges.

## 3. Build and Correctness Verification

The latest correctness job (PBS `1482726.stdct-mgmt-02`, CN-152, 24 October 2026) shows successful MPI-enabled Release configuration and a full build through `run_mpi_benchmark`. **All 10 CTest targets passed**, including `mpi_partition`, `mpi_spmv`, and `mpi_cg` at four ranks. The job then ran each MPI executable with one, two, and four ranks: **all 9 executable/rank combinations passed**. The single-rank MPI-versus-serial CG comparison ran at one rank and is intentionally skipped inside two- and four-rank executions.

| Executable | Source-level coverage | Observed ranks |
|---|---|---|
| `test_mpi_partition` | Single-rank properties; 2D Poisson 16×16 and 32×32 partitions; 81-row nondivisible case; gather/scatter round trip; row integrity; partition validation | 1, 2, 4 passed |
| `test_mpi_spmv` | Distributed versus serial SpMV on 2D 16×16, 2D 32×32, 3D 8×8×8; ones and zero vectors; dot product and norm | 1, 2, 4 passed |
| `test_mpi_cg` | 2D 8×8, 16×16, 32×32; 3D 4³ and 8³; one-rank serial comparison; 16×16 and 32×32 solution checks across separately run rank counts; 81-row case | 1, 2, 4 passed |

CG checks include convergence, true relative residual at most `1e-8`, and manufactured relative solution error below `1e-4` in the 2D/3D solution tests. The SpMV serial comparisons use `|a-b| <= 1e-9*(1+|a|+|b|)`; the one-rank CG comparison uses the same mixed absolute/relative form with `1e-12`, and requires matching iteration counts. MPI tests use a custom `CHECK` that calls `MPI_Abort` on failure and remains active in Release. The routines labelled “cross-rank consistency” check each run against the known solution; they do not directly compare solution vectors from different jobs. Earlier retained jobs contain build/test and launch failures, followed by successful reruns. They are historical troubleshooting records, not the latest verification verdict.

`CMakeLists.txt` requires C++17, CMake 3.20 or later, and OpenMP even for this project's MPI build. MPI targets are conditional on `CG_ENABLE_MPI` and a found C++ MPI library; `cg_mpi_ops` links `MPI::MPI_CXX`, `cg_mpi` adds the solver, and `cg_mpi_benchmark` adds timing/CSV output. `run_mpi_benchmark` also links the shared Poisson generator and Matrix Market loader. CTest registers the three MPI tests with four processes. Its default `matrix_market` test uses small fixtures, not the optional full thermal2 integration test.

## 4. Datasets and Benchmark Matrices

The shared generator supplies five-point 2D and seven-point 3D Dirichlet Poisson matrices. The imported `Schmid/thermal2` matrix is an unstructured finite-element steady-state thermal problem in the SuiteSparse Matrix Collection [4], [5]. Its Matrix Market header declares real symmetric coordinate storage; the project loader mirrors off-diagonal entries and combines duplicates. All six workloads use the manufactured solution $x^*=\mathbf{1}$ and $b=A\mathbf{1}$, so the reported relative solution error has a known target. The supplied `thermal2_b.mtx` is not used. Convergence of this manufactured system does not independently establish that `thermal2` is positive definite.

**Table 1. Benchmark matrices and selected rank configurations.**

| Matrix | Unknowns N | NNZ | MPI rank configurations |
|---|---:|---:|---:|
| 2D Poisson 512×512 | 262,144 | 1,308,672 | 8 |
| 2D Poisson 1024×1024 | 1,048,576 | 5,238,784 | 8 |
| Schmid/thermal2 | 1,228,045 | 8,580,313 | 5 |
| 3D Poisson 40³ | 64,000 | 438,400 | 8 |
| 3D Poisson 64³ | 262,144 | 1,810,432 | 8 |
| 3D Poisson 102³ | 1,061,208 | 7,366,032 | 8 |

## 5. Experimental Setup and Methodology

### 5.1 Job provenance and timing boundary

The original 2D workloads are Poisson 512×512 (N = 262,144; NNZ = 1,308,672) and 1024×1024 (N = 1,048,576; NNZ = 5,238,784). Each uses P = 1, 2, 4, 8, 16, 32, 36, 72, four untimed warmups, and ten timed repetitions per P. The selected jobs are `1482451.stdct-mgmt-02` and `1482491.stdct-mgmt-02`, respectively. Both scripts request two nodes, 36 MPI slots per node, and `--map-by slot --bind-to core`; the 512 job requests 32 GB per node, the 1024 job 64 GB. Both benchmark logs show `CMake/3.27.6-GCCcore-13.2.0` and `OpenMPI/4.1.6-GCC-13.2.0` loaded. The 1024 log also prints GCC 13.2.0, lists nodes CN-093/CN-094, and reports Intel Xeon Platinum 8452Y on the launch host CN-093. The 512 log lists GN-A40-048/GN-A40-050 but does not print a compiler version or CPU model; GCC 13.2.0 is identified by the loaded toolchain module. Allocation and launch options are documented, but none of these logs contains a process-to-node mapping. The correctness script names a 4.1.8 MPI module, while that job's CMake output resolves a 4.1.6 MPI compiler; the benchmark logs independently establish the OpenMPI 4.1.6 environment for both scaling jobs.

The CSV field `solve_seconds` is the maximum rank elapsed time in seconds. After a barrier, `MPI_Wtime` surrounds `solve_mpi_cg`; `MPI_Reduce(MPI_MAX)` selects the slowest rank. Timed work includes local computation and solver collectives. Matrix generation, partitioning, RHS formation, warmups, post-solve solution-error calculation, and CSV writing are excluded. The CSV has no communication-only or total-job timing. `true_relative_residual` and `relative_solution_error` are dimensionless; `iterations` counts CG iterations.

The recovered output `dsa5208_mpi_scaling_3d.o1502451` identifies job `1502451.stdct-mgmt-02`, launch host `CN-033`, and a run from 14:15:56 to 14:21:48 +08 on 28 October 2026. Its allocated-node listing and generated hostfile each show 36 slots on `cn-033.stdct.loc` and 36 on `cn-040.stdct.loc`, or 72 available slots. It prints `g++ (GCC) 13.2.0`, `mpirun (Open MPI) 4.1.6`, loaded `CMake/3.27.6-GCCcore-13.2.0` and `OpenMPI/4.1.6-GCC-13.2.0` modules, MPI discovery through the corresponding `mpicxx`, and successful CMake generation and build through `run_mpi_benchmark`. The log does not print the PBS resource request, a CMake build-type flag or cache, the exact `mpirun` commands, mapping or binding flags, a per-rank placement map, CPU model, or a scheduler exit status. The job name is suggested by the output filename but is not printed as PBS metadata. A matching historical 3D script was subsequently recovered from the Vanda download backup and archived at `docs/task_b/evidence/scripts/benchmark_mpi_scaling_3d.pbs`. It requests two nodes with 36 MPI slots and 64 GB per node, Release configuration, slot mapping and core binding. These are recovered script settings, not a recorded actual process map; no submission-time script hash proves the exact submitted bytes.

The same log shows consecutive 40³, 64³ and 102³ campaigns, each at P = 1, 2, 4, 8, 16, 32, 36 and 72. Every configuration reports four warmups, ten timed runs, a matching job-suffixed CSV path, and `Converged: 10/10`. Its median, first-run iteration count and first-run residual summaries agree with the selected CSVs at printed precision; the tables below retain mean-based statistics. The final `=== 3D MPI scaling completed ===` marker and end timestamp establish normal progress through the scripted output, but no scheduler accounting record confirms an exit code. Repeated Open MPI `mca_plm_tm` and `mca_ras_tm` component warnings say that `libpbs.so.0` could not be loaded and mark the condition `(ignored)`. The module-purge warning and first-launch SSH known-host notice also appear. No fatal error or failed solve is reported; the warnings' timing impact and launcher fallback are unknown.

### 5.2 Selection and CSV-record audit

The 2D audit selected only `results/mpi/mpi_2d512_p<P>_1482451.stdct-mgmt-02.csv` and `results/mpi/mpi_2d1024_p<P>_1482491.stdct-mgmt-02.csv`. All 16 files have the expected 12 columns, exactly ten rows with run indices 0–9 once each, complete fields, finite positive solver times, consistent size/rank/NNZ metadata, `solver=mpi`, and `threads=1`. No selected run is missing or duplicated. Other validation files and earlier/partial 512 attempts in the directory are separate experiments and excluded. All selected rows have `converged=1`: **80/80 for each size, 160/160 overall**. Iterations are 894 for every 512 row and 1,755 for every 1024 row. Maximum recorded true relative residuals are about `9.5261e-09` and `9.8021e-09`, below `1e-8`; maximum recorded relative solution errors are about `2.2112e-08` and `4.8189e-08`.

The additional `thermal2` input is the SuiteSparse `Schmid/thermal2` unstructured finite-element steady-state thermal matrix, as described in its local Matrix Market header. Symmetric coordinate storage is expanded to CSR (N = 1,228,045; NNZ = 8,580,313); the loader mirrors off-diagonal entries and combines duplicates. As in Person A's experiment, the benchmark **does not use** the supplied `thermal2_b.mtx`; it constructs `x* = ones` and `b = Ax*` to make solution error measurable. The `benchmark_mpi_thermal2.pbs` script requests two nodes with 36 CPUs, 36 MPI slots, and 64 GB per node (72 CPU task slots in total), with an eight-hour walltime limit. These allocated resources and the walltime limit are distinct from the 1–16 launched processes and solver-only timings. Two allocated nodes and PBS `Exit_status: 0` are supplied experiment context; no scheduler accounting record is retained in the execution log.

The retained log `results/mpi/evidence/logs/dsa5208_mpi_thermal2.o1495684` confirms loading the CMake/GCC 13.2.0 and OpenMPI 4.1.6 modules, MPI discovery through the 4.1.6 `mpicxx` path, successful configuration/build through `run_mpi_benchmark`, and sequential configurations P = 1, 2, 4, 8, 16. Release is requested by the current script; the log does not echo the build-type flag or archive a CMake cache. Each configuration reports the matching matrix dimensions and communicator size, four warmups, ten measured runs, its corresponding CSV path, and `Converged: 10/10`. Thus `10` and `4` are repetitions and warmups, respectively, as confirmed by the runner's argument parser. All five logged medians match the CSV medians at the log's display precision. Logged iterations and residuals are from the first record of each group, not aggregate statistics; all rows were independently checked.

Before each launch, OpenMPI reports that `mca_plm_tm` and `mca_ras_tm` cannot load `libpbs.so.0`, explicitly marked `(ignored)`: ten component warnings in total. No fatal application error, abort, or nonconvergence message appears; all five configurations reach their CSV output and summaries, including 16 ranks. The warnings do not establish their performance impact or the fallback launcher. The `CN-115` prefixes identify the warning-emitting host, not a map of solver processes. No rank-to-node or binding record is present, so placement across one or both nodes is unverified.

There is a historical script discrepancy: the log loads `data/thermal2/thermal2.mtx`, whereas the current PBS script sets `MATRIX="data/suitesparse/thermal2/thermal2.mtx"`. The current script is therefore a reproduction recipe. The Vanda backup's recovered historical script, archived at `docs/task_b/evidence/scripts/benchmark_mpi_thermal2.pbs`, uses the logged `data/thermal2/thermal2.mtx` path; no submission-time hash establishes its exact submitted bytes. The recorded matrix dimensions, rank sequence, output filenames, warmups, and repeat counts agree with the CSVs and current code.

The five selected `results/mpi/thermal2/thermal2_p<P>_1495684.stdct-mgmt-02.csv` files each contain ten per-run measurements, rather than a summary row or mixed configurations. Their header is `solver,matrix_name,n,nnz,ranks,threads,run_index,iterations,converged,solve_seconds,true_relative_residual,relative_solution_error`. All 50 rows have complete fields, run indices 0–9 once per file, matching matrix metadata and rank counts, `solver=mpi`, `threads=1`, and finite positive solve times. All 50 have `converged=1` and true relative residual below `1e-8`. The CSV records the **relative** true residual and relative solution error, not an absolute final residual norm. A zero PBS exit status by itself would not establish these numerical checks.

## 6. Numerical Correctness and Validation

The separate correctness job passed 10/10 CTest targets and all nine direct MPI executable/rank combinations described in Section 3. For the performance campaigns, the audit checks saved CSV records rather than re-executing the solver: 16 selected 2D files contain 160 timed records, five `thermal2` files contain 50, 24 3D files contain 240, and 15 extension files contain 150. All 60 files have the expected schema, metadata and run indices, and all 600 records have a convergence flag, finite positive time, positive iteration count, recorded true relative residual at most $10^{-8}$, and finite relative solution error below $10^{-4}$. The benchmark implementation computed the true residual with a distributed SpMV and the solution error against $x^*=\mathbf{1}$ before writing these fields. No solution vectors were retained for independently recomputing all 600 residuals and errors from the CSV audit.

**Table 2. Recorded numerical maxima across selected timed runs; residuals and errors are relative and dimensionless.**

| Matrix | Converged/runs | Iterations | Max. true relative residual | Max. relative solution error |
|---|---:|---:|---:|---:|
| 2D Poisson 512×512 | 80/80 | 894 | 9.526076e-09 | 2.211184e-08 |
| 2D Poisson 1024×1024 | 80/80 | 1,755 | 9.802120e-09 | 4.818934e-08 |
| Schmid/thermal2 | 50/50 | 4,331–4,333 | 9.986189e-09 | 4.429738e-07 |
| 3D Poisson 40³ | 80/80 | 101 | 8.836375e-09 | 3.876745e-09 |
| 3D Poisson 64³ | 80/80 | 158 | 9.032060e-09 | 8.909344e-09 |
| 3D Poisson 102³ | 80/80 | 238 | 9.893192e-09 | 1.678557e-08 |

The 3D output log independently confirms that every one of its 24 configuration summaries reached `Converged: 10/10`; those summaries echo only one iteration/residual pair per configuration, not all individual numerical measurements. The detailed 240-record numerical checks come from the CSVs. The analogous 2D, `thermal2`, and extension record counts are also CSV checks. Thus 600/600 describes validated recorded benchmark outcomes, not independently reconstructed solutions.

## 7. Performance and Scaling

`T(P)` is the arithmetic mean of the ten timed `solve_seconds` records. SD is the sample standard deviation (`ddof = 1`, denominator 9). `S(P) = T(1)/T(P)` and percentage efficiency `E(P) = 100 × S(P)/P` use only the same matrix and timing field. The tables show efficiency as a fraction except for `thermal2`, where it is a percentage; multiplying a fractional entry by 100 gives the percentage. Ratios were computed before display rounding. Min–max is the observed timed-run range. The runner prints medians for monitoring, but no median-derived speedup is mixed into these mean-based tables or figures. Person A primarily reports median time and uses an independent serial solver as the OpenMP speedup reference; the one-rank MPI baseline here includes MPI code and collectives. Their speedup ratios do not by themselves establish absolute MPI-versus-OpenMP performance.

### 7.1 512×512 2D Poisson — job 1482451

**Table 3. 2D Poisson 512×512 solver-only scaling; ten timed runs per rank count.**

| Ranks | Converged/runs | Mean s | SD s | Min–max s | Speedup | Efficiency (fraction) | Iterations |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/10 | 1.803076 | 0.001980 | 1.801473–1.806533 | 1.0000 | 1.0000 | 894 |
| 2 | 10/10 | 1.494327 | 0.044425 | 1.464502–1.618214 | 1.2066 | 0.6033 | 894 |
| 4 | 10/10 | 0.776791 | 0.006629 | 0.770091–0.790760 | 2.3212 | 0.5803 | 894 |
| 8 | 10/10 | 0.647911 | 0.004128 | 0.643622–0.653115 | 2.7829 | 0.3479 | 894 |
| 16 | 10/10 | 0.635083 | 0.004096 | 0.631542–0.642764 | 2.8391 | 0.1774 | 894 |
| 32 | 10/10 | 0.876198 | 0.013568 | 0.850373–0.903760 | 2.0578 | 0.0643 | 894 |
| 36 | 10/10 | 0.978048 | 0.010010 | 0.965682–0.993241 | 1.8435 | 0.0512 | 894 |
| 72 | 10/10 | 3.111284 | 0.013676 | 3.088768–3.136455 | 0.5795 | 0.0080 | 894 |

### 7.2 1024×1024 2D Poisson — job 1482491

**Table 4. 2D Poisson 1024×1024 solver-only scaling; ten timed runs per rank count.**

| Ranks | Converged/runs | Mean s | SD s | Min–max s | Speedup | Efficiency (fraction) | Iterations |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/10 | 29.194377 | 0.038686 | 29.133957–29.260419 | 1.0000 | 1.0000 | 1,755 |
| 2 | 10/10 | 15.122517 | 0.014858 | 15.102769–15.140765 | 1.9305 | 0.9653 | 1,755 |
| 4 | 10/10 | 9.944964 | 0.041479 | 9.891228–10.025768 | 2.9356 | 0.7339 | 1,755 |
| 8 | 10/10 | 8.603368 | 0.147283 | 8.219117–8.681642 | 3.3934 | 0.4242 | 1,755 |
| 16 | 10/10 | 8.540302 | 0.013833 | 8.522929–8.565000 | 3.4184 | 0.2137 | 1,755 |
| 32 | 10/10 | 10.781777 | 0.013728 | 10.765208–10.810218 | 2.7078 | 0.0846 | 1,755 |
| 36 | 10/10 | 11.192095 | 0.015402 | 11.172821–11.213291 | 2.6085 | 0.0725 | 1,755 |
| 72 | 10/10 | 40.043157 | 0.169592 | 39.678259–40.236733 | 0.7291 | 0.0101 | 1,755 |

### 7.3 2D scaling figure

![MPI strong-scaling speedup and efficiency for 512×512 and 1024×1024 2D Poisson matrices](docs/task_b/mpi_scaling.svg)

**Figure 1.** Speedup (solid lines, left panel) is normalized to the mean one-rank `solve_seconds` for the same matrix. Efficiency (dashed lines, right panel) is speedup divided by rank count, displayed as a percentage. Each point uses the arithmetic mean of ten converged timed runs. All three figures place the tested rank settings at equally spaced categorical positions; horizontal distances do not represent proportional changes in rank count.

### 7.4 SuiteSparse thermal2 — job 1495684

The same `T(P)`, sample SD, speedup, and efficiency definitions apply. Here `T(1)` is **MPI with one rank**, not a separately measured serial solver. The min–max and median columns describe the ten individual `solve_seconds` values in each file. Residual and solution-error columns report maxima over each configuration's ten runs; iteration counts are constant within each configuration.

**Table 5. SuiteSparse thermal2 solver-only scaling; ten timed runs per rank count.**

| Ranks | Converged/runs | Mean s | SD s | Median s | Min–max s | Speedup | Efficiency (%) | Iterations | Max. true relative residual | Max. relative solution error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/10 | 88.523455 | 0.253923 | 88.526741 | 88.175257–89.022706 | 1.0000 | 100.00% | 4,333 | 9.860597e-09 | 4.412101e-07 |
| 2 | 10/10 | 45.098825 | 0.268347 | 45.098867 | 44.764948–45.386621 | 1.9629 | 98.14% | 4,333 | 9.843217e-09 | 4.410869e-07 |
| 4 | 10/10 | 31.069230 | 0.143527 | 31.098854 | 30.771459–31.217149 | 2.8492 | 71.23% | 4,332 | 9.965738e-09 | 4.418496e-07 |
| 8 | 10/10 | **25.690135** | 0.008274 | 25.691592 | 25.678339–25.701494 | **3.4458** | 43.07% | 4,331 | 9.986189e-09 | 4.429738e-07 |
| 16 | 10/10 | 28.187397 | 0.014522 | 28.182656 | 28.163636–28.209606 | 3.1405 | 19.63% | 4,332 | 9.937960e-09 | 4.412162e-07 |

![Thermal2 MPI mean solver time, speedup, and parallel efficiency](docs/task_b/mpi_thermal2_scaling.svg)

**Figure 2.** Mean slowest-rank solver time with sample-SD bars, measured speedup against the MPI one-rank mean and ideal linear speedup, and parallel efficiency. Ideal time is `T(1)/P`. The bars are small relative to the time-axis range; the exact SD values are in the table. Matrix loading, distribution, RHS construction, warmups, accuracy calculation, and CSV output are excluded from `solve_seconds`.

![Within-job MPI speedup for thermal2 and two Poisson matrices](docs/task_b/mpi_thermal2_comparison.svg)

**Figure 3.** Each matrix's mean solver time is normalized to its **own MPI one-rank run**. Only common rank counts 1–16 are shown. This permits comparison of scaling ratios but does not make absolute runtimes or hardware conditions equivalent across jobs.

### 7.5 3D Poisson MPI scaling — job 1502451

The selected `results/mpi/3d/mpi_3d<size>_p<ranks>_1502451.stdct-mgmt-02.csv` series covers 40³ (N = 64,000; NNZ = 438,400), 64³ (N = 262,144; NNZ = 1,810,432), and 102³ (N = 1,061,208; NNZ = 7,366,032), each at P = 1, 2, 4, 8, 16, 32, 36, 72. All 24 expected files contain ten complete measurements with run indices 0–9 exactly once, the expected 12-column schema and matrix/rank metadata, `solver=mpi`, `threads=1`, and finite positive solve times. **All 240/240 recorded runs pass the CSV numerical checks:** `converged=1`, true relative residual at most `1e-8`, and finite relative solution error below `1e-4`; no missing, duplicate, or invalid records were found. These are MPI performance scaling runs that also record numerical correctness, not a separate CTest campaign.

The generated seven-point Dirichlet stencil uses the manufactured solution $x^*=\mathbf{1}$ and $b=A\mathbf{1}$. The runner sets a `1e-8` relative true-residual tolerance and at most 10,000 CG iterations. As in Section 5, `solve_seconds` is the slowest rank's solver-only time: a pre-solve barrier precedes `MPI_Wtime`, the full `solve_mpi_cg` call includes local work and MPI collectives, and a post-timer `MPI_Reduce(MPI_MAX)` selects the maximum. Matrix generation, partition/distribution, RHS construction, four logged warmup solves, solution-error calculation, and CSV output are excluded. Each table uses the unrounded ten-run mean for its own 3D matrix's one-rank baseline; efficiency is a fraction. The recovered output verifies the warmup count; the recovered historical script records intended launch flags, but actual rank placement and binding were not logged.

**40³, N = 64,000, NNZ = 438,400**

**Table 6. 3D Poisson 40³ solver-only scaling; ten timed runs per rank count.**

| Ranks | Converged/runs | Mean s | SD s | Min–max s | Speedup | Efficiency (fraction) | Iterations | Max. true relative residual | Max. relative solution error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/10 | 0.074592 | 0.000847 | 0.074265–0.076999 | 1.0000 | 1.0000 | 101 | 8.836375e-09 | 3.876745e-09 |
| 2 | 10/10 | 0.044283 | 0.000080 | 0.044152–0.044377 | 1.6844 | 0.8422 | 101 | 8.836375e-09 | 3.876745e-09 |
| 4 | 10/10 | 0.032087 | 0.000067 | 0.031982–0.032165 | 2.3247 | 0.5812 | 101 | 8.836375e-09 | 3.876745e-09 |
| 8 | 10/10 | 0.025895 | 0.000115 | 0.025811–0.026190 | 2.8805 | 0.3601 | 101 | 8.836375e-09 | 3.876745e-09 |
| 16 | 10/10 | 0.024449 | 0.000144 | 0.024270–0.024737 | 3.0509 | 0.1907 | 101 | 8.836375e-09 | 3.876745e-09 |
| 32 | 10/10 | 0.036038 | 0.001832 | 0.034409–0.040674 | 2.0698 | 0.0647 | 101 | 8.836375e-09 | 3.876745e-09 |
| 36 | 10/10 | 0.038537 | 0.002431 | 0.036714–0.043184 | 1.9356 | 0.0538 | 101 | 8.836375e-09 | 3.876745e-09 |
| 72 | 10/10 | 0.176804 | 0.004820 | 0.169233–0.187393 | 0.4219 | 0.0059 | 101 | 8.836375e-09 | 3.876745e-09 |

**64³, N = 262,144, NNZ = 1,810,432**

**Table 7. 3D Poisson 64³ solver-only scaling; ten timed runs per rank count.**

| Ranks | Converged/runs | Mean s | SD s | Min–max s | Speedup | Efficiency (fraction) | Iterations | Max. true relative residual | Max. relative solution error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/10 | 0.385489 | 0.055199 | 0.364546–0.540558 | 1.0000 | 1.0000 | 158 | 9.032060e-09 | 8.909344e-09 |
| 2 | 10/10 | 0.300197 | 0.036964 | 0.270893–0.349176 | 1.2841 | 0.6421 | 158 | 9.032060e-09 | 8.909344e-09 |
| 4 | 10/10 | 0.196216 | 0.001610 | 0.194733–0.200045 | 1.9646 | 0.4912 | 158 | 9.032060e-09 | 8.909344e-09 |
| 8 | 10/10 | 0.160453 | 0.004079 | 0.158862–0.172044 | 2.4025 | 0.3003 | 158 | 9.032060e-09 | 8.909344e-09 |
| 16 | 10/10 | 0.149819 | 0.003167 | 0.147595–0.158357 | 2.5730 | 0.1608 | 158 | 9.032060e-09 | 8.909344e-09 |
| 32 | 10/10 | 0.173979 | 0.006074 | 0.170850–0.189534 | 2.2157 | 0.0692 | 158 | 9.032060e-09 | 8.909344e-09 |
| 36 | 10/10 | 0.188778 | 0.007565 | 0.184869–0.209978 | 2.0420 | 0.0567 | 158 | 9.032060e-09 | 8.909344e-09 |
| 72 | 10/10 | 0.406179 | 0.025234 | 0.363897–0.431233 | 0.9491 | 0.0132 | 158 | 9.032060e-09 | 8.909344e-09 |

**102³, N = 1,061,208, NNZ = 7,366,032**

**Table 8. 3D Poisson 102³ solver-only scaling; ten timed runs per rank count.**

| Ranks | Converged/runs | Mean s | SD s | Min–max s | Speedup | Efficiency (fraction) | Iterations | Max. true relative residual | Max. relative solution error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/10 | 3.208173 | 0.012278 | 3.193351–3.233066 | 1.0000 | 1.0000 | 238 | 9.893192e-09 | 1.678557e-08 |
| 2 | 10/10 | 1.850347 | 0.003642 | 1.846212–1.856727 | 1.7338 | 0.8669 | 238 | 9.893192e-09 | 1.678557e-08 |
| 4 | 10/10 | 1.303801 | 0.007332 | 1.291214–1.313873 | 2.4606 | 0.6152 | 238 | 9.893192e-09 | 1.678557e-08 |
| 8 | 10/10 | 1.076023 | 0.007647 | 1.071063–1.087415 | 2.9815 | 0.3727 | 238 | 9.893192e-09 | 1.678557e-08 |
| 16 | 10/10 | 1.221172 | 0.007626 | 1.211960–1.235717 | 2.6271 | 0.1642 | 238 | 9.893192e-09 | 1.678557e-08 |
| 32 | 10/10 | 1.524467 | 0.009686 | 1.515728–1.540715 | 2.1045 | 0.0658 | 238 | 9.893192e-09 | 1.678557e-08 |
| 36 | 10/10 | 1.594732 | 0.010768 | 1.583152–1.612881 | 2.0117 | 0.0559 | 238 | 9.893192e-09 | 1.678557e-08 |
| 72 | 10/10 | 3.208241 | 0.065777 | 3.112128–3.316048 | 1.0000 | 0.0139 | 238 | 9.893192e-09 | 1.678557e-08 |

Across all 240 3D records, the maximum true relative residual is `9.893192e-09` and the maximum relative solution error is `1.678557e-08`. The 102³ speedup at 72 ranks rounds to 1.0000 in the table; at greater precision it is approximately 0.9999786, so it is marginally slower than one rank. Across each 3D size, iteration counts and the reported maximum error metrics are constant across ranks. The 64³ one-rank timing range includes a 0.540558 s measurement and gives a larger SD than most settings; all ten measurements are retained.

### 7.6 Extended 2D strong scaling to 256 ranks — job 1509066

This separate job held the 1024 x 1024 Poisson grid fixed (N = 1,048,576; NNZ = 5,238,784) and tested P = 1, 2, 4, 8, 16, 32, 36, 72, 144, 256. The recovered PBS script requests eight hosts with 36 slots each, a Release build, slot mapping and core binding. The retained PBS output confirms the Open MPI 4.1.6 environment, four untimed warmups, ten timed solves per configuration, and a completion marker; it does not independently echo the CMake build-type flag. All 100 CSV runs converged; every true relative residual is at most `1e-8` and every relative solution error is below `1e-4`. Values below derive from the ten slowest-rank solver times in each CSV. `S(P) = T(1)/T(P)` and `E(P) = S(P)/P`, where `T(1) = 28.855629833 s` is the **same job's** one-rank mean. SD is the sample SD. Time per iteration is `T(P)/1755`.

**Table 9. Extended strong scaling; seconds unless stated otherwise.**

| Ranks | Converged/runs | Mean s | SD s | Min-max s | Speedup | Efficiency (fraction) | Iterations | s/iteration | Max. true residual | Max. solution error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/10 | 28.855630 | 0.074730 | 28.768574-29.028995 | 1.0000 | 1.0000 | 1755 | 0.016442 | 9.802e-09 | 4.819e-08 |
| 2 | 10/10 | 14.892024 | 0.014700 | 14.864599-14.910291 | 1.9377 | 0.9688 | 1755 | 0.008485 | 9.802e-09 | 4.819e-08 |
| 4 | 10/10 | 9.635185 | 0.038098 | 9.575441-9.699832 | 2.9948 | 0.7487 | 1755 | 0.005490 | 9.802e-09 | 4.819e-08 |
| 8 | 10/10 | 7.488011 | 0.008232 | 7.477821-7.499196 | 3.8536 | 0.4817 | 1755 | 0.004267 | 9.802e-09 | 4.819e-08 |
| 16 | 10/10 | 8.311225 | 0.007128 | 8.302022-8.322686 | 3.4719 | 0.2170 | 1755 | 0.004736 | 9.802e-09 | 4.819e-08 |
| 32 | 10/10 | 10.702259 | 0.012008 | 10.689487-10.733922 | 2.6962 | 0.0843 | 1755 | 0.006098 | 9.802e-09 | 4.819e-08 |
| 36 | 10/10 | 11.272234 | 0.033736 | 11.238995-11.344888 | 2.5599 | 0.0711 | 1755 | 0.006423 | 9.802e-09 | 4.819e-08 |
| 72 | 10/10 | 26.065762 | 0.263830 | 25.627629-26.388369 | 1.1070 | 0.0154 | 1755 | 0.014852 | 9.802e-09 | 4.819e-08 |
| 144 | 10/10 | 29.930964 | 0.167740 | 29.646970-30.279604 | 0.9641 | 0.0067 | 1755 | 0.017055 | 9.802e-09 | 4.819e-08 |
| 256 | 10/10 | 24.540093 | 0.462706 | 23.677565-25.056423 | 1.1759 | 0.0046 | 1755 | 0.013983 | 9.802e-09 | 4.819e-08 |

![Extended 2D Poisson strong scaling with logarithmic rank axis](results/mpi/extension/analysis/strong_scaling.svg)

**Figure 4.** Mean solver time, speedup, parallel efficiency, and mean time per CG iteration. The horizontal axis is logarithmic in MPI ranks; the baseline is this job's one-rank mean, and each point represents ten timed runs. The fastest mean is at eight ranks. The mean rises to 8.311225 s at 16 ranks and 29.930964 s at 144 ranks, then falls to 24.540093 s at 256 ranks. Thus the curve is nonmonotonic; additional ranks beyond eight do not give sustained speedup. The iteration count remains 1,755, so the time changes occur within iterations rather than from a changing iteration count. Collective overhead and placement effects are plausible, but these CSVs contain no per-collective timing, so neither a communication fraction nor a definitive cause can be calculated. The earlier 1024 x 1024 experiment was a different job and allocation; its absolute times are not a controlled comparison with this one.

### 7.7 Weak scaling to 256 ranks — job 1509067

Grid width doubles when P quadruples, keeping `N/P = 16,384` unknowns at every configuration. The job used the same eight-node resource request and timing protocol as Section 7.6. All 50 CSV runs converged and met the same residual and solution-error checks. `E_weak(P) = T_weak(1)/T_weak(P)`, with this job's one-rank mean `T_weak(1) = 0.0369419313 s`; it is the conventional full-solve weak efficiency, not an iteration-normalized metric. `s/iteration = T_weak(P)/iterations` exposes changes in iteration cost separately from changes in count. All ratios use unrounded means.

**Table 10. Weak scaling with constant unknowns per rank; seconds unless stated otherwise.**

| Ranks | Grid | N/rank | Converged/runs | Mean s | SD s | Min-max s | Weak efficiency (fraction) | Iterations | s/iteration | Max. true residual | Max. solution error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128x128 | 16384 | 10/10 | 0.036942 | 0.000031 | 0.036910-0.036991 | 1.000000 | 231 | 0.000160 | 9.915e-09 | 1.101e-08 |
| 4 | 256x256 | 16384 | 10/10 | 0.162465 | 0.001392 | 0.160017-0.165144 | 0.227385 | 454 | 0.000358 | 9.917e-09 | 1.186e-08 |
| 16 | 512x512 | 16384 | 10/10 | 0.640462 | 0.048613 | 0.620295-0.778231 | 0.057680 | 894 | 0.000716 | 9.526e-09 | 2.211e-08 |
| 64 | 1024x1024 | 16384 | 10/10 | 19.849509 | 0.144582 | 19.620453-20.090057 | 0.001861 | 1755 | 0.011310 | 9.802e-09 | 4.819e-08 |
| 256 | 2048x2048 | 16384 | 10/10 | 291.132526 | 2.664642 | 288.323654-295.614486 | 0.000127 | 3439 | 0.084656 | 9.993e-09 | 1.068e-07 |

![2D Poisson weak scaling with logarithmic rank axis](results/mpi/extension/analysis/weak_scaling.svg)

**Figure 5.** Mean solver time, mean time per iteration, full-solve weak efficiency, and CG iteration count, with a logarithmic rank axis. Iterations increase from 231 to 3,439 as the grid grows from 128x128 to 2048x2048. Therefore fixed unknowns per rank does not fix total solver work; the rising iteration count contributes to the loss of full-solve weak efficiency. Per-iteration mean time also rises from 0.000160 to 0.084656 s. The experiment does not isolate communication from local work, process placement, or other system effects; no MPI collective breakdown was recorded.

## 8. Discussion and Limitations

### 8.1 Scaling interpretation

Both 2D sizes improve through 16 ranks in mean solver time, but scaling is not close to linear across the full range. For 512, the mean falls from 1.803076 to 0.635083 s (2.8391×; 17.74% efficiency); for 1024, from 29.194377 to 8.540302 s (3.4184×; 21.37%). The larger workload has higher within-job efficiency at every P > 1, including 96.53% at two ranks versus 60.33% for 512. These dimensionless ratios do not imply identical hardware between jobs.

Figure 1 makes the shared trend visible: both speedup curves peak at 16 ranks, then fall, while efficiency declines as ranks increase. The 1024×1024 case retains more useful scaling through 16 ranks than the 512×512 case, but neither maintains the gain at 32 or 36 ranks. At 72 ranks both speedups fall below 1, so adding ranks is slower than the corresponding one-rank run.

Mean time rises beyond 16 ranks. At 32, 36 and 72 ranks, respectively, the 512 means are 0.876198, 0.978048 and 3.111284 s; the 1024 means are 10.781777, 11.192095 and 40.043157 s. Their 72-rank speedups are 0.5795× and 0.7291× against their own one-rank baselines. The 2D Poisson scripts supply a hostfile with 36 slots on each of two nodes and request slot mapping and core binding. This specifies the intended launch policy, but the logs contain no explicit process mapping, so actual node occupancy is not asserted. An inter-node contribution is plausible under the intended policy but unverified. Constant iteration counts and successful true-residual checks rule out iteration-count changes or failed solves as explanations for these timing trends.

For `thermal2`, mean solve time drops from 88.523455 s at one rank to 25.690135 s at eight ranks, then rises to 28.187397 s at 16 ranks (9.72% above the eight-rank mean). Eight ranks gives the best observed execution time and 3.4458× speedup, far below the ideal 8×. Excluding the one-rank baseline, two ranks give the highest efficiency at 98.14%; efficiency falls to 43.07% at eight and 19.63% at 16. Every run converged below the `1e-8` true relative-residual tolerance. Iterations vary from 4,331 to 4,333 across ranks: 16 ranks requires one more iteration than eight ranks (4,332 versus 4,331, a 0.023% increase). That difference alone is far smaller than the 9.72% runtime increase under an equal-cost-per-iteration comparison; it does not fully control work because residual-check/restart counts were not logged. The relative solution errors are near `4.4e-07` for every configuration. Minor iteration and residual differences are compatible with different floating-point reduction orders; their precise cause is not measured.

At common rank counts, `thermal2` has higher speedup than both Poisson cases at eight ranks (3.4458× versus 2.7829× and 3.3934×), but at 16 ranks its 3.1405× lies between their 2.8391× and 3.4184×. These are descriptive sample-mean comparisons, not statistical significance claims. The `thermal2` curve peaks at eight ranks within its tested range, whereas both Poisson sample means continue improving to 16 before declining at higher counts. The Poisson matrices come from regular five-point stencils with roughly five nonzeros per row; `thermal2` is an unstructured finite-element matrix with about 6.99 nonzeros per row on average and requires substantially more iterations than either Poisson case. Its rowwise nonzero distribution and off-rank dependencies were not profiled in the experiment. Distinct sparsity patterns can change local SpMV work and memory access, but this implementation always gathers the full N-entry input vector regardless of sparsity. The data do not isolate which factor produced the different curve. The jobs were separate and actual rank placement is unknown, so this is a comparison of within-job ratios, not a controlled matrix-only comparison.

Within the 3D campaign, 40³ and 64³ have their lowest mean solve times at 16 ranks (0.024449 s, 3.0509×; 0.149819 s, 2.5730×), while 102³ is fastest at eight ranks (1.076023 s, 2.9815×). At 32, 36 and 72 ranks, the 40³ means rise to 0.036038, 0.038537 and 0.176804 s; the 64³ means rise to 0.173979, 0.188778 and 0.406179 s. For 102³, they are 1.524467, 1.594732 and 3.208241 s, with the last essentially equal to its 3.208173 s one-rank mean at displayed precision. The recovered log verifies that these runs took place in a two-host, 36-slot-per-host allocation, but it does not identify the hosts occupied by individual ranks. The fixed 101, 158 and 238 iterations per size and successful recorded true-residual checks show that these timing changes are not iteration-count changes or failed solves.

The 64³ matrix has exactly the same N = 262,144 as the existing 2D 512×512 case, but its seven-point stencil has 1,810,432 NNZ and 6.90625 nonzeros per row on average, versus 1,308,672 NNZ and 4.99219 per row for the 2D five-point stencil. The 102³ matrix has N = 1,061,208, about 1.20% more unknowns than 2D 1024×1024 (N = 1,048,576); their NNZ counts are 7,366,032 versus 5,238,784, or 6.94118 versus 4.99609 per row. These are geometric workload comparisons, not controlled runtime comparisons: the jobs, stencil work, iteration counts, and unknown process placement differ. Each speedup uses only its own matrix's one-rank MPI baseline. The smaller 40³ result has no comparable MPI 2D 256×256 performance baseline among the selected records.

Each added rank receives fewer CSR rows and usually fewer local nonzeros, reducing useful SpMV and vector arithmetic per rank. Each ordinary CG iteration nevertheless gathers an N-entry input vector through `MPI_Allgatherv` and performs global `MPI_Allreduce` operations for dot products and norms. As local work shrinks, these collective costs can occupy a larger share of solver time; fixed-size messages, collective synchronization, and possibly unequal row nonzeros may limit strong scaling. The workload sizes, nonzero counts, sparsity patterns and iteration counts differ, so the 2D, 3D and `thermal2` curves are descriptive rather than a controlled causal comparison. No per-collective time, per-rank work or placement map was recorded. Communication and imbalance remain plausible source-based explanations, not measured root causes.

### 8.2 Evidence and implementation limits

The original 2D evidence covers two structured Poisson matrices, one successful job per size, and ten sequential timed repetitions per rank setting. Within-job SD does not measure between-job, node, or placement variation. The 1024 eight-rank minimum, 8.219117 s, is below the other measurements and raises its SD; this valid record is retained. The 512 two-rank setting also varies more than adjacent configurations. Jobs run rank settings in ascending order, leaving time-dependent cluster effects possible. The 512 CPU model is not established in its retained log, so absolute cross-job times are not treated as a controlled hardware comparison.

The `thermal2` evidence adds one completed benchmark campaign with five sequential rank configurations and ten timed repetitions each, supported by its execution log and all five CSVs. Module loading and build completion are recorded, and `CN-115` appears in launcher warnings, but process mapping, CPU model, total job time, and scheduler exit status are not recorded there. PBS `Exit_status: 0` remains user-supplied context. The two-node resource request and maximum 16 ranks do not prove either single-node or multi-node execution. Sample SD describes within-configuration repeats only, not independent job or node variability. These results exclude matrix loading and distribution, so they characterize solver strong scaling rather than end-to-end application runtime. No saved thermal2 solution vectors or independent reference-solver run for this MPI campaign are available; validation uses the known manufactured solution and distributed true residual.

The recovered 3D output resolves the earlier uncertainty about the allocation listing, software modules and versions, build completion, warmup count, launch order and completion marker. A matching historical script recovered from the Vanda backup now records requested resources, Release configuration and intended launch flags. Submission-time script identity, actual rank placement/binding and scheduler exit status remain unverified; scheduler accounting is unavailable. The ten repeats per configuration are within one job allocation, not independent HPC allocations. The 64³ one-rank range is relatively broad, so its mean-based speedups should be read with that dispersion in view. The retained CSV metrics do not provide independently saved solution vectors or isolate communication, load balance or cross-node effects.

The solver replicates a full vector per rank, uses `int` Allgatherv counts/displacements, and has no halo exchange, overlap, or communication-only profiling. The correctness suite exercises representative paths at one, two, and four ranks; the Poisson scaling experiments additionally show convergence through 72 ranks for two 2D and three 3D matrices. The thermal2 CSVs show convergence through 16 ranks. These results do not provide formal proof for arbitrary sparse inputs.

**Post-audit fix status:** collective-safety fixes and bounded regression tests have now been prepared. They have not been compiled or run in this Windows session; Vanda acceptance is pending. The historical CSVs/logs above were produced before these changes. The following paragraph describes the audited pre-fix implementation, not a claim that the new tests have passed.

Additional source-level limits matter outside the exercised cases. `std::size_t` metadata is sent as `MPI_UNSIGNED_LONG`, consistent with the intended LP64 Linux environment but not portable to every ABI. The ceil-block rule can leave trailing empty ranks even when P <= N (for example N = 5, P = 4); the benchmark's conditional solution-error reduction is unsafe when some ranks have zero local entries. All reviewed experiments have nonempty partitions. Solver scalar checks are collective, but some partition/loader failures throw on rank 0 or an individual rank before peers reach collectives; general exception recovery is not demonstrated. The report therefore makes no claim of portable large-count support, safe arbitrary rank counts, or fully collective recovery from every input error.

## 9. Reproduction Information

From the repository root on Vanda, the recorded workflows are:

```bash
qsub scripts/vanda/test_mpi_correctness.pbs
qsub scripts/vanda/benchmark_mpi_scaling.pbs
qsub scripts/vanda/benchmark_mpi_scaling_1024.pbs
qsub scripts/vanda/benchmark_mpi_thermal2.pbs
```

The PBS jobs `1482726.stdct-mgmt-02` (correctness), `1482451.stdct-mgmt-02` (512×512), and `1482491.stdct-mgmt-02` (1024×1024) provide the build, test, and benchmark records. The 16 selected CSVs in `results/mpi/` provide the per-run measurements used in the tables. The scripts record the module, PBS, and cluster-specific path settings needed to reproduce the workflows.

The `thermal2` measurements are retained in `results/mpi/thermal2/` and the canonical execution log in `results/mpi/evidence/logs/`, with job ID `1495684.stdct-mgmt-02`. The logged command loads `data/thermal2/thermal2.mtx`, also the current local coefficient-matrix path. To reproduce with the current PBS script, stage the same dataset at its configured `data/suitesparse/thermal2/thermal2.mtx` path before submission. `thermal2_b.mtx` is not used. Within an allocated Vanda compute environment, the explicit build and a single rank-configuration command are:

```bash
module load CMake/3.27.6-GCCcore-13.2.0
module load OpenMPI/4.1.6-GCC-13.2.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCG_ENABLE_MPI=ON
cmake --build build --parallel 8 --target run_mpi_benchmark
mpirun -np 8 ./build/run_mpi_benchmark thermal2 data/thermal2/thermal2.mtx 10 results/mpi/thermal2/thermal2_p8_reproduction.csv 4
```

Use a new output filename for each reproduced configuration because the CSV writer truncates existing files. These are reproduction instructions; no new HPC job or solver benchmark was executed during this audit. The current correctness script mixes a 4.1.8 module request with a build cache that historically resolved 4.1.6; reproduce with a consistent MPI compiler, runtime, and fresh build directory. Adapt account/scratch paths in the historical Poisson and correctness scripts. Recovered script snapshots are archived under `docs/task_b/evidence/scripts/`, but submission-time script hashes, binary hashes and historical per-job CMake caches are unavailable, so exact source/build provenance is not established for every job.

The 24 additional 3D CSVs reside in `results/mpi/3d/`; their `1502451.stdct-mgmt-02` suffix matches the recovered root-level `dsa5208_mpi_scaling_3d.o1502451` output. That log records the allocation and environment summarized in Section 5. A matching historical 3D script has now been recovered and archived at `docs/task_b/evidence/scripts/benchmark_mpi_scaling_3d.pbs`. `docs/archive/vanda/benchmark_mpi_3d_reproduce.pbs` is a reproduction plan, not a historical submitted script; the original `mpirun` options must not be inferred from it. Figures 1–3 describe the earlier 2D/thermal2 experiments, Section 7.5 presents the 3D results, and Figures 4–5 cover the new jobs. Run `python scripts/analyze_mpi_scaling_extension.py results/mpi/extension/strong_1509066.stdct-mgmt-02 results/mpi/extension/weak_1509067.stdct-mgmt-02 --output-dir results/mpi/extension/analysis` to regenerate the new summaries and SVGs. Run `python scripts/validate_task_b_report.py` to check all 60 CSVs, the tables, chart coordinates, log markers, DOCX key numbers, and figure paths. Figure 1 is a retained standalone SVG; its original plotting generator is unavailable.

Canonical tracked copies of all seven correctness/campaign logs are under `results/mpi/evidence/logs/`. See `docs/task_b/evidence/README.md` for the 60-file/600-record selection and recovered-script limitations, and `results/mpi/evidence/manifest.json` for 76 imported file hashes. Run `python -B scripts/verify_mpi_evidence.py` to verify them. Hashes were computed at local import, not at historical submission. Archiving evidence does not resolve the implementation limitations in Section 8.2.

Environment statements above use the particular retained job logs, rather than treating the older `docs/ENVIRONMENT.md` as proof of every job's environment. Exact binary hashes, the historical CMake caches and all independent allocation effects remain unavailable.

## 10. Conclusion

Task B's MPI implementation, Vanda correctness verification, earlier 2D Poisson, `thermal2`, and 3D campaigns, plus the 256-rank strong and weak extension are documented by source, logs and CSVs. The latest CTest passed 10/10; all 600 timed runs in 60 selected CSVs met the recorded convergence and numerical checks, without independently reconstructing solutions. The new fixed-grid strong job is fastest at eight ranks (7.488011 s), with nonmonotonic performance loss beyond that point. In the weak job, iterations rise from 231 to 3,439 as the grid grows, alongside a substantial rise in mean time per iteration. Each scaling ratio uses its own job's one-rank baseline; separate jobs are not controlled absolute-time comparisons. The CSVs do not isolate MPI collective time, so no communication fraction or definitive performance cause is established. The 3D reproduction PBS is a proposed method, not the submitted historical script; exact historical process placement and scheduler exit status remain unavailable.

## 11. References

[1] M. R. Hestenes and E. Stiefel, “Methods of Conjugate Gradients for Solving Linear Systems,” *Journal of Research of the National Bureau of Standards*, vol. 49, no. 6, pp. 409–436, 1952.

[2] Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. Philadelphia, PA, USA: SIAM, 2003.

[3] MPI Forum, *MPI: A Message-Passing Interface Standard*, collective communication chapters, [Online]. Available: https://www.mpi-forum.org/docs/ (accessed 28 October 2026).

[4] T. A. Davis and Y. Hu, “The University of Florida Sparse Matrix Collection,” *ACM Transactions on Mathematical Software*, vol. 38, no. 1, article 1, 2011.

[5] SuiteSparse Matrix Collection, “Schmid/thermal2,” [Online]. Available: https://sparse.tamu.edu/Schmid/thermal2 (accessed 28 October 2026). Dataset identity and application description are also recorded in `data/thermal2/thermal2.mtx`.
