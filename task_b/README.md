# Task B: standalone historical MPI Conjugate Gradient

This package uses production sources from `56ac360` with three narrowly scoped safety patches. It builds independently of Tasks A and C, without OpenMP or GPU libraries. Historical CSVs/logs are not runtime inputs. Independent verification on NUS Vanda on 13 November 2026 confirmed successful standalone compilation, all 11 registered CTests passing, and a converged two-rank MPI benchmark smoke test. See [Vanda verification](../docs/VANDA_VERIFICATION_2026-11-13.md).

## Environment and build

Use 64-bit Linux with the LP64 ABI (size_t is unsigned long), a C++17 compiler, CMake >=3.20 and MPI. The recorded benchmark recipes load GCC 13.2 through CMake/3.27.6-GCCcore-13.2.0 and OpenMPI/4.1.6-GCC-13.2.0. Python 3 is required with BUILD_TESTING=ON. The asymmetric runner regression specifically requires OpenMPI MPMD colon syntax; an unsupported launcher fails explicitly.

From task_b, configure a fresh build using the same compiler/MPI modules used for execution:

```bash
module purge
module load CMake/3.27.6-GCCcore-13.2.0
module load OpenMPI/4.1.6-GCC-13.2.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DMPI_CXX_COMPILER=mpicxx -DBUILD_TESTING=ON
cmake --build build --parallel 4
```

MPI is required, not an optional fallback. The original MPI target definitions impose no extra production optimization flags; Release uses the compiler's CMake defaults, historically GCC -O3 -DNDEBUG. All test targets explicitly enable assertions in Release. The serial reference is built only when testing is enabled.

## Correctness and runner tests

Inside a Vanda compute allocation with at least four MPI slots, from task_b:

```bash
ctest --test-dir build --output-on-failure --no-tests=error -j 1
```

Nine original correctness registrations run partition, SpMV and CG tests at 1, 2 and 4 ranks. A new four-rank benchmark regression uses nine rows, producing an empty final rank; it checks zero-iteration nonconvergence records and a subsequent valid solve. Each has a 90-second CTest timeout.

The new Python runner test launches the actual executable, checks successful CSV output, invalid arguments, setup/output failures, and MPMD mismatches. A real successful two-rank MPMD solve and its CSV must pass before negative MPMD tests. Each subprocess is limited to 25 seconds (version probe 10 seconds); CTest limits the suite to 360 seconds. Negative cases require both nonzero status and application diagnostics; launcher failures do not count as passing tests. These tests were executed successfully with MPI on NUS Vanda during the 13 November 2026 standalone verification.

The recovered scripts/test_mpi_correctness.pbs retains its original openmpi/gcc/64/4.1.8 module, which differs from benchmark recipes. Use a fresh build directory/cache when switching module families. Its additional original manual launches are bounded by PBS walltime, not by CTest's per-test timeout.

## CLI and a small benchmark

Run only within an appropriate compute allocation:

```bash
mkdir -p results/mpi
mpirun -np 4 --map-by slot --bind-to core ./build/run_mpi_benchmark \
  2d 512 10 results/mpi/mpi_2d512_p4.csv 4
```

CLI is unchanged: `run_mpi_benchmark <2d|3d|thermal2> <grid_size|matrix.mtx> <repeats> <output.csv> [warmups]`.
Default warmups=1; campaigns explicitly use four warmups and ten measured runs. Zero warmups remain valid. Tolerance=1e-8 and maximum iterations=10000 are fixed in the runner; there is no max-iterations CLI extension. All ranks must supply equivalent arguments (including the same output filename). Malformed integer arguments are now rejected strictly. Ranks come from the MPI communicator. Poisson generators, manufactured exact solution of ones, RHS A*ones and zero initial guess are preserved.

## Reproduce the report configurations

The seven scripts in scripts/ are copied from the recovered original recipes at main revision `2e1c491`, with only local paths/account selection and removal of the obsolete CG_ENABLE_MPI option adapted. They retain original placement/resource directives and module settings. They write fresh results, not read saved measurements.

From task_b set its absolute location and submit each applicable script with your own account:

```bash
export TASK_B_DIR="$PWD"
qsub -P YOUR_ACCOUNT -v TASK_B_DIR scripts/test_mpi_correctness.pbs
# After inspecting correctness results and preparing a fresh benchmark-module build:
qsub -P YOUR_ACCOUNT -v TASK_B_DIR scripts/benchmark_mpi_scaling.pbs
qsub -P YOUR_ACCOUNT -v TASK_B_DIR scripts/benchmark_mpi_scaling_1024.pbs
qsub -P YOUR_ACCOUNT -v TASK_B_DIR scripts/benchmark_mpi_scaling_3d.pbs
qsub -P YOUR_ACCOUNT -v TASK_B_DIR scripts/benchmark_mpi_thermal2.pbs
qsub -P YOUR_ACCOUNT -v TASK_B_DIR scripts/benchmark_mpi_strong_256.pbs
qsub -P YOUR_ACCOUNT -v TASK_B_DIR scripts/benchmark_mpi_weak_256.pbs
```

These are separate jobs, not a dependency chain. Do not run jobs that configure the same build directory concurrently. The 512 script requires an already built executable; other campaign scripts configure/build their recorded targets. Strong/weak extension builds use job-specific directories. Preserve outputs before resubmitting a job; ordinary output names include the PBS job ID.

| Recipe | Workload | MPI ranks | Allocation |
|---|---|---|---|
| benchmark_mpi_scaling.pbs | 2D512 | 1,2,4,8,16,32,36,72 | 2 nodes, 36 CPUs/MPI slots each, 32 GB/node, 2 hours |
| benchmark_mpi_scaling_1024.pbs | 2D1024 | same eight counts | 2 x 36 slots, 64 GB/node, 3 hours |
| benchmark_mpi_scaling_3d.pbs | 3D40,64,102 | same eight counts for each grid | 2 x 36 slots, 64 GB/node, 3 hours |
| benchmark_mpi_thermal2.pbs | Schmid/thermal2 | 1,2,4,8,16 | 2 x 36 slots, 64 GB/node, 8 hours |
| benchmark_mpi_strong_256.pbs | 2D1024 | 1,2,4,8,16,32,36,72,144,256 | 8 x 36 slots, 64 GB/node, 6 hours |
| benchmark_mpi_weak_256.pbs | 2D grid per rank count | 1:128,4:256,16:512,64:1024,256:2048 | 8 x 36 slots, 64 GB/node, 6 hours |

All measured configurations retain ten runs/four warmups. The 2D/3D and extension scripts derive hostfiles from PBS_NODEFILE and use slot mapping/core binding. Extensions also preserve no-oversubscription and display-map flags and require at least 256 allocated slots. Scripts that originally exported OMP_NUM_THREADS=1 still do; no OpenMP solver is linked. Thermal2's original recipe has no explicit hostfile, mapping or binding flags; those settings depend on the MPI/PBS environment. No undocumented placement policy has been invented. To reproduce thermal2 placement precisely, recover the original effective map/environment.

Outputs go under results/mpi (3d/, thermal2/ and extension/ subdirectories where specified). Preserve module versions, host allocation and launcher output with newly collected results. Full timing reproduction also needs equivalent hardware and load; these recipes reproduce documented configurations, not exact historical times.

## Matrix data

Poisson inputs are generated. For thermal2, use Schmid/thermal2 from the original archive:
`https://www.cise.ufl.edu/research/sparse/MM/Schmid/thermal2.tar.gz`.

The unchanged downloader can be run from task_b on a machine with PowerShell and tar:

```powershell
pwsh -File scripts/download_thermal2.ps1 -Destination data/suitesparse
```

Transfer the extracted data/suitesparse/thermal2 directory to the same relative location under task_b on Vanda. Alternatively export THERMAL2_PATH to an absolute matrix filename and include it in qsub's -v list. The adapted thermal2 recipe fixes the historical mismatch between its data/thermal2 path and downloader's data/suitesparse path. No thermal2_b input is used: the runner manufactures b=A*ones. The expected matrix is 1,228,045 rows with 8,580,313 stored entries after loading; a verified expected archive/matrix checksum is not available. Download/network execution has not been verified here.

## Results, timing and limitations

The unchanged 12-column CSV is:
`solver,matrix_name,n,nnz,ranks,threads,run_index,iterations,converged,solve_seconds,true_relative_residual,relative_solution_error`.
MPI records have threads=1 and zero-based run_index. solve_seconds is the maximum rank's elapsed solver-call time: the pre-solve barrier, max-time reduction, input/distribution, warmups, solution-error calculation and CSV output lie outside the measured interval. Original internal solver checks remain timed; no newer validation reductions or profiling were imported.

Check all convergence flags and true residuals before using timings. Measured nonconvergence preserves CSV records but now produces a shared nonzero application exit status. Numerical breakdown throws and aborts instead; CSV completion is not promised for fatal errors. Warmup nonconvergence remains a warning as in the original.

Safety scope is the supplied runner, not a fully hardened library: caught C++ exceptions after MPI initialization abort MPI_COMM_WORLD, including rank-local allocation failures. Direct library callers must supply identical controls, valid coherent partitions and compatible collective sequences and arrange communicator-wide failure handling. Mismatched direct solver controls, malformed direct operation inputs or an unhandled allocation exception can still strand peers. MPI error returns, process failure and out-of-memory kills are not recoverable by C++ catch handlers. The successful Vanda tests cover the registered correctness and runner regressions, but do not certify every possible MPI failure path or resource-exhaustion scenario.

Pre-distribution checks reject global row counts above INT_MAX, per-rank row-pointer counts or CSR payloads exceeding int, and unsupported metadata ABIs. The original ceil(N/P) partition formula and P>N rejection remain. Empty trailing ranks can occur even when P<=N and now participate in benchmark error reductions. Large-count MPI/chunked transfers are not supported. See SOURCE_PROVENANCE.md for exact source differences and validation status.
