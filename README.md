# DSA5208: standalone Conjugate Gradient experiments

Three independent tasks solve sparse SPD systems using double-precision CG. Each package preserves its own historical numerical implementation and benchmark methodology. There is no root CMake build or shared runtime dependency between tasks.

| Task | Package and instructions | Dependencies |
|---|---|---|
| A: serial, CPU OpenMP, optional GPU | [Task A](task_a/README.md) | C++17, CMake >=3.20, CPU OpenMP; optional NVHPC GPU toolchain |
| B: MPI | [Task B](task_b/README.md) | LP64 Linux, C++17, CMake >=3.20, MPI; Python 3 for tests |
| C: hybrid MPI/OpenMP | [Task C overview](task_c/README.md) | C++17, CMake >=3.20, MPI, CPU OpenMP |
| C handoff: unprofiled WSL experiments | [Handoff](task_c/handoff/README.md) | Recorded Ubuntu 24.04/WSL2 environment |
| C V1: workspace reuse and profiling | [V1 profiling](task_c/v1_profile/README.md) | Python 3 for profile summaries; original profiled comparators included |

## Build and correctness tests

Use separate build directories and matching compiler/MPI modules. On Vanda run correctness tests and benchmarks inside allocated compute jobs, never on the login node. These commands are from the repository root; consult each task README for full environment details.

Task A CPU (recorded GCC 13.2 toolchain):

```bash
module load CMake/3.27.6-GCCcore-13.2.0
cmake -S task_a -B task_a/build -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS= \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG' \
  -DCG_ENABLE_OPENMP_GPU=OFF -DBUILD_TESTING=ON
cmake --build task_a/build --parallel 4
ctest --test-dir task_a/build --output-on-failure --no-tests=error
```

Task B (allocate at least four MPI slots for its tests):

```bash
module purge
module load CMake/3.27.6-GCCcore-13.2.0
module load OpenMPI/4.1.6-GCC-13.2.0
cmake -S task_b -B task_b/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DMPI_CXX_COMPILER=mpicxx -DBUILD_TESTING=ON
cmake --build task_b/build --parallel 4
ctest --test-dir task_b/build --output-on-failure --no-tests=error -j 1
```

Build Task C variants separately. The handoff's historical environment was WSL, not Vanda. On Vanda the V1 documentation specifies CMake/3.27.6-GCCcore-13.2.0 and openmpi/gcc/64/4.1.8. Use fresh caches when changing toolchains. Allocate four MPI slots and eight cores for either variant's tests.

```bash
cmake -S task_c/handoff -B task_c/handoff/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DMPI_CXX_COMPILER=mpicxx -DBUILD_TESTING=ON
cmake --build task_c/handoff/build --parallel 4
ctest --test-dir task_c/handoff/build --output-on-failure --no-tests=error -j 1

cmake -S task_c/v1_profile -B task_c/v1_profile/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DMPI_CXX_COMPILER=mpicxx -DBUILD_TESTING=ON
cmake --build task_c/v1_profile/build --parallel 4
ctest --test-dir task_c/v1_profile/build --output-on-failure --no-tests=error -j 1
```

## Benchmark entry points

Run from the specified package directory, with sufficient allocated cores/ranks. Use fresh output names: original writers can overwrite files. These examples do not replace the complete campaigns documented in each task README.

```bash
# From task_a: serial and OpenMP examples
mkdir -p results
./build/run_serial_benchmark 2d 512 10 results/serial_2d512.csv 4
OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE \
  ./build/run_openmp_benchmark 2d 512 8 static 0 10 results/openmp_2d512_t8.csv 4

# From task_b: MPI example
mkdir -p results/mpi
mpirun -np 4 --map-by slot --bind-to core ./build/run_mpi_benchmark \
  2d 512 10 results/mpi/mpi_2d512_p4.csv 4

# From task_c/handoff: original unprofiled configuration
mkdir -p results/hybrid
OMP_PROC_BIND=true OMP_PLACES=cores mpirun -np 2 ./build/run_hybrid_benchmark \
  2d 256 4 static 0 5 results/hybrid/hybrid_2d256_r2_t4_static.csv 1

# From task_c/v1_profile: original profiled V1 campaign
BUILD_DIR=build bash scripts/run_hybrid_v1_profile_2x4.sh
```

V1 retains grid512, two ranks/four threads, static/0, ten measurements/five warmups, original profiling overhead and per-rank sidecars. Handoff scheduling cases include grids64/256, two ranks/four threads and five measurements/one warmup. Their timing series are not interchangeable.

For PBS campaigns, from `task_a` export `TASK_A_DIR="$PWD"`; from `task_b` export `TASK_B_DIR="$PWD"`. Submit with your authorized account, for example:

```bash
# From task_a, after the required CPU build
export TASK_A_DIR="$PWD"
qsub -P YOUR_ACCOUNT -v TASK_A_DIR scripts/benchmark_openmp_scaling_72.pbs
# From task_b, after the benchmark-module build
export TASK_B_DIR="$PWD"
qsub -P YOUR_ACCOUNT -v TASK_B_DIR scripts/benchmark_mpi_scaling.pbs
```

Task A contains serial scaling, two OpenMP scaling recipes and the GPU campaign. Task B contains correctness, 2D512/1024, 3D, thermal2 and strong/weak 256-rank recipes. Follow task-specific resource, module, output and binding requirements; do not configure a shared build directory concurrently. Archived scripts are historical snapshots, not supported launch entry points.

## Optional Task A GPU

See [GPU compilation and execution](task_a/README.md). Load the documented CMake module and NVHPC 24.7/CUDA 12.3 environment, configure with `nvc++`, GPU support enabled and the documented Release flags. Build targets are `test_gpu_cg` and `run_gpu_benchmark`; offloading uses `-mp=gpu -gpu=cc86` and requires a compatible GPU. GPU compilation is distinct from execution: correctness tests and benchmarks require a properly allocated Vanda GPU job and `OMP_TARGET_OFFLOAD=MANDATORY`. Historical CPU/GPU timings used different toolchains/nodes.

## thermal2 dataset on Linux/Vanda

Poisson matrices are generated; no archived CSV is needed. From the package directory that will run thermal2, use the original [University of Florida archive](https://www.cise.ufl.edu/research/sparse/MM/Schmid/thermal2.tar.gz), corresponding to [SuiteSparse Schmid/thermal2](https://sparse.tamu.edu/Schmid/thermal2):

```bash
mkdir -p data/suitesparse
curl -fL --retry 3 -o data/suitesparse/thermal2.tar.gz \
  https://www.cise.ufl.edu/research/sparse/MM/Schmid/thermal2.tar.gz
gzip -t data/suitesparse/thermal2.tar.gz
tar -xzf data/suitesparse/thermal2.tar.gz -C data/suitesparse
head -n 1 data/suitesparse/thermal2/thermal2.mtx
```

Expected header: `%%MatrixMarket matrix coordinate real symmetric`. The historical loaded matrix has 1,228,045 rows and 8,580,313 expanded CSR entries. Pass `data/suitesparse/thermal2/thermal2.mtx` to the runner; Task A GPU/Task B recipes also accept the documented `THERMAL2_PATH` override. No verified checksum is supplied and current network availability was not tested. The accompanying `thermal2_b.mtx` is not used: runners manufacture `b=A*ones`, with a zero initial guess, tolerance 1e-8 and maximum 10000 iterations. Dataset files are local generated inputs, excluded from Git.

## Reports, provenance and verification status

- [Task A report](REPORT_TASK_A.md), [Task A methodology](docs/task_a/benchmark_methodology.md), [Task B report](REPORT_TASK_B.md), [Task B evidence](docs/task_b/evidence/README.md).
- [Historical handoff/archive index](docs/archive/README.md), including original reports, scheduling evidence and the exact cleanup inventory. Existing `results/` evidence is retained for report verification, not used as benchmark input.
- [Task A provenance](task_a/SOURCE_PROVENANCE.md), [Task B provenance](task_b/SOURCE_PROVENANCE.md), [Task C provenance](task_c/SOURCE_PROVENANCE.md), [source manifest](STANDALONE_SOURCE_MANIFEST.md).

Task A and historical Task B derive from 56ac360; Task B retains its documented three safety deviations outside the normal timed solver operations. Task C handoff derives from 56ac360 and V1 from 846239d. Protected integration work remains available separately. Profiled V0 source provenance and the final Task C report remain unresolved; neither variant reconstructs V0-to-V1 speedups. Complete original Task A scheduling scripts/raw OpenMP evidence and some historical build/placement fingerprints remain missing. Exact historical timing identity is not promised.

Standalone compilation and runtime are **NOT VERIFIED** in the cleanup environment. Historical reports describe historical executions, not tests of these packages. Original Task C MPI failure-path limitations remain; use identical valid arguments and appropriate allocations.

From the repository root, these Python checks validate archived evidence only:

```bash
python3 scripts/verify_mpi_evidence.py
python3 scripts/validate_task_b_report.py
python3 scripts/validate_gpu_evidence.py
```

Analysis/plotting utilities under `scripts/` can require matplotlib/Pillow and archived evidence. `archive_mpi_evidence.py` is a historical import utility requiring an external backup; it is not needed for build, benchmarking or evidence validation.
