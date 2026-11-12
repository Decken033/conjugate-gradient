# Task A: Serial, OpenMP and optional GPU Conjugate Gradient

This directory is self-contained. It needs no MPI, hybrid code, sibling task, historical CSV or log. Production code is copied unchanged from 56ac360. Build and runtime execution are NOT VERIFIED in the migration environment; the commands below match the created targets and original CLI.

## CPU dependencies and build

Use Linux with CMake >=3.20, a C++17 compiler and CPU OpenMP. The recorded CPU campaign used GCC 13.2.0 and CMake/3.27.6-GCCcore-13.2.0 on Vanda. PBS campaigns also require Bash, environment modules and the Linux utilities referenced in each script.

From this directory, on Vanda load the recorded module, then configure a fresh build:

```bash
module load CMake/3.27.6-GCCcore-13.2.0
cmake -S . -B build -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS= \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG' \
  -DCG_ENABLE_OPENMP_GPU=OFF -DBUILD_TESTING=ON
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure --no-tests=error
```

Serial and CPU OpenMP are enabled. Seven CPU tests cover CSR, matrix loading, serial CG/benchmark and OpenMP operations/CG. Assertions remain enabled in Release test translation units. Production flags are unchanged. GPU is disabled by default.

## Benchmark interfaces and examples

```bash
mkdir -p results
./build/run_serial_benchmark 2d 512 10 results/serial_2d512.csv 4
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE
./build/run_openmp_benchmark 2d 512 8 static 0 10 results/openmp_2d512_t8.csv 4
```

Serial/GPU arguments: `matrix_type matrix_arg repeats output.csv [warmups]`.
OpenMP arguments: `matrix_type matrix_arg threads schedule chunk_size repeats output.csv [warmups]`.
Matrix types are `2d`, `3d` (grid edge length), or `thermal2` (Matrix Market filename). Schedules are `static`, `dynamic`, `guided`; the example uses static chunk 0. Runners fix tolerance at 1e-8 and maximum iterations at 10000; omitted warmups default to 1. Campaigns explicitly use 10 measurements and 4 warmups. Inputs use manufactured RHS A*1 and zero initial guess. Do not interpret an example as a replacement for a missing historical recipe.

## Reproduction scripts

Run PBS scripts from an allocation account authorized for the requested resources. Set TASK_A_DIR to this directory's absolute path. Account is supplied through qsub, not hardcoded. For example, after the CPU build above:

```bash
export TASK_A_DIR="$PWD"
qsub -P YOUR_ACCOUNT -v TASK_A_DIR scripts/benchmark_openmp_scaling_72.pbs
```

- `benchmark_serial_scaling.pbs`: one CPU, 8 GB, one hour; binds to the first allowed CPU, requires GCC 13.2.0, builds fresh and checks serial tests. Preserved order: 2D512, 3D16, 2D1024, 3D40, 2D64, 3D64, 2D256, repeated 2D512. Writes a unique campaign directory with eight CSVs, metadata, command records, source hashes and build logs.
- `benchmark_openmp_scaling_72.pbs`: 2D512, one 72-CPU node, 8 GB, 30 minutes; same-job serial reference then threads 1,2,4,8,16,32,36,72, static chunk 0.
- `benchmark_openmp_scaling_1024.pbs`: same sequence for 2D1024, 16 GB, one hour. Both OpenMP scripts retain close/core binding and OMP_DYNAMIC=FALSE and require the CPU build beforehand.
- `gpu_scaling_full.pbs`: recorded GPU campaign described below; requires a GPU build and thermal2 input beforehand.

OUTPUT_DIR optionally changes the output directory. BUILD_DIR optionally selects the prebuilt directory for OpenMP/GPU scripts; defaults are task_a/build and task_a/build-gpu. Export overrides and include their variable names in qsub -v. Serial always builds fresh under its unique output directory. Use fresh output directories for OpenMP/GPU runs to avoid overwriting previous results. Resource requests, modules, workload order and repetition counts are preserved.

No complete original OpenMP scheduling script was found. The report identifies 2D512, eight threads, static/dynamic/guided experiments and a start/end static drift check; exact historical job settings are not fully established. No replacement campaign is invented. See SOURCE_PROVENANCE.md.

## Optional GPU

Requires NVHPC nvc++, OpenMP GPU support and an NVIDIA GPU compatible with the preserved `-gpu=cc86` target (reported hardware: A40). Use a separate build directory. The recorded GPU environment used GCC/13.2.0 and NVHPC/24.7-CUDA-12.3.0; the recorded Release optimization was -O2.

Compile from `task_a/` on the login node, using the separate GPU build directory:

```bash
module load GCC/13.2.0
module load CMake/3.27.6-GCCcore-13.2.0
module load NVHPC/24.7-CUDA-12.3.0
cmake -S . -B build-gpu -DCMAKE_CXX_COMPILER=nvc++ \
  -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG' \
  -DCG_ENABLE_OPENMP_GPU=ON -DBUILD_TESTING=ON
cmake --build build-gpu --target test_gpu_cg run_gpu_benchmark --verbose
```

Run GPU correctness tests and benchmarks **only inside a properly allocated Vanda GPU job, not on the login node**. From `task_a/` inside that job, load the same modules and require GPU offloading:

```bash
module load GCC/13.2.0
module load CMake/3.27.6-GCCcore-13.2.0
module load NVHPC/24.7-CUDA-12.3.0
OMP_TARGET_OFFLOAD=MANDATORY ctest --test-dir build-gpu -R '^gpu_cg$' --output-on-failure --no-tests=error
mkdir -p results
OMP_TARGET_OFFLOAD=MANDATORY ./build-gpu/run_gpu_benchmark 2d 256 10 results/gpu_2d256.csv 4
```

Alternatively, submit the existing GPU campaign from `task_a/` on the login node after building and supplying thermal2. PBS allocates the GPU resources before executing its benchmarks:

```bash
export TASK_A_DIR="$PWD"
qsub -P YOUR_ACCOUNT -v TASK_A_DIR scripts/gpu_scaling_full.pbs
```

GPU campaign: two CPUs, one GPU, 12 GB, two hours; 2D64/256/512/1024, 3D16/40/64, thermal2, repeated 2D256. Mandatory offloading, ten repetitions and four warmups are preserved.

## Matrix inputs and outputs

Poisson matrices are generated internally. For thermal2 provide Schmid/thermal2 in Matrix Market format; no historical output is needed. The original PowerShell downloader is included:

```powershell
pwsh -File scripts/download_thermal2.ps1 -Destination data/suitesparse
```

It downloads from the original University of Florida sparse-matrix archive and requires tar. On Vanda, supply the matrix separately if PowerShell is unavailable; GPU script accepts THERMAL2_PATH, defaulting to task_a/data/suitesparse/thermal2/thermal2.mtx. A verified expected archive/matrix hash and a portable Linux download recipe remain missing; network download has not been tested here.

CSV schemas and run numbering are unchanged: measured solve times, iterations, convergence, true residual and solution error, plus configuration fields (OpenMP includes schedule/chunk). Warmups are excluded from measured records. Inspect convergence and residuals before computing summaries or speedups. Timing boundaries are unchanged; no profiling is added. Hardware, binding, compiler and background load affect times; exact historical times are not promised. CPU/GPU report comparisons used different nodes/toolchains. The scheduling report records substantial within-job drift.
