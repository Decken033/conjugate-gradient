# DSA5208 — Vanda Environment and Build Configuration

## 1. Overview

The project uses NUS Vanda HPC for CPU and GPU compilation, correctness testing and performance benchmarking.

Environment verification was conducted on 19 October 2026.

The tested configurations support:

- Serial C++17
- CPU OpenMP
- MPI
- OpenMP target offloading to NVIDIA GPU

Hybrid MPI + OpenMP execution remains to be tested as an integrated implementation.

## 2. Access and Storage

Connect through NUS VPN and SSH:

```bash
ssh vanda
```

The Vanda login node is used for file management, job submission and other lightweight operations. Compilation and performance-sensitive testing should be performed on allocated compute nodes.

User storage locations:

- Home: `/home/svu/e1536052`
- Scratch: `/scratch/e1536052`

Vanda scratch files older than 60 days may be purged. Preserve project source code and important results in Git or other durable storage.

The verified PBS project is `personal-e1536052`. Other team members must substitute their own authorized PBS project names.

## 3. CPU Environment

Verified on compute node `CN-149`.

| Component | Verified version |
|---|---|
| GCC C++ compiler | 13.1.0 |
| Additional tested GCC version | 13.2.0 |
| C++ language standard | C++17 |
| CPU OpenMP | `_OPENMP=201511` |
| Open MPI | 4.1.8 |
| CMake | 3.27.6 |
| GNU Make | 4.3 |

### CPU compiler modules

```bash
module load gcc/13.1.0
module load openmpi/gcc/64/4.1.8
```

### CMake module

```bash
module load CMake/3.27.6-GCCcore-13.2.0
```

**Toolchain caution:** Loading the CMake module was observed to change the active GCC compiler to 13.2.0. Verify the final active compiler and MPI wrapper after loading all modules.

```bash
g++ --version
mpicxx --showme
cmake --version
```

Use a consistent compiler and MPI configuration for production builds.

## 4. CPU Interactive Jobs

Example two-core allocation for CPU OpenMP testing:

```bash
qsub -I \
  -P personal-e1536052 \
  -q interactive_cpu \
  -l select=1:ncpus=2:mem=4gb \
  -l walltime=00:30:00
```

Example two-process MPI allocation:

```bash
qsub -I \
  -P personal-e1536052 \
  -q interactive_cpu \
  -l select=1:ncpus=2:mpiprocs=2:ompthreads=1:mem=4gb \
  -l walltime=00:30:00
```

For MPI execution, explicitly specify `mpiprocs`.

A previous allocation requesting `ncpus=2` without `mpiprocs=2` provided only one MPI slot and caused a two-process `mpirun` command to fail.

The corrected allocation produced two entries in `$PBS_NODEFILE` and successfully executed two MPI ranks.

### CPU OpenMP compilation

```bash
g++ -std=c++17 -O3 -fopenmp source.cpp -o program
```

Example execution:

```bash
export OMP_NUM_THREADS=2
./program
```

### MPI compilation

```bash
mpicxx -std=c++17 -O3 source.cpp -o program
```

Example execution inside a correctly allocated PBS job:

```bash
mpirun -np 2 ./program
```

## 5. MPI Verification

Open MPI 4.1.8 successfully compiled and executed a C++17 MPI smoke test.

Observed output:

```text
Hello from rank 0 of 2
Hello from rank 1 of 2
```

A warning was also observed:

```text
mca_osc_ucx: libucp.so.0: cannot open shared object file
```

MPI execution completed successfully despite this warning.

Investigate the UCX environment before relying on communication performance benchmarks. Do not assume the warning has no performance implications.

## 6. GPU Environment

Verified on GPU compute node `GN-A40-041`.

| Component | Verified value |
|---|---|
| GPU | NVIDIA A40 |
| GPU memory | 46,068 MiB reported |
| NVIDIA driver | 575.57.08 |
| NVIDIA HPC SDK | 24.5 |
| NVIDIA C++ compiler | `nvc++ 24.5-1` |
| GPU compute capability | 8.6 |
| OpenMP target devices detected | 1 |

### GPU interactive allocation

```bash
qsub -I \
  -P personal-e1536052 \
  -l select=1:ngpus=1 \
  -l walltime=00:15:00
```

The allocation used the GPU interactive job configuration documented by `hpc gpu`.

### Load NVIDIA HPC SDK

```bash
module load nvhpc/24.5
```

### GPU offloading compilation

```bash
nvc++ -std=c++17 -mp=gpu -gpu=cc86 \
  source.cpp -o program
```

### GPU execution

```bash
OMP_TARGET_OFFLOAD=MANDATORY ./program
```

## 7. GPU Offloading Verification

A minimal C++17 OpenMP program was compiled using NVIDIA HPC SDK 24.5 and executed on the NVIDIA A40.

The program used `omp_is_initial_device()` to verify that its target region executed on an accelerator rather than the host CPU.

Observed results:

```text
OpenMP devices: 1
Executed on GPU: YES
Computed sum: 100000
Expected sum: 100000
Exit status: 0
```

This confirms basic OpenMP target offloading and numerical correctness.

It does not yet establish the performance or scalability of a GPU Conjugate Gradient solver.

## 8. Build and Benchmarking Principles

- Build on allocated compute nodes when targeting Vanda compute hardware.
- Keep CPU-only builds independent of GPU compiler requirements.
- Record compiler version, modules, PBS resources and runtime configuration.
- Use consistent optimization settings when comparing implementations.
- Distinguish MPI ranks, CPU OpenMP threads and GPUs.
- Benchmark only after correctness checks pass.
- Repeat measurements and report the chosen timing statistic.
- Preserve scripts and configurations needed to reproduce results.

## 9. Outstanding Environment Checks

- Verify a complete CMake build of the shared project.
- Test integrated hybrid MPI + OpenMP execution.
- Investigate the optional UCX component warning.
- Establish reproducible PBS batch scripts for full benchmarks.
- Test the actual GPU CG kernels and their data-transfer behavior.
- Validate CPU/GPU build options across all team members' environments.

## 10. Vanda Documentation

NUS HPC onboarding:

https://nusit.nus.edu.sg/hpc/

Useful Vanda commands:

```bash
hpc cpu
hpc gpu
hpc project
qstat -Q
qstat -awn1
```