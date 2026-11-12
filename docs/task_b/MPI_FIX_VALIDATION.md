# MPI collective-safety fix: validation status

## Status

Source fixes and bounded regression tests have been prepared. Native C++/MPI build tools are unavailable in the current Windows session and WSL discovery failed with access denied. **The new C++ changes have NOT been compiled or executed here. Final correctness acceptance remains pending an MPI-equipped Vanda run.** Historical CSV/log checks are separate and do not certify the changed implementation.

## Changes

- Benchmark error norms are reduced unconditionally, including zero-row ranks. Input dimensions/values, partition boundaries, repeats, warmups and solver options are checked collectively before solver loops.
- Root matrix validation reports failure before dimension distribution. Partition/vector collective APIs coordinate local validation failures.
- Runner root matrix preparation and CSV output failures are propagated before peers proceed. Unexpected exceptions use communicator fail-fast termination instead of potentially hanging in Finalize.
- Existing ceil-block partitioning is retained; supported zero-row trailing partitions are exercised. The P>N partition_matrix restriction remains.
- CTest safety tests use timeouts. Runner integration tests require Python3 and test application diagnostics as well as nonzero return codes, so launcher failures are not accepted as successful negative tests.

## Vanda acceptance procedure

Use a new source checkout/branch with the exact staged changes committed by the user. Do not reuse historical build caches or overwrite historical CSVs.

```bash
module purge
module load CMake/3.27.6-GCCcore-13.2.0
module load OpenMPI/4.1.6-GCC-13.2.0
export OMP_NUM_THREADS=1
cmake -S . -B build-mpi-safety -DCMAKE_BUILD_TYPE=Release -DCG_ENABLE_MPI=ON -DBUILD_TESTING=ON
cmake --build build-mpi-safety --parallel 8
ctest --test-dir build-mpi-safety -N
ctest --test-dir build-mpi-safety --output-on-failure --timeout 180
python3 -B scripts/verify_mpi_evidence.py
python3 -B scripts/validate_task_b_report.py
```

Run inside a Vanda allocation with at least four MPI slots. Confirm the listed tests include mpi_safety_1/2/4, mpi_ops_safety_1/2/4, mpi_cg_single and mpi_runner_safety. If Python or launcher settings cause runner integration tests to be omitted, resolve the test environment rather than claiming full acceptance. Preserve commit ID, module/compiler/runtime versions, configure/build output, CTest output/exit code and PBS job ID. Successful negative cases must terminate within timeout; timeout/launcher errors are failures. Add representative 2D/3D/thermal2 numerical smoke tests after regression passes; rerun affected performance campaigns if changed timed code impacts published comparisons.

## Performance comparison caution

Collective validation adds error-agreement reductions to SpMV, dot and residual operations. These occur inside the solver timing boundary, even though the boundary itself is unchanged. Historical timing results must remain labeled pre-fix; do not present them as measurements of this revision. Rebenchmark before making performance claims for the fixed revision.

## Remaining limits

This is not general recovery from allocation failure or MPI runtime/process failure. Library users must invoke collective APIs consistently on the same communicator. Existing size_t/MPI_UNSIGNED_LONG metadata outside benchmark NNZ remains intended for LP64 Linux, not a Windows portability certification. No new Vanda job has been submitted during this work.
