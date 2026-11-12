#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-v1-profile}"
OUTPUT="${1:-results/v1-profile/hybrid_2d512_r2_t4_v1_profile.csv}"
REPEATS="${REPEATS:-10}"
WARMUPS="${WARMUPS:-5}"

mkdir -p "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=true
export OMP_PLACES=cores

mpirun -np 2 --map-by slot:PE=4 --bind-to core \
  "./${BUILD_DIR}/run_hybrid_benchmark" \
  2d 512 4 static 0 "$REPEATS" "$OUTPUT" "$WARMUPS"
