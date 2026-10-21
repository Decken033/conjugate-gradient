# DSA5208 — Serial and OpenMP CG Benchmark Tables

## Table 1. OpenMP Strong-Scaling Performance

| Threads | 512² median (s) | 512² speedup | 1024² median (s) | 1024² speedup |
|---:|---:|---:|---:|---:|
| Serial | 2.0751 | 1.000× | 36.5815 | 1.000× |
| 1 | 2.1283 | 0.975× | 29.8673 | 1.225× |
| 2 | 1.4373 | 1.444× | 21.1335 | 1.731× |
| 4 | 1.0752 | 1.930× | 14.7371 | 2.482× |
| 8 | 0.9165 | 2.264× | 11.4778 | 3.187× |
| 16 | 0.9395 | 2.209× | 9.4665 | 3.864× |
| 32 | 0.9325 | 2.225× | 9.0823 | 4.028× |
| 36 | 0.9702 | 2.139× | 8.2717 | 4.422× |
| 72 | 1.0397 | 1.996× | 7.1867 | 5.090× |

*Medians of 10 measured runs following 4 warmups per configuration. Speedup is relative to serial CG for the corresponding matrix. Matrix sizes were measured in separate Vanda jobs.*

## Table 2. OpenMP Parallel Efficiency

| Threads | 512² efficiency | 1024² efficiency |
|---:|---:|---:|
| 1 | 100.00% | 100.00% |
| 2 | 74.04% | 70.66% |
| 4 | 49.48% | 50.67% |
| 8 | 29.03% | 32.53% |
| 16 | 14.16% | 19.72% |
| 32 | 7.13% | 10.28% |
| 36 | 6.09% | 10.03% |
| 72 | 2.84% | 5.77% |

*Efficiency is calculated relative to OpenMP one-thread performance: (T_OMP1 / T_OMPp) / p × 100%.*

## Table 3. Numerical Correctness

| Matrix | Solver | Threads | N | NNZ | Iterations | Max. true residual | Max. solution error | Converged |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| poisson_2d_512 | serial | 1 | 262,144 | 1,308,672 | 894 | 9.526e-09 | 2.211e-08 | 10/10 |
| poisson_2d_512 | openmp | 8 | 262,144 | 1,308,672 | 894 | 9.526e-09 | 2.211e-08 | 10/10 |
| poisson_2d_1024 | serial | 1 | 1,048,576 | 5,238,784 | 1755 | 9.802e-09 | 4.819e-08 | 10/10 |
| poisson_2d_1024 | openmp | 72 | 1,048,576 | 5,238,784 | 1755 | 9.802e-09 | 4.819e-08 | 10/10 |
| poisson_3d_40 | serial | 1 | 64,000 | 438,400 | 101 | 8.836e-09 | 3.877e-09 | 5/5 |
| poisson_3d_40 | openmp | 4 | 64,000 | 438,400 | 101 | 8.836e-09 | 3.877e-09 | 3/3 |
| thermal2 | serial | 1 | 1,228,045 | 8,580,313 | 4333 | 9.861e-09 | 4.412e-07 | 3/3 |
| thermal2 | openmp | 4 | 1,228,045 | 8,580,313 | 4331 | 9.974e-09 | 4.432e-07 | 3/3 |

*All selected runs converged. The thermal2 experiment uses a constructed right-hand side b = A × 1. True relative residual and relative solution error are distinct metrics.*

## Table 4. OpenMP Scheduling Investigation

| Experiment group | Configuration | Median (s) | Min (s) | Max (s) | Converged |
|---|---|---:|---:|---:|---:|
| initial_policies | static | 0.9861 | 0.9421 | 1.1651 | 10/10 |
| initial_policies | dynamic | 15.5717 | 15.2828 | 15.6243 | 10/10 |
| initial_policies | guided | 0.9991 | 0.9577 | 1.2052 | 10/10 |
| dynamic_chunks | dynamic_chunk64 | 3.1496 | 3.0928 | 3.2506 | 10/10 |
| dynamic_chunks | dynamic_chunk256 | 2.7785 | 2.7592 | 2.8231 | 10/10 |
| dynamic_chunks | dynamic_chunk1024 | 2.6682 | 2.6271 | 2.7390 | 10/10 |
| final_policies_and_chunks | static_start | 0.9920 | 0.9842 | 1.0076 | 10/10 |
| final_policies_and_chunks | dynamic_default | 18.9238 | 18.8078 | 20.8426 | 10/10 |
| final_policies_and_chunks | dynamic_64 | 2.0314 | 1.6419 | 2.2552 | 10/10 |
| final_policies_and_chunks | dynamic_256 | 2.7606 | 1.4764 | 2.8287 | 10/10 |
| final_policies_and_chunks | dynamic_1024 | 2.6564 | 2.5950 | 2.6802 | 10/10 |
| final_policies_and_chunks | dynamic_4096 | 2.5925 | 2.4869 | 2.6415 | 10/10 |
| final_policies_and_chunks | guided_default | 2.7099 | 2.6905 | 2.7818 | 10/10 |
| final_policies_and_chunks | static_end | 2.3906 | 2.3370 | 2.4266 | 10/10 |

*The scheduling experiments used the 512×512 Poisson matrix with 8 OpenMP threads. The final job's static baseline slowed by 2.410× from beginning to end. Consequently, fine-grained scheduling comparisons are confounded by execution-condition drift.*
