
#include "openmp/openmp_ops.hpp"
#include "common/ops_profiler.hpp"

#include <omp.h>

#include <stdexcept>
#include <vector>

namespace cg {

std::vector<double> csr_matvec_openmp(
    const CSRMatrix& A,
    const std::vector<double>& x,
    const OpenMPOptions& options
) {
    // Validate inputs before entering any OpenMP region.
    A.validate();

    if (x.size() != A.cols) {
        throw std::invalid_argument(
            "OpenMP SpMV: dimension mismatch"
        );
    }

    if (options.num_threads < 0) {
        throw std::invalid_argument(
            "OpenMP SpMV: negative thread count"
        );
    }

    if (options.chunk_size < 0) {
        throw std::invalid_argument(
            "OpenMP SpMV: negative chunk size"
        );
    }

    const int threads = options.num_threads == 0
        ? omp_get_max_threads()
        : options.num_threads;

    std::vector<double> y(A.rows, 0.0);
    ProfileScope timer(ProfileSection::SpmvCompute);

    // Each row is independent:
    // - A and x are shared and read-only.
    // - Each thread writes only its assigned y[i].
    // - sum and k are private to the row calculation.
    const auto compute_row = [&](std::size_t i) {
        double sum = 0.0;

        for (std::size_t k = A.row_ptr[i];
             k < A.row_ptr[i + 1]; ++k) {
            sum += A.values[k] * x[A.col_idx[k]];
        }

        y[i] = sum;
    };

    const int chunk = options.chunk_size;

    switch (options.schedule) {
        case OpenMPSchedule::Static:
            if (chunk == 0) {
                #pragma omp parallel for schedule(static) num_threads(threads)
                for (std::size_t i = 0; i < A.rows; ++i) {
                    compute_row(i);
                }
            } else {
                #pragma omp parallel for schedule(static, chunk) num_threads(threads)
                for (std::size_t i = 0; i < A.rows; ++i) {
                    compute_row(i);
                }
            }
            break;

        case OpenMPSchedule::Dynamic:
            if (chunk == 0) {
                #pragma omp parallel for schedule(dynamic) num_threads(threads)
                for (std::size_t i = 0; i < A.rows; ++i) {
                    compute_row(i);
                }
            } else {
                #pragma omp parallel for schedule(dynamic, chunk) num_threads(threads)
                for (std::size_t i = 0; i < A.rows; ++i) {
                    compute_row(i);
                }
            }
            break;

        case OpenMPSchedule::Guided:
            if (chunk == 0) {
                #pragma omp parallel for schedule(guided) num_threads(threads)
                for (std::size_t i = 0; i < A.rows; ++i) {
                    compute_row(i);
                }
            } else {
                #pragma omp parallel for schedule(guided, chunk) num_threads(threads)
                for (std::size_t i = 0; i < A.rows; ++i) {
                    compute_row(i);
                }
            }
            break;

        default:
            throw std::invalid_argument(
                "OpenMP SpMV: unsupported scheduling policy"
            );
    }

    return y;
}

/*
 * Parallel dot product using OpenMP reduction.
 *
 * Each thread accumulates a private partial sum.
 * OpenMP combines the partial sums at the end.
 *
 * The reduction order may differ from serial execution,
 * so small floating-point differences are expected.
 */
double dot_openmp(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const OpenMPOptions& options
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "OpenMP dot product: dimension mismatch"
        );
    }

    if (options.num_threads < 0) {
        throw std::invalid_argument(
            "OpenMP dot product: negative thread count"
        );
    }

    if (options.chunk_size < 0) {
        throw std::invalid_argument(
            "OpenMP dot product: negative chunk size"
        );
    }

    const int threads = options.num_threads == 0
        ? omp_get_max_threads()
        : options.num_threads;

    double sum = 0.0;
    ProfileScope timer(ProfileSection::DotCompute);

    // Static scheduling is used for this reduction.
    // Scheduling comparisons initially focus on CSR SpMV.
    #pragma omp parallel for schedule(static) \
        num_threads(threads) reduction(+:sum)
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

/*
 * Parallel CG solution and residual update.
 *
 * x[i] = x[i] + alpha * p[i]
 * r[i] = r[i] - alpha * Ap[i]
 *
 * Each thread processes independent vector indices.
 * No reduction, lock or critical section is required.
 */
void update_solution_residual_openmp(
    std::vector<double>& x,
    std::vector<double>& r,
    const std::vector<double>& p,
    const std::vector<double>& Ap,
    double alpha,
    const OpenMPOptions& options
) {
    const std::size_t n = x.size();

    // Validate dimensions before entering the parallel region.
    if (r.size() != n || p.size() != n || Ap.size() != n) {
        throw std::invalid_argument(
            "OpenMP solution/residual update: dimension mismatch"
        );
    }

    if (options.num_threads < 0 || options.chunk_size < 0) {
        throw std::invalid_argument(
            "OpenMP solution/residual update: invalid OpenMP options"
        );
    }

    const int threads = options.num_threads == 0
        ? omp_get_max_threads()
        : options.num_threads;

    // Static scheduling: equal work per vector element.
    ProfileScope timer(ProfileSection::SolutionResidualUpdate);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
    }
}

/*
 * Parallel CG search-direction update.
 *
 * p[i] = r[i] + beta * p[i]
 *
 * Each thread updates a separate element of p.
 * r is read-only and must not alias p.
 */
void update_direction_openmp(
    std::vector<double>& p,
    const std::vector<double>& r,
    double beta,
    const OpenMPOptions& options
) {
    const std::size_t n = p.size();

    // Validate before starting parallel execution.
    if (r.size() != n) {
        throw std::invalid_argument(
            "OpenMP direction update: dimension mismatch"
        );
    }

    if (options.num_threads < 0 || options.chunk_size < 0) {
        throw std::invalid_argument(
            "OpenMP direction update: invalid OpenMP options"
        );
    }

    const int threads = options.num_threads == 0
        ? omp_get_max_threads()
        : options.num_threads;

    // Static scheduling: equal work per vector element.
    ProfileScope timer(ProfileSection::DirectionUpdate);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = r[i] + beta * p[i];
    }
}

/*
 * Parallel vector subtraction for true residual verification.
 *
 * residual[i] = b[i] - Ax[i]
 *
 * Each thread writes to a separate residual element.
 * No reduction or explicit synchronization is required.
 */
std::vector<double> subtract_openmp(
    const std::vector<double>& b,
    const std::vector<double>& Ax,
    const OpenMPOptions& options
) {
    const std::size_t n = b.size();

    // Validate before entering the parallel region.
    if (Ax.size() != n) {
        throw std::invalid_argument(
            "OpenMP subtraction: dimension mismatch"
        );
    }

    if (options.num_threads < 0 || options.chunk_size < 0) {
        throw std::invalid_argument(
            "OpenMP subtraction: invalid OpenMP options"
        );
    }

    const int threads = options.num_threads == 0
        ? omp_get_max_threads()
        : options.num_threads;

    std::vector<double> residual(n, 0.0);
    ProfileScope timer(ProfileSection::ResidualSubtract);

    // Static scheduling: independent element-wise subtraction.
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < n; ++i) {
        residual[i] = b[i] - Ax[i];
    }

    return residual;
}

} // namespace cg
