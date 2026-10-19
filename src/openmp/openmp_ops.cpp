
#include "openmp_ops.hpp"

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

} // namespace cg
