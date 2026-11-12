#include "gpu_solver.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace cg {
namespace {

// Compute y = A*x using CSR storage on a GPU.
//
// The caller must allocate row_ptr, col_idx, values, x and y
// using omp_target_alloc on the default device.
//
// All pointers are device addresses, not mapped host pointers.
void gpu_csr_matvec(
    const std::size_t* row_ptr,
    const std::size_t* col_idx,
    const double* values,
    const double* x,
    double* y,
    std::size_t n
) {
    #pragma omp target teams distribute parallel for \
        is_device_ptr(row_ptr, col_idx, values, x, y)
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;

        for (std::size_t k = row_ptr[i];
             k < row_ptr[i + 1]; ++k) {
            sum += values[k] * x[col_idx[k]];
        }

        y[i] = sum;
    }
}

// GPU dot product using an OpenMP reduction.
//
// The input pointers must refer to GPU device memory.
double gpu_dot(
    const double* a,
    const double* b,
    std::size_t n
) {
    double sum = 0.0;

    #pragma omp target teams distribute parallel for \
        is_device_ptr(a, b) reduction(+:sum)
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

// Update the solution and recursive residual on the GPU.
// x = x + alpha*p
// r = r - alpha*Ap
void gpu_update_solution_residual(
    double* x,
    double* r,
    const double* p,
    const double* Ap,
    double alpha,
    std::size_t n
) {
    #pragma omp target teams distribute parallel for \
        is_device_ptr(x, r, p, Ap)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
    }
}

// Update the CG search direction on the GPU.
// p = r + beta*p
void gpu_update_direction(
    double* p,
    const double* r,
    double beta,
    std::size_t n
) {
    #pragma omp target teams distribute parallel for \
        is_device_ptr(p, r)
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = r[i] + beta * p[i];
    }
}

// Recompute the true residual on the GPU:
// verified_r = b - A*x.
//
// scratch is used for A*x and may overwrite its previous contents.
// All array pointers must refer to GPU device memory.
void gpu_true_residual(
    const std::size_t* row_ptr,
    const std::size_t* col_idx,
    const double* values,
    const double* b,
    const double* x,
    double* scratch,
    double* verified_r,
    std::size_t n
) {
    gpu_csr_matvec(
        row_ptr, col_idx, values, x, scratch, n
    );

    #pragma omp target teams distribute parallel for \
        is_device_ptr(b, scratch, verified_r)
    for (std::size_t i = 0; i < n; ++i) {
        verified_r[i] = b[i] - scratch[i];
    }
}

// Copy one GPU-resident vector into another.
// Used to restart CG from a verified residual.
void gpu_copy_vector(
    double* destination,
    const double* source,
    std::size_t n
) {
    #pragma omp target teams distribute parallel for \
        is_device_ptr(destination, source)
    for (std::size_t i = 0; i < n; ++i) {
        destination[i] = source[i];
    }
}

// Verify that OpenMP can execute on a real target device.
// This check must run before GPU memory allocation.
void require_gpu_device() {
    if (omp_get_num_devices() < 1) {
        throw std::runtime_error(
            "GPU CG: no OpenMP target device available"
        );
    }

    int on_device = 0;

    #pragma omp target map(from:on_device)
    {
        on_device = !omp_is_initial_device();
    }

    if (!on_device) {
        throw std::runtime_error(
            "GPU CG: OpenMP target executed on host"
        );
    }
}

// Owns a GPU allocation and releases it automatically.
// Noncopyable to prevent double-free errors.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer(std::size_t count, int device)
        : count_(count), device_(device) {
        const std::size_t allocated_count =
            count == 0 ? 1 : count;

        ptr_ = static_cast<T*>(
            omp_target_alloc(
                allocated_count * sizeof(T), device_
            )
        );

        if (!ptr_) {
            throw std::runtime_error(
                "GPU CG: device allocation failed"
            );
        }
    }

    ~DeviceBuffer() {
        if (ptr_) {
            omp_target_free(ptr_, device_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const noexcept {
        return ptr_;
    }

    void upload(const T* host) {
        if (count_ == 0) return;

        const int status = omp_target_memcpy(
            ptr_, host, count_ * sizeof(T),
            0, 0,
            device_, omp_get_initial_device()
        );

        if (status != 0) {
            throw std::runtime_error(
                "GPU CG: host-to-device copy failed"
            );
        }
    }

    void download(T* host) const {
        if (count_ == 0) return;

        const int status = omp_target_memcpy(
            host, ptr_, count_ * sizeof(T),
            0, 0,
            omp_get_initial_device(), device_
        );

        if (status != 0) {
            throw std::runtime_error(
                "GPU CG: device-to-host copy failed"
            );
        }
    }

private:
    std::size_t count_;
    int device_;
    T* ptr_ = nullptr;
};

} // namespace

CGResult solve_gpu_cg(
    const CSRMatrix& A,
    const std::vector<double>& b,
    const CGOptions& options
) {
    A.validate();

    if (b.size() != A.rows) {
        throw std::invalid_argument("GPU CG: RHS dimension mismatch");
    }

    if (!std::isfinite(options.relative_tolerance) ||
        options.relative_tolerance <= 0.0) {
        throw std::invalid_argument(
            "GPU CG: tolerance must be positive and finite"
        );
    }

    for (double value : b) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("GPU CG: nonfinite RHS value");
        }
    }

    // Match the serial solver's initial guess and RHS norm.
    CGResult result;
    const std::size_t n = A.rows;
    result.x.assign(n, 0.0);

    double bb = 0.0;
    for (double value : b) {
        bb += value * value;
    }

    const double b_norm = std::sqrt(bb);

    if (!std::isfinite(b_norm)) {
        throw std::invalid_argument("GPU CG: invalid RHS norm");
    }

    // Require an actual OpenMP GPU device.
    require_gpu_device();

    if (b_norm == 0.0) {
        result.converged = true;
        result.relative_residual = 0.0;
        return result;
    }

    result.relative_residual = 1.0;

    // The initial CG state is x = 0, r = b, p = r.
    std::vector<double> r = b;
    std::vector<double> p = b;

    const int device = omp_get_default_device();

    DeviceBuffer<std::size_t> d_row_ptr(
        A.row_ptr.size(), device
    );
    DeviceBuffer<std::size_t> d_col_idx(
        A.col_idx.size(), device
    );
    DeviceBuffer<double> d_values(
        A.values.size(), device
    );

    DeviceBuffer<double> d_b(n, device);
    DeviceBuffer<double> d_x(n, device);
    DeviceBuffer<double> d_r(n, device);
    DeviceBuffer<double> d_p(n, device);
    DeviceBuffer<double> d_Ap(n, device);
    DeviceBuffer<double> d_verified_r(n, device);

    // Transfer the fixed matrix and initial CG vectors once.
    d_row_ptr.upload(A.row_ptr.data());
    d_col_idx.upload(A.col_idx.data());
    d_values.upload(A.values.data());

    d_b.upload(b.data());
    d_x.upload(result.x.data());
    d_r.upload(r.data());
    d_p.upload(p.data());

    // All subsequent kernels operate on GPU device pointers.
    const auto* row_ptr = d_row_ptr.data();
    const auto* col_idx = d_col_idx.data();
    const auto* values = d_values.data();

    const auto* db = d_b.data();
    auto* dx = d_x.data();
    auto* dr = d_r.data();
    auto* dp = d_p.data();
    auto* dAp = d_Ap.data();
    auto* dverified = d_verified_r.data();

    double rr = gpu_dot(dr, dr, n);

    // Same classical CG recurrence as the serial solver.
    for (std::size_t k = 0; k < options.max_iterations; ++k) {
        gpu_csr_matvec(
            row_ptr, col_idx, values, dp, dAp, n
        );

        const double pAp = gpu_dot(dp, dAp, n);

        if (!std::isfinite(pAp) || pAp <= 0.0) {
            throw std::runtime_error(
                "GPU CG breakdown: invalid p^T A p"
            );
        }

        const double alpha = rr / pAp;

        if (!std::isfinite(alpha)) {
            throw std::runtime_error(
                "GPU CG breakdown: invalid alpha"
            );
        }

        gpu_update_solution_residual(
            dx, dr, dp, dAp, alpha, n
        );

        result.iterations = k + 1;

        const double rr_new = gpu_dot(dr, dr, n);

        if (!std::isfinite(rr_new) || rr_new < 0.0) {
            throw std::runtime_error(
                "GPU CG breakdown: invalid recursive residual"
            );
        }

        const double recursive_relative =
            std::sqrt(rr_new) / b_norm;

        if (!std::isfinite(recursive_relative)) {
            throw std::runtime_error(
                "GPU CG breakdown: invalid relative residual"
            );
        }

        if (recursive_relative <= options.relative_tolerance) {
            // Compute b - A*x independently on the GPU.
            gpu_true_residual(
                row_ptr, col_idx, values,
                db, dx, dAp, dverified, n
            );

            const double verified_rr =
                gpu_dot(dverified, dverified, n);

            const double verified_relative =
                std::sqrt(verified_rr) / b_norm;

            if (!std::isfinite(verified_relative)) {
                throw std::runtime_error(
                    "GPU CG breakdown: invalid true residual"
                );
            }

            result.relative_residual = verified_relative;

            if (verified_relative <= options.relative_tolerance) {
                result.converged = true;
                d_x.download(result.x.data());
                return result;
            }

            // Recursive residual drift: restart using true residual.
            gpu_copy_vector(dr, dverified, n);
            gpu_copy_vector(dp, dverified, n);
            rr = verified_rr;
            continue;
        }

        const double beta = rr_new / rr;

        if (!std::isfinite(beta)) {
            throw std::runtime_error(
                "GPU CG breakdown: invalid beta"
            );
        }

        gpu_update_direction(dp, dr, beta, n);
        rr = rr_new;
    }

    // Report the true residual on iteration-limit exit.
    gpu_true_residual(
        row_ptr, col_idx, values,
        db, dx, dAp, dverified, n
    );

    const double final_rr =
        gpu_dot(dverified, dverified, n);

    result.relative_residual =
        std::sqrt(final_rr) / b_norm;

    if (!std::isfinite(result.relative_residual)) {
        throw std::runtime_error(
            "GPU CG breakdown: invalid final true residual"
        );
    }

    result.converged =
        result.relative_residual <= options.relative_tolerance;

    d_x.download(result.x.data());
    return result;
}

} // namespace cg
