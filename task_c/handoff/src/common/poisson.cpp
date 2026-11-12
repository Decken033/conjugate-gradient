
#include "poisson.hpp"

#include <limits>
#include <stdexcept>

namespace cg {
namespace {

// Ensure the grid dimensions do not overflow size_t.
std::size_t checked_multiply(
    std::size_t a,
    std::size_t b
) {
    if (b != 0 &&
        a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(
            "Poisson grid dimensions overflow size_t"
        );
    }
    return a * b;
}

void validate_grid_size(std::size_t n) {
    if (n == 0) {
        throw std::invalid_argument(
            "Poisson grid size must be at least 1"
        );
    }
}

} // namespace

/*
 * Construct a 2D five-point Poisson matrix in CSR format.
 *
 * Discretization of the negative Laplacian using
 * homogeneous Dirichlet boundary conditions:
 *
 *              -1
 *          -1   4   -1
 *              -1
 *
 * Reference:
 * Saad, Y. (2003). Iterative Methods for Sparse
 * Linear Systems, 2nd edition, SIAM.
 *
 * The resulting matrix is symmetric positive definite.
 */
CSRMatrix make_poisson_2d(std::size_t n) {
    validate_grid_size(n);

    const std::size_t N = checked_multiply(n, n);

    CSRMatrix A;
    A.rows = N;
    A.cols = N;

    A.row_ptr.reserve(N + 1);
    A.row_ptr.push_back(0);

    for (std::size_t y = 0; y < n; ++y) {
        for (std::size_t x = 0; x < n; ++x) {
            const std::size_t i = y * n + x;

            auto add = [&](std::size_t col, double value) {
                A.col_idx.push_back(col);
                A.values.push_back(value);
            };

            add(i, 4.0);

            if (x > 0)     add(i - 1, -1.0);
            if (x + 1 < n) add(i + 1, -1.0);
            if (y > 0)     add(i - n, -1.0);
            if (y + 1 < n) add(i + n, -1.0);

            A.row_ptr.push_back(A.values.size());
        }
    }

    A.validate();
    return A;
}

/*
 * Construct a 3D seven-point Poisson matrix in CSR format.
 *
 * The negative discrete Laplacian has:
 *   diagonal coefficient:  6
 *   six neighbor values:  -1
 *
 * Homogeneous Dirichlet boundary conditions are assumed.
 *
 * Reference:
 * Saad, Y. (2003). Iterative Methods for Sparse
 * Linear Systems, 2nd edition, SIAM.
 */
CSRMatrix make_poisson_3d(std::size_t n) {
    validate_grid_size(n);

    const std::size_t plane = checked_multiply(n, n);
    const std::size_t N = checked_multiply(plane, n);

    CSRMatrix A;
    A.rows = N;
    A.cols = N;

    A.row_ptr.reserve(N + 1);
    A.row_ptr.push_back(0);

    for (std::size_t z = 0; z < n; ++z) {
        for (std::size_t y = 0; y < n; ++y) {
            for (std::size_t x = 0; x < n; ++x) {
                const std::size_t i =
                    z * plane + y * n + x;

                auto add = [&](std::size_t col, double value) {
                    A.col_idx.push_back(col);
                    A.values.push_back(value);
                };

                add(i, 6.0);

                if (x > 0)     add(i - 1, -1.0);
                if (x + 1 < n) add(i + 1, -1.0);
                if (y > 0)     add(i - n, -1.0);
                if (y + 1 < n) add(i + n, -1.0);
                if (z > 0)     add(i - plane, -1.0);
                if (z + 1 < n) add(i + plane, -1.0);

                A.row_ptr.push_back(A.values.size());
            }
        }
    }

    A.validate();
    return A;
}

} // namespace cg
