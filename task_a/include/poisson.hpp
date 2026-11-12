
#pragma once

#include "csr_matrix.hpp"

#include <cstddef>

namespace cg {

// Generate the SPD matrix for the negative 2D Laplacian
// on an n-by-n interior grid with homogeneous Dirichlet
// boundary conditions.
//
// Uses the standard five-point finite-difference stencil:
// diagonal = 4, immediate neighbors = -1.
//
// Number of unknowns: n*n.
// Requires n >= 1.
CSRMatrix make_poisson_2d(std::size_t n);

// Generate the SPD matrix for the negative 3D Laplacian
// on an n-by-n-by-n interior grid with homogeneous
// Dirichlet boundary conditions.
//
// Uses the standard seven-point finite-difference stencil:
// diagonal = 6, immediate neighbors = -1.
//
// Number of unknowns: n*n*n.
// Requires n >= 1.
CSRMatrix make_poisson_3d(std::size_t n);

} // namespace cg
