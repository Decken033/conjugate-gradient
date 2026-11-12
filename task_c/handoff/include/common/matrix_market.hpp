
#pragma once

#include "common/csr_matrix.hpp"

#include <string>

namespace cg {

// Load a real-valued Matrix Market coordinate matrix
// and convert it into the shared CSR representation.
//
// Supported storage:
//   - coordinate real general
//   - coordinate real symmetric
//
// Matrix Market indices are converted from 1-based to 0-based.
//
// For symmetric matrices, off-diagonal entries are
// expanded to reconstruct the full matrix.
//
// Duplicate coordinates are combined by summing values.
//
// Throws std::runtime_error for invalid or unsupported
// Matrix Market input.
//
// The returned CSR matrix is validated before use.
CSRMatrix load_matrix_market(const std::string& path);

}  // namespace cg
