
#include "matrix_market.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg {

namespace {

struct Entry {
    std::size_t row;
    std::size_t col;
    double value;
};

bool entry_less(const Entry& a, const Entry& b) {
    if (a.row != b.row) {
        return a.row < b.row;
    }
    return a.col < b.col;
}

bool is_blank(const std::string& line) {
    return line.find_first_not_of(" \t\r\n") == std::string::npos;
}

}  // namespace

CSRMatrix load_matrix_market(const std::string& path) {
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            "Cannot open Matrix Market file: " + path);
    }

    // --------------------------------------------------
    // 1. Read and validate Matrix Market header.
    // --------------------------------------------------
    std::string line;

    if (!std::getline(input, line)) {
        throw std::runtime_error("Empty Matrix Market file");
    }

    std::istringstream header(line);

    std::string magic;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    std::string extra;

    if (!(header >> magic >> object >> format
                 >> field >> symmetry) ||
        (header >> extra)) {
        throw std::runtime_error(
            "Invalid Matrix Market header");
    }

    if (magic != "%%MatrixMarket" ||
        object != "matrix" ||
        format != "coordinate" ||
        field != "real" ||
        (symmetry != "general" &&
         symmetry != "symmetric")) {
        throw std::runtime_error(
            "Unsupported Matrix Market format");
    }

    const bool symmetric = (symmetry == "symmetric");

    // --------------------------------------------------
    // 2. Skip comments and read matrix dimensions.
    // --------------------------------------------------
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t stored_entries = 0;
    bool dimensions_found = false;

    while (std::getline(input, line)) {
        if (is_blank(line) || line[0] == '%') {
            continue;
        }

        std::istringstream dimensions(line);

        if (!(dimensions >> rows >> cols >> stored_entries) ||
            (dimensions >> extra)) {
            throw std::runtime_error(
                "Invalid Matrix Market dimensions");
        }

        dimensions_found = true;
        break;
    }

    if (!dimensions_found || rows == 0 || cols == 0) {
        throw std::runtime_error(
            "Missing or invalid matrix dimensions");
    }

    if (rows != cols) {
        throw std::runtime_error(
            "CG requires a square matrix");
    }

    // --------------------------------------------------
    // 3. Read coordinate entries.
    //
    // Matrix Market indices start at 1.
    // CSR indices start at 0.
    //
    // For symmetric matrices, mirror off-diagonal
    // entries to reconstruct the full matrix.
    // --------------------------------------------------
    std::vector<Entry> entries;

    if (stored_entries > entries.max_size() /
                             (symmetric ? 2 : 1)) {
        throw std::runtime_error(
            "Matrix Market entry count is too large");
    }

    entries.reserve(
        stored_entries * (symmetric ? 2 : 1));

    std::size_t entries_read = 0;

    while (entries_read < stored_entries &&
           std::getline(input, line)) {
        if (is_blank(line) || line[0] == '%') {
            continue;
        }

        std::istringstream data(line);

        std::size_t row = 0;
        std::size_t col = 0;
        double value = 0.0;

        if (!(data >> row >> col >> value) ||
            (data >> extra)) {
            throw std::runtime_error(
                "Invalid Matrix Market coordinate entry");
        }

        if (row == 0 || row > rows ||
            col == 0 || col > cols ||
            !std::isfinite(value)) {
            throw std::runtime_error(
                "Matrix Market entry out of range "
                "or non-finite");
        }

        --row;
        --col;

        entries.push_back({row, col, value});

        if (symmetric && row != col) {
            entries.push_back({col, row, value});
        }

        ++entries_read;
    }

    if (entries_read != stored_entries) {
        throw std::runtime_error(
            "Matrix Market entry count mismatch");
    }

    // --------------------------------------------------
    // 4. Sort entries into CSR row/column order.
    // --------------------------------------------------
    std::sort(entries.begin(), entries.end(), entry_less);

    CSRMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    matrix.row_ptr.assign(rows + 1, 0);

    matrix.col_idx.reserve(entries.size());
    matrix.values.reserve(entries.size());

    // --------------------------------------------------
    // 5. Combine duplicate coordinates.
    //
    // Duplicate entries are summed. This ensures
    // that each CSR (row, column) coordinate appears
    // only once.
    // --------------------------------------------------
    bool have_previous = false;
    std::size_t previous_row = 0;
    std::size_t previous_col = 0;

    for (const Entry& entry : entries) {
        if (have_previous &&
            entry.row == previous_row &&
            entry.col == previous_col) {

            matrix.values.back() += entry.value;

            if (!std::isfinite(matrix.values.back())) {
                throw std::runtime_error(
                    "Non-finite value after duplicate summation");
            }

            continue;
        }

        matrix.col_idx.push_back(entry.col);
        matrix.values.push_back(entry.value);

        ++matrix.row_ptr[entry.row + 1];

        previous_row = entry.row;
        previous_col = entry.col;
        have_previous = true;
    }

    // --------------------------------------------------
    // 6. Convert row counts into CSR row offsets.
    // --------------------------------------------------
    for (std::size_t i = 0; i < rows; ++i) {
        matrix.row_ptr[i + 1] += matrix.row_ptr[i];
    }

    // --------------------------------------------------
    // 7. Validate CSR structure before returning.
    // --------------------------------------------------
    matrix.validate();

    return matrix;
}

}  // namespace cg
