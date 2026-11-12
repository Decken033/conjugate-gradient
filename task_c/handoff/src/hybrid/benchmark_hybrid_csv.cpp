#include "hybrid/hybrid_benchmark.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace cg {
namespace {
std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    return out + '"';
}

const char* schedule_name(OpenMPSchedule schedule) {
    switch (schedule) {
        case OpenMPSchedule::Static: return "static";
        case OpenMPSchedule::Dynamic: return "dynamic";
        case OpenMPSchedule::Guided: return "guided";
        default: throw std::invalid_argument("Hybrid CSV: unsupported schedule");
    }
}
} // namespace

void write_hybrid_benchmark_csv(
    const std::vector<HybridBenchmarkRecord>& records,
    const std::string& output_path,
    int rank
) {
    if (rank != 0) return;
    if (output_path.empty()) throw std::invalid_argument("Hybrid CSV: output path is empty");
    std::ofstream out(output_path, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("Hybrid CSV: cannot open " + output_path);

    out << "solver,matrix_name,n,nnz,ranks,threads,run_index,iterations,converged,solve_seconds,"
           "true_relative_residual,relative_solution_error,schedule,chunk_size\n";
    out << std::setprecision(17);
    for (const auto& r : records) {
        out << "hybrid," << csv_escape(r.matrix_name) << ',' << r.n << ',' << r.nnz << ','
            << r.ranks << ',' << r.threads << ',' << r.run_index << ',' << r.iterations << ','
            << (r.converged ? 1 : 0) << ',' << r.solve_seconds << ','
            << r.true_relative_residual << ',' << r.relative_solution_error << ','
            << schedule_name(r.schedule) << ',' << r.chunk_size << '\n';
    }
    out.flush();
    if (!out) throw std::runtime_error("Hybrid CSV: failed writing " + output_path);
}

} // namespace cg
