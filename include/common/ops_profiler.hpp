#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace cg {

enum class ProfileSection : std::size_t {
    SpmvAllgatherv = 0,
    SpmvCompute,
    DotCompute,
    Allreduce,
    SolutionResidualUpdate,
    DirectionUpdate,
    ResidualSubtract,
    CollectiveCheck,
    SolverTotal,
    WorkspaceInit,
    Count
};

inline constexpr std::size_t profile_section_count =
    static_cast<std::size_t>(ProfileSection::Count);

inline const char* profile_section_name(ProfileSection s) {
    switch (s) {
        case ProfileSection::SpmvAllgatherv: return "spmv_allgatherv";
        case ProfileSection::SpmvCompute: return "spmv_compute";
        case ProfileSection::DotCompute: return "dot_compute";
        case ProfileSection::Allreduce: return "mpi_allreduce";
        case ProfileSection::SolutionResidualUpdate: return "solution_residual_update";
        case ProfileSection::DirectionUpdate: return "direction_update";
        case ProfileSection::ResidualSubtract: return "true_residual_subtract";
        case ProfileSection::CollectiveCheck: return "collective_check";
        case ProfileSection::SolverTotal: return "solver_total";
        case ProfileSection::WorkspaceInit: return "workspace_init";
        default: return "unknown";
    }
}

struct OpsProfile {
    std::array<double, profile_section_count> seconds{};
    std::array<unsigned long long, profile_section_count> calls{};

    void reset() { seconds.fill(0.0); calls.fill(0); }
    void add(ProfileSection section, double elapsed) {
        const auto i = static_cast<std::size_t>(section);
        seconds[i] += elapsed;
        calls[i] += 1;
    }
};

inline thread_local OpsProfile* active_ops_profile = nullptr;
inline thread_local OpsProfile process_ops_profile{};

class ProfileSession {
public:
    explicit ProfileSession(OpsProfile& profile)
        : previous_(active_ops_profile) { active_ops_profile = &profile; }
    ~ProfileSession() { active_ops_profile = previous_; }
    ProfileSession(const ProfileSession&) = delete;
    ProfileSession& operator=(const ProfileSession&) = delete;
private:
    OpsProfile* previous_;
};

class ProfileScope {
public:
    explicit ProfileScope(ProfileSection section)
        : profile_(active_ops_profile), section_(section), start_(Clock::now()) {}
    ~ProfileScope() {
        if (profile_) {
            const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
            profile_->add(section_, elapsed);
        }
    }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
private:
    using Clock = std::chrono::steady_clock;
    OpsProfile* profile_;
    ProfileSection section_;
    Clock::time_point start_;
};

inline void write_ops_profile_csv(const OpsProfile& profile,
                                  const std::string& path,
                                  const std::string& solver,
                                  int rank,
                                  int ranks,
                                  int threads) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot write profiling CSV: " + path);
    out << "solver,rank,ranks,threads,operation,calls,total_seconds,avg_seconds\n";
    out << std::setprecision(17);
    double accounted = 0.0;
    for (std::size_t i = 0; i < profile_section_count; ++i) {
        const auto calls = profile.calls[i];
        const double total = profile.seconds[i];
        if (i != static_cast<std::size_t>(ProfileSection::SolverTotal)) accounted += total;
        out << solver << ',' << rank << ',' << ranks << ',' << threads << ','
            << profile_section_name(static_cast<ProfileSection>(i)) << ','
            << calls << ',' << total << ','
            << (calls ? total / static_cast<double>(calls) : 0.0) << '\n';
    }
    const double solver_total = profile.seconds[static_cast<std::size_t>(ProfileSection::SolverTotal)];
    const double other = solver_total > accounted ? solver_total - accounted : 0.0;
    out << solver << ',' << rank << ',' << ranks << ',' << threads
        << ",other_uninstrumented,1," << other << ',' << other << '\n';
}

} // namespace cg
