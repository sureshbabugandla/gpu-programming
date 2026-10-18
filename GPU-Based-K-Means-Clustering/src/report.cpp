#include "report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>

namespace kmeans {
namespace {

// %.17g round-trips a binary64 exactly, which matters for t_alg_ms and tol: a summary that cannot
// reproduce the number it reports is not evidence of anything (PERF-011).
std::string num(double value)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", value);
    return std::string(buf);
}

// JSON has no literal for NaN or infinity. No field written here should ever hold one -- VAL-005
// rejects a non-finite tol and VAL-006 a non-finite coordinate -- so if one appears it is a fault
// to be seen, not hidden: it becomes null and the reader notices the missing number.
std::string json_num(double value)
{
    return std::isfinite(value) ? num(value) : "null";
}

std::string json_string(const std::string &s)
{
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                out += buf;
            } else {
                out += c;
            }
        }
    }
    out += "\"";
    return out;
}

// A CSV field is quoted only when it has to be, so the common case stays readable in a terminal.
std::string csv_field(const std::string &s)
{
    const bool needs_quotes = s.find_first_of(",\"\n\r") != std::string::npos;
    if (!needs_quotes) {
        return s;
    }
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

std::string hex64(uint64_t value)
{
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
    return std::string(buf);
}

// FNV-1a. Used only to shorten a configuration description into a row identifier; nothing
// depends on its collision behaviour beyond readability.
uint64_t fnv1a(const std::string &s)
{
    uint64_t h = 1469598103934665603ull;
    for (char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull;
    }
    return h;
}

const char *na_or(const std::string &s)
{
    return s.empty() ? "NA" : s.c_str();
}

}  // namespace

// The basename of the data file without its extension. The generator names its files
// <dataset_id>_X.kmd (FR-003), so this recovers the dataset identity for a benchmark row without
// the executable having to know the generator parameters.
std::string dataset_id_from_path(const std::string &path)
{
    size_t begin = path.find_last_of("/\\");
    begin = (begin == std::string::npos) ? 0 : begin + 1;
    size_t end = path.find_last_of('.');
    if (end == std::string::npos || end < begin) {
        end = path.size();
    }
    std::string stem = path.substr(begin, end - begin);
    // Drop the _X suffix the generator appends to the data file, so the data and centroid files
    // of one dataset report the same id.
    if (stem.size() > 2 && stem.compare(stem.size() - 2, 2, "_X") == 0) {
        stem.erase(stem.size() - 2);
    }
    return stem.empty() ? std::string("unknown") : stem;
}

Statistics statistics(std::vector<double> values)
{
    Statistics s;
    s.count = static_cast<int64_t>(values.size());
    if (values.empty()) {
        return s;
    }
    std::sort(values.begin(), values.end());
    s.min = values.front();
    s.max = values.back();

    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    s.mean = sum / static_cast<double>(values.size());

    const size_t mid = values.size() / 2;
    s.median = (values.size() % 2 == 1) ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);

    if (values.size() > 1) {
        double squares = 0.0;
        for (double v : values) {
            const double d = v - s.mean;
            squares += d * d;
        }
        s.stddev = std::sqrt(squares / static_cast<double>(values.size() - 1));
    }
    s.cv = (s.mean != 0.0) ? s.stddev / s.mean : 0.0;
    return s;
}

std::string utc_timestamp()
{
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return std::string(buf);
}

std::string make_run_id(const RunReport &report)
{
    const std::string key =
        report.dataset_id + "|" + std::to_string(report.config.N) + "x" +
        std::to_string(report.config.D) + "k" + std::to_string(report.config.K) + "|" +
        strategy_option_name(report.config.strategy) + "|" +
        assign_option_name(report.config.assign_kernel) + "|b" +
        std::to_string(report.config.block_size) + "|g" +
        std::to_string(report.reduce_chunks_g) + "|i" + std::to_string(report.config.max_iter) +
        "|t" + num(report.config.tol) + "|" + (report.config.fixed_iters ? "fixed" : "conv") +
        "|" + report.timestamp_utc + "|" + report.env.git_commit;
    return hex64(fnv1a(key)) + "-r" + std::to_string(report.rep_index);
}

std::string summary_json(const RunReport &report)
{
    const Config &c = report.config;
    std::string out = "{\n";

    out += "  \"run_id\": " + json_string(report.run_id) + ",\n";
    out += "  \"timestamp_utc\": " + json_string(report.timestamp_utc) + ",\n";
    out += "  \"dataset_id\": " + json_string(report.dataset_id) + ",\n";
    out += "  \"data_path\": " + json_string(report.data_path) + ",\n";
    out += "  \"centroids_path\": " + json_string(report.centroids_path) + ",\n";

    // Section 2.7 outcome, FR-015.
    out += "  \"status\": " + json_string(status_name(report.status)) + ",\n";
    out += "  \"converged\": " + std::string(is_converged(report.status) ? "true" : "false") +
           ",\n";
    out += "  \"iterations\": " + std::to_string(report.iterations) + ",\n";
    // FR-015: null, not 0, when no update was performed (t == 0). The two are different claims.
    out += "  \"last_delta\": " +
           (report.has_last_delta ? json_num(report.last_delta) : std::string("null")) + ",\n";
    out += "  \"centroid_consistent\": " +
           std::string(report.centroid_consistent ? "true" : "false") + ",\n";
    out += "  \"assignment_consistent\": " +
           std::string(report.assignment_consistent ? "true" : "false") + ",\n";
    out += "  \"objective\": " + json_num(report.objective) + ",\n";
    out += "  \"correctness_status\": " + json_string(report.correctness_status) + ",\n";

    out += "  \"configuration\": {\n";
    out += "    \"N\": " + std::to_string(c.N) + ",\n";
    out += "    \"D\": " + std::to_string(c.D) + ",\n";
    out += "    \"K\": " + std::to_string(c.K) + ",\n";
    out += "    \"max_iter\": " + std::to_string(c.max_iter) + ",\n";
    out += "    \"tolerance\": " + json_num(c.tol) + ",\n";
    out += "    \"fixed_iters\": " + std::string(c.fixed_iters ? "true" : "false") + ",\n";
    out += "    \"strategy\": " + json_string(strategy_option_name(c.strategy)) + ",\n";
    out += "    \"assign_kernel\": " + json_string(assign_option_name(c.assign_kernel)) + ",\n";
    out += "    \"accum_precision\": \"fp64\",\n";
    out += "    \"block_size\": " + std::to_string(c.block_size) + ",\n";
    out += "    \"tile_k\": " + std::to_string(report.tile_k) + ",\n";
    out += "    \"tile_d\": " + std::to_string(report.tile_d) + ",\n";
    // RED-006: G exists only for Approach B, where 0 would be a wrong number rather than a
    // missing one.
    out += "    \"reduce_chunks_G\": " +
           (report.reduce_chunks_g > 0 ? std::to_string(report.reduce_chunks_g)
                                       : std::string("null")) + ",\n";
    out += "    \"verify\": " + json_string(verify_name(report.verify)) + ",\n";
    out += "    \"profile_phases\": " + std::string(report.profile_phases ? "true" : "false") +
           "\n";
    out += "  },\n";

    // PERF-005 timings.
    out += "  \"timing_ms\": {\n";
    out += "    \"t_alg\": " + json_num(report.t_alg_ms) + ",\n";
    out += "    \"t_e2e\": " + json_num(report.t_e2e_ms) + ",\n";
    out += "    \"t_alloc\": " + json_num(report.t_alloc_ms) + ",\n";
    out += "    \"t_h2d\": " + json_num(report.t_h2d_ms) + ",\n";
    out += "    \"t_d2h\": " + json_num(report.t_d2h_ms) + ",\n";
    out += "    \"valid_as_benchmark_sample\": " +
           std::string(report.timing_valid ? "true" : "false") + "\n";
    out += "  },\n";

    // PERF-007 phase sums, null unless --profile-phases measured them. They cover the T_alg
    // region; the remainder is the per-iteration 16-byte transfer and the host decision gap,
    // which belong to no phase.
    out += "  \"phase_ms\": {\n";
    const bool ph = report.has_phases;
    out += "    \"t_assign\": " + (ph ? json_num(report.t_assign_ms) : std::string("null")) +
           ",\n";
    out += "    \"t_accum\": " + (ph ? json_num(report.t_accum_ms) : std::string("null")) + ",\n";
    out += "    \"t_update\": " + (ph ? json_num(report.t_update_ms) : std::string("null")) +
           ",\n";
    out += "    \"t_conv\": " + (ph ? json_num(report.t_conv_ms) : std::string("null")) + "\n";
    out += "  },\n";

    // PERF-008 environment record.
    out += "  \"environment\": {\n";
    out += "    \"gpu_name\": " + json_string(report.env.gpu_name) + ",\n";
    out += "    \"compute_capability\": " + json_string(report.env.compute_capability) + ",\n";
    out += "    \"sm_count\": " + std::to_string(report.env.sm_count) + ",\n";
    out += "    \"driver_version\": " + std::to_string(report.env.driver_version) + ",\n";
    out += "    \"cuda_runtime\": " + std::to_string(report.env.cuda_runtime) + ",\n";
    out += "    \"nvcc_version\": " + json_string(report.env.nvcc_version) + ",\n";
    out += "    \"host_compiler\": " + json_string(report.env.host_compiler) + ",\n";
    out += "    \"os\": " + json_string(report.env.os) + ",\n";
    out += "    \"build_flags\": " + json_string(report.env.build_flags) + ",\n";
    out += "    \"git_commit\": " + json_string(report.env.git_commit) + "\n";
    out += "  },\n";

    // FR-015 third output: the final counts. Sum_k n_k == N (FR-009) is checkable from here.
    out += "  \"counts\": [";
    for (size_t k = 0; k < report.counts.size(); ++k) {
        out += (k == 0 ? "" : ", ") + std::to_string(report.counts[k]);
    }
    out += "]\n";

    out += "}\n";
    return out;
}

std::string csv_header()
{
    return "run_id,timestamp_utc,git_commit,dataset_id,generator_seed,N,D,K,strategy,"
           "assign_kernel,accum_precision,block_size,tile_k,tile_d,reduce_chunks_G,max_iter,"
           "tolerance,fixed_iters,rep_index,iterations,status,centroid_consistent,"
           "assignment_consistent,t_alg_ms,t_e2e_ms,t_alloc_ms,t_h2d_ms,t_d2h_ms,t_assign_ms,"
           "t_accum_ms,t_update_ms,t_conv_ms,objective_J,correctness_status,gpu_name,"
           "compute_capability,sm_count,driver_version,cuda_runtime,nvcc_version,host_compiler,"
           "os,build_flags\n";
}

std::string csv_row(const RunReport &report)
{
    const Config &c = report.config;
    std::string r;
    r += csv_field(report.run_id) + ",";
    r += csv_field(report.timestamp_utc) + ",";
    r += csv_field(na_or(report.env.git_commit)) + ",";
    r += csv_field(report.dataset_id) + ",";
    // The executable is handed files, not generator parameters, so the seed is part of the
    // dataset id rather than a column it can fill (FR-003).
    r += "NA,";
    r += std::to_string(c.N) + ",";
    r += std::to_string(c.D) + ",";
    r += std::to_string(c.K) + ",";
    r += std::string(strategy_option_name(c.strategy)) + ",";
    r += std::string(assign_option_name(c.assign_kernel)) + ",";
    r += "fp64,";
    r += std::to_string(c.block_size) + ",";
    // The tiles exist only for the smem kernel; NA where they describe nothing.
    if (c.assign_kernel == AssignKernel::kSharedMemory) {
        r += std::to_string(report.tile_k) + "," + std::to_string(report.tile_d) + ",";
    } else {
        r += "NA,NA,";
    }
    r += (report.reduce_chunks_g > 0 ? std::to_string(report.reduce_chunks_g)
                                     : std::string("NA")) + ",";
    r += std::to_string(c.max_iter) + ",";
    r += num(c.tol) + ",";
    r += std::string(c.fixed_iters ? "1" : "0") + ",";
    r += std::to_string(report.rep_index) + ",";
    r += std::to_string(report.iterations) + ",";
    r += std::string(status_name(report.status)) + ",";
    r += std::string(report.centroid_consistent ? "1" : "0") + ",";
    r += std::string(report.assignment_consistent ? "1" : "0") + ",";
    r += num(report.t_alg_ms) + ",";
    r += num(report.t_e2e_ms) + ",";
    r += num(report.t_alloc_ms) + ",";
    r += num(report.t_h2d_ms) + ",";
    r += num(report.t_d2h_ms) + ",";
    // PERF-007 phase columns. Phase timing is a separate mode and is not mixed with headline
    // numbers, so the columns stay NA unless --profile-phases measured them.
    if (report.has_phases) {
        r += num(report.t_assign_ms) + "," + num(report.t_accum_ms) + "," +
             num(report.t_update_ms) + "," + num(report.t_conv_ms) + ",";
    } else {
        r += "NA,NA,NA,NA,";
    }
    r += num(report.objective) + ",";
    r += csv_field(report.correctness_status) + ",";
    r += csv_field(report.env.gpu_name) + ",";
    r += csv_field(report.env.compute_capability) + ",";
    r += std::to_string(report.env.sm_count) + ",";
    r += std::to_string(report.env.driver_version) + ",";
    r += std::to_string(report.env.cuda_runtime) + ",";
    r += csv_field(report.env.nvcc_version) + ",";
    r += csv_field(report.env.host_compiler) + ",";
    r += csv_field(report.env.os) + ",";
    r += csv_field(report.env.build_flags);
    r += "\n";
    return r;
}

int write_text_file(const std::string &path, const std::string &content, std::string *err)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        *err = "cannot open '" + path + "' for writing";
        return kExitIoError;
    }
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.close();
    if (!out) {
        *err = "writing '" + path + "' failed";
        return kExitIoError;
    }
    return kExitOk;
}

int append_csv_row(const std::string &path, const RunReport &report, std::string *err)
{
    bool needs_header = true;
    {
        // ate puts the get pointer at the end, so tellg is the size: an existing non-empty file
        // already has its header and a new or empty one does not.
        std::ifstream probe(path, std::ios::binary | std::ios::ate);
        if (probe && probe.tellg() > 0) {
            needs_header = false;
        }
    }

    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) {
        *err = "cannot open '" + path + "' for appending";
        return kExitIoError;
    }
    if (needs_header) {
        const std::string header = csv_header();
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
    }
    const std::string row = csv_row(report);
    out.write(row.data(), static_cast<std::streamsize>(row.size()));
    out.close();
    if (!out) {
        *err = "writing '" + path + "' failed";
        return kExitIoError;
    }
    return kExitOk;
}

}  // namespace kmeans
