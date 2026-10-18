// report.h — the FR-015 run summary and the section 13.1 benchmark CSV.
//
// One record, two renderings: a JSON object per run (BLD-004: "Run summary: JSON with the fields
// of FR-015") and one CSV row per measured repetition. Keeping them in one struct is what makes
// TEST-I08 checkable -- the notebook parses the JSON and the CSV and compares both against the
// values the executable printed, and a field that exists in only one of the two renderings would
// make that comparison meaningless.
//
// Free of CUDA constructs, so the JSON and CSV writers are compiled and tested with g++ on a
// machine with no GPU (CLAUDE.md). Everything the record knows about the device arrives as a
// plain string or integer from the caller, which is the .cu entry point.

#ifndef KMEANS_REPORT_H
#define KMEANS_REPORT_H

#include <cstdint>
#include <string>
#include <vector>

#include "cli_options.h"
#include "common.h"

namespace kmeans {

// PERF-008: the build and environment identity every CSV row carries, so a measurement can be
// traced back to the GPU, the toolchain and the source revision that produced it.
struct BuildEnvironment {
    std::string gpu_name;
    std::string compute_capability;  // "7.5"
    int sm_count = 0;
    int driver_version = 0;  // as cudaDriverGetVersion reports it
    int cuda_runtime = 0;    // as cudaRuntimeGetVersion reports it
    std::string nvcc_version;
    std::string host_compiler;
    std::string os;
    std::string build_flags;
    std::string git_commit;  // or the notebook run identity; "unknown" when not supplied
};

// Everything FR-015 asks a run to report, plus the section 13.1 columns.
struct RunReport {
    BuildEnvironment env;

    std::string run_id;        // unique per CSV row
    std::string timestamp_utc; // ISO-8601
    std::string dataset_id;    // the stem of the data file, which encodes the generator parameters
    std::string data_path;
    std::string centroids_path;

    Config config;             // N, D, K, max_iter, tol, fixed_iters, strategy, assign, block, G
    int tile_k = 0;            // SMEM-001 compile-time tiles, reported as built
    int tile_d = 0;
    int64_t reduce_chunks_g = 0;  // the selected G; 0 is reported as NA for Approach A
    VerifyMode verify = VerifyMode::kNone;
    bool profile_phases = false;

    int64_t rep_index = 0;  // 0 for a single unmeasured run, 1..R for a repetition series
    int64_t iterations = 0;
    Status status = Status::ERROR;
    bool centroid_consistent = false;
    bool assignment_consistent = true;
    double last_delta = 0.0;
    bool has_last_delta = false;  // false when t == 0, where FR-015 requires null

    double t_alg_ms = 0.0;
    double t_e2e_ms = 0.0;
    double t_alloc_ms = 0.0;
    double t_h2d_ms = 0.0;
    double t_d2h_ms = 0.0;
    bool timing_valid = true;  // FR-019: false for a lockstep run

    double objective = 0.0;

    // PERF-001: PASS or FAIL from --verify, NOT_CHECKED without it.
    std::string correctness_status = "NOT_CHECKED";

    std::vector<uint32_t> counts;  // n_final, the third FR-015 output
};

// PERF-006 statistics over the measured repetitions. Median is the primary figure.
struct Statistics {
    int64_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double stddev = 0.0;  // sample standard deviation, divisor count - 1
    double min = 0.0;
    double max = 0.0;
    double cv = 0.0;  // coefficient of variation, stddev / mean (TEST-P04)
};

// `values` is copied and sorted internally, so the caller's order is preserved.
Statistics statistics(std::vector<double> values);

// ISO-8601 UTC, e.g. "2026-10-18T07:14:22Z".
std::string utc_timestamp();

// Dataset identity from the data file's path: the basename without directory, extension or the
// `_X` suffix the generator appends. The generator names its files `<dataset_id>_X.kmd` (FR-003),
// so a benchmark row is traceable to its generator parameters without the executable having to
// know them.
std::string dataset_id_from_path(const std::string &path);

// A run identifier that is unique per row and reproducible from the row's own fields: the
// configuration and timestamp are hashed, so two rows of the same repetition series differ only
// in the repetition index.
std::string make_run_id(const RunReport &report);

// The FR-015 summary. Pretty-printed with one field per line because a human reads it in the
// notebook output as often as a parser does.
std::string summary_json(const RunReport &report);

// The section 13.1 header, and one raw row. `append_csv_row` writes the header first if the file
// does not yet exist, so a benchmark sweep can append without a separate setup step.
std::string csv_header();
std::string csv_row(const RunReport &report);

// Both return kExitOk or kExitIoError with a BLD-006 style message in `err`.
int write_text_file(const std::string &path, const std::string &content, std::string *err);
int append_csv_row(const std::string &path, const RunReport &report, std::string *err);

}  // namespace kmeans

#endif  // KMEANS_REPORT_H
