// cli_options.h — the BLD-003a command line of the `kmeans` executable.
//
//   kmeans --data X.kmd --centroids C0.kmd --k K --max-iter I --tol EPS
//          --strategy {atomic|reduction} [--assign {baseline|smem}]
//          [--block-size B] [--reduce-chunks G] [--fixed-iters]
//          [--verify {none|final|lockstep}] [--profile-phases]
//          [--out-prefix PATH] [--reps R --warmup W --csv FILE]
//
// Deliberately free of CUDA constructs, like validate.cpp and launch_config.h: argument parsing
// is where a program of this shape is easiest to get wrong and hardest to debug on a remote
// runtime, so it is compiled and tested with g++ on a machine with no GPU (CLAUDE.md). The .cu
// entry point turns the parsed result into kmeans::gpu::Options and never looks at argv itself.
//
// N and D are not options: they come from the header of the data file (FR-002). K is an option
// because the initial-centroid file is checked against it (VAL-008).

#ifndef KMEANS_CLI_OPTIONS_H
#define KMEANS_CLI_OPTIONS_H

#include <cstdint>
#include <string>

#include "common.h"

namespace kmeans {

// FR-019 verification modes, mirrored here so this header stays CUDA-free. kmeans_main maps the
// value onto kmeans::gpu::Verify, and the two enumerations are kept in the same order.
enum class VerifyMode { kNone, kFinal, kLockstep };

inline const char *verify_name(VerifyMode v)
{
    switch (v) {
    case VerifyMode::kNone:     return "none";
    case VerifyMode::kFinal:    return "final";
    case VerifyMode::kLockstep: return "lockstep";
    }
    return "unknown";
}

inline const char *strategy_option_name(Strategy s)
{
    return s == Strategy::kAtomic ? "atomic" : "reduction";
}

inline const char *assign_option_name(AssignKernel a)
{
    return a == AssignKernel::kBaseline ? "baseline" : "smem";
}

struct CliOptions {
    std::string data_path;       // --data, required
    std::string centroids_path;  // --centroids, required
    int64_t K = 0;               // --k, required
    int64_t max_iter = 0;        // --max-iter, required (0 is legal, VAL-004)
    double tol = 0.0;            // --tol, required

    Strategy strategy = Strategy::kAtomic;                   // --strategy, required
    AssignKernel assign_kernel = AssignKernel::kBaseline;    // --assign
    int block_size = 256;                                    // --block-size (CUDA-007 default)
    int64_t reduce_chunks = 0;                               // --reduce-chunks, 0 = auto (RED-006)
    bool fixed_iters = false;                                // --fixed-iters (FR-020)
    VerifyMode verify = VerifyMode::kNone;                   // --verify (FR-019)
    bool profile_phases = false;                             // --profile-phases (PERF-007)

    std::string out_prefix;  // --out-prefix; no outputs are written when empty (FR-015)
    std::string csv_path;    // --csv; the section 13.1 raw file
    int64_t reps = 0;        // --reps; 0 means one unmeasured run, not a repetition series
    int64_t warmup = 3;      // --warmup; the PERF-006 default

    // Config for the clustering run, once N and D are known from the data file (FR-002). The
    // tile sizes are not part of Config: SMEM-001 makes them compile-time constants.
    Config to_config(int64_t N, int64_t D) const;
};

struct CliParse {
    bool ok = true;
    std::string rule;     // the rule ID for the BLD-006 prefix, empty when ok
    std::string message;  // cause and offending value
    int exit_code = kExitOk;
    bool usage = false;  // --help: print usage and exit 0 without running
};

// Parses argv[1..argc) into `out`. Rejects unknown options, missing values, unparsable numbers,
// unknown enumeration values and missing required options with exit code 2 (BLD-005) and a
// message naming the rule (BLD-006). The value ranges that VAL-004, VAL-005 and VAL-010 own are
// checked by validate_parameters and validate_options, not here, so there is exactly one place
// per rule; what this function owns is the syntax of the command line.
CliParse parse_cli(int argc, const char *const *argv, CliOptions *out);

// The BLD-003a usage text, printed on a parse failure and by --help.
std::string cli_usage();

}  // namespace kmeans

#endif  // KMEANS_CLI_OPTIONS_H
