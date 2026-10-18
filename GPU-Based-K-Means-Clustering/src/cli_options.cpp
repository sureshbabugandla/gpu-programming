#include "cli_options.h"

#include <cstdlib>
#include <cstring>

namespace kmeans {
namespace {

CliParse fail(const char *rule, const std::string &message)
{
    CliParse p;
    p.ok = false;
    p.rule = rule;
    p.message = message;
    p.exit_code = kExitInvalidInput;
    return p;
}

// strtoll/strtod with the whole token required to be consumed. "20x", "" and "1 2" are errors
// rather than silently truncated values, which is the difference between a typo being caught and
// a benchmark row being produced for a configuration nobody asked for.
bool parse_int64(const char *text, int64_t *out)
{
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char *end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    *out = static_cast<int64_t>(value);
    return true;
}

bool parse_double(const char *text, double *out)
{
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

}  // namespace

Config CliOptions::to_config(int64_t N, int64_t D) const
{
    Config c;
    c.N = N;
    c.D = D;
    c.K = K;
    c.max_iter = max_iter;
    c.tol = tol;
    c.fixed_iters = fixed_iters;
    c.strategy = strategy;
    c.assign_kernel = assign_kernel;
    c.block_size = block_size;
    c.reduce_chunks = reduce_chunks;
    return c;
}

std::string cli_usage()
{
    return "usage: kmeans --data X.kmd --centroids C0.kmd --k K --max-iter I --tol EPS\n"
           "              --strategy {atomic|reduction} [--assign {baseline|smem}]\n"
           "              [--block-size B] [--reduce-chunks G] [--fixed-iters]\n"
           "              [--verify {none|final|lockstep}] [--profile-phases]\n"
           "              [--out-prefix PATH] [--reps R --warmup W --csv FILE]\n"
           "\n"
           "  --data, --centroids   binary .kmd input files (BLD-004); N and D come from the\n"
           "                        data header, and the centroid file must have K rows\n"
           "  --k                   number of clusters; K > N is accepted with a warning\n"
           "  --max-iter            iteration limit; 0 runs the initial assignment only\n"
           "  --tol                 epsilon of the movement criterion; 0 demands bit-identical\n"
           "                        centroids\n"
           "  --strategy            accumulation strategy (FR-018)\n"
           "  --assign              assignment kernel: baseline or the tiled smem one (SMEM-001)\n"
           "  --block-size          multiple of 32 in [32, 1024] (CUDA-007)\n"
           "  --reduce-chunks       Approach B chunk count G; omit or 0 to choose it (RED-006)\n"
           "  --fixed-iters         disable both convergence criteria (FR-020)\n"
           "  --verify              final: check P1 and P2 after the run; lockstep: check every\n"
           "                        iteration against the CPU reference (FR-019)\n"
           "  --profile-phases      per-phase timings (PERF-007)\n"
           "  --out-prefix          write PREFIX_A.kma, PREFIX_C.kmd and PREFIX_summary.json\n"
           "  --reps, --warmup      measured repetitions and warm-up runs (PERF-006)\n"
           "  --csv                 append one raw row per measured repetition (section 13.1)\n"
           "\n"
           "exit codes: 0 success, 2 invalid input, 3 unsupported device or insufficient\n"
           "resources, 4 CUDA runtime error, 5 I/O error, 6 verification failure (BLD-005)\n";
}

CliParse parse_cli(int argc, const char *const *argv, CliOptions *out)
{
    CliOptions o;
    bool have_k = false, have_max_iter = false, have_tol = false, have_strategy = false;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            CliParse p;
            p.usage = true;
            return p;
        }
        if (std::strcmp(arg, "--fixed-iters") == 0) {
            o.fixed_iters = true;
            continue;
        }
        if (std::strcmp(arg, "--profile-phases") == 0) {
            o.profile_phases = true;
            continue;
        }

        // Every remaining option takes exactly one value. Checking that here means the branches
        // below never read past the end of argv.
        if (std::strncmp(arg, "--", 2) != 0) {
            return fail("BLD-003a", "unexpected argument '" + std::string(arg) +
                                        "'; every input is named by an option");
        }
        if (i + 1 >= argc) {
            return fail("BLD-003a", "option " + std::string(arg) + " needs a value");
        }
        const char *value = argv[++i];

        if (std::strcmp(arg, "--data") == 0) {
            o.data_path = value;
        } else if (std::strcmp(arg, "--centroids") == 0) {
            o.centroids_path = value;
        } else if (std::strcmp(arg, "--k") == 0) {
            if (!parse_int64(value, &o.K)) {
                return fail("VAL-001", "--k expects an integer, got '" + std::string(value) + "'");
            }
            have_k = true;
        } else if (std::strcmp(arg, "--max-iter") == 0) {
            if (!parse_int64(value, &o.max_iter)) {
                return fail("VAL-004",
                            "--max-iter expects an integer, got '" + std::string(value) + "'");
            }
            have_max_iter = true;
        } else if (std::strcmp(arg, "--tol") == 0) {
            // VAL-005 distinguishes "unparsable" from "parsed but not finite and non-negative".
            // This is the unparsable half; validate_parameters owns the other half, so "nan" and
            // "-1" get past here and are rejected there with their own message.
            if (!parse_double(value, &o.tol)) {
                return fail("VAL-005",
                            "--tol expects a number, got '" + std::string(value) + "'");
            }
            have_tol = true;
        } else if (std::strcmp(arg, "--strategy") == 0) {
            if (std::strcmp(value, "atomic") == 0) {
                o.strategy = Strategy::kAtomic;
            } else if (std::strcmp(value, "reduction") == 0) {
                o.strategy = Strategy::kReduction;
            } else {
                return fail("BLD-003a", "--strategy expects atomic or reduction, got '" +
                                            std::string(value) + "'");
            }
            have_strategy = true;
        } else if (std::strcmp(arg, "--assign") == 0) {
            if (std::strcmp(value, "baseline") == 0) {
                o.assign_kernel = AssignKernel::kBaseline;
            } else if (std::strcmp(value, "smem") == 0) {
                o.assign_kernel = AssignKernel::kSharedMemory;
            } else {
                return fail("BLD-003a", "--assign expects baseline or smem, got '" +
                                            std::string(value) + "'");
            }
        } else if (std::strcmp(arg, "--block-size") == 0) {
            int64_t block = 0;
            if (!parse_int64(value, &block) || block < 1 || block > 1048576) {
                return fail("VAL-010", "--block-size expects a positive integer, got '" +
                                           std::string(value) + "'");
            }
            o.block_size = static_cast<int>(block);
        } else if (std::strcmp(arg, "--reduce-chunks") == 0) {
            if (!parse_int64(value, &o.reduce_chunks)) {
                return fail("VAL-010", "--reduce-chunks expects an integer, got '" +
                                           std::string(value) + "'");
            }
        } else if (std::strcmp(arg, "--verify") == 0) {
            if (std::strcmp(value, "none") == 0) {
                o.verify = VerifyMode::kNone;
            } else if (std::strcmp(value, "final") == 0) {
                o.verify = VerifyMode::kFinal;
            } else if (std::strcmp(value, "lockstep") == 0) {
                o.verify = VerifyMode::kLockstep;
            } else {
                return fail("BLD-003a", "--verify expects none, final or lockstep, got '" +
                                            std::string(value) + "'");
            }
        } else if (std::strcmp(arg, "--out-prefix") == 0) {
            o.out_prefix = value;
        } else if (std::strcmp(arg, "--csv") == 0) {
            o.csv_path = value;
        } else if (std::strcmp(arg, "--reps") == 0) {
            if (!parse_int64(value, &o.reps) || o.reps < 1) {
                return fail("PERF-006", "--reps expects a positive integer, got '" +
                                            std::string(value) + "'");
            }
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (!parse_int64(value, &o.warmup) || o.warmup < 0) {
                return fail("PERF-006", "--warmup expects a non-negative integer, got '" +
                                            std::string(value) + "'");
            }
        } else {
            return fail("BLD-003a", "unknown option " + std::string(arg));
        }
    }

    // Required options. BLD-003a shows these without brackets, so a missing one is a usage error
    // and not a silent default: defaulting --strategy or --tol would produce a plausible-looking
    // result for a configuration the caller never specified.
    if (o.data_path.empty()) {
        return fail("BLD-003a", "--data is required");
    }
    if (o.centroids_path.empty()) {
        return fail("BLD-003a", "--centroids is required");
    }
    if (!have_k) {
        return fail("BLD-003a", "--k is required");
    }
    if (!have_max_iter) {
        return fail("BLD-003a", "--max-iter is required");
    }
    if (!have_tol) {
        return fail("BLD-003a", "--tol is required");
    }
    if (!have_strategy) {
        return fail("BLD-003a", "--strategy is required");
    }

    // A CSV row describes one measured repetition (section 13.1), so --csv without --reps has no
    // rows to write and almost certainly means the caller expected a benchmark that will not run.
    if (!o.csv_path.empty() && o.reps == 0) {
        return fail("PERF-006", "--csv needs --reps R: a raw row describes one measured "
                                "repetition, and a run without --reps measures none");
    }
    // FR-019: lockstep downloads sit inside the measured region, so a lockstep run is not a
    // benchmark sample. Producing CSV rows from one would corrupt the benchmark set silently.
    if (o.reps > 0 && o.verify == VerifyMode::kLockstep) {
        return fail("FR-019", "--verify lockstep cannot be combined with --reps: the per-iteration "
                              "downloads sit inside the measured region, so such a run must never "
                              "become a benchmark sample");
    }

    *out = o;
    return CliParse{};
}

}  // namespace kmeans
