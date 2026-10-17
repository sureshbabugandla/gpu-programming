// gen_main.cpp — the kmeans-gen executable (requirements.md BLD-003a, FR-003).
//
//   kmeans-gen --n N --d D --k K --sep S --spread SIGMA --seed SEED --out-prefix PATH
//
// Writes PATH_X.kmd (N x D) and PATH_C0.kmd (K x D) in the BLD-004 format and prints a one-line
// JSON summary including the dataset ID and the measured minimum centre separation, so a benchmark
// row can be traced back to the exact dataset (PERF-008) and the "well-separated" claim of §10.3
// is a measurement rather than an assumption.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common.h"
#include "gen.h"
#include "io_kmd.h"
#include "validate.h"

namespace {

void usage()
{
    std::fprintf(stderr,
                 "usage: kmeans-gen --n N --d D --k K --sep S --spread SIGMA --seed SEED "
                 "--out-prefix PATH\n");
}

bool parse_int64(const char *text, int64_t *out)
{
    char *end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    *out = static_cast<int64_t>(value);
    return true;
}

bool parse_uint64(const char *text, uint64_t *out)
{
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    *out = static_cast<uint64_t>(value);
    return true;
}

bool parse_double(const char *text, double *out)
{
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

}  // namespace

int main(int argc, char **argv)
{
    using namespace kmeans;

    GenParams params;
    std::string out_prefix;
    bool have_n = false, have_d = false, have_k = false;

    for (int i = 1; i < argc; ++i) {
        const char *flag = argv[i];
        const bool needs_value = std::strncmp(flag, "--", 2) == 0;
        if (!needs_value) {
            report_error("VAL-010", "unexpected argument '" + std::string(flag) + "'");
            usage();
            return kExitInvalidInput;
        }
        if (i + 1 >= argc) {
            report_error("VAL-010", "option '" + std::string(flag) + "' needs a value");
            usage();
            return kExitInvalidInput;
        }
        const char *value = argv[++i];
        bool parsed = true;
        if (std::strcmp(flag, "--n") == 0) {
            parsed = parse_int64(value, &params.N);
            have_n = true;
        } else if (std::strcmp(flag, "--d") == 0) {
            parsed = parse_int64(value, &params.D);
            have_d = true;
        } else if (std::strcmp(flag, "--k") == 0) {
            parsed = parse_int64(value, &params.K);
            have_k = true;
        } else if (std::strcmp(flag, "--sep") == 0) {
            parsed = parse_double(value, &params.sep);
        } else if (std::strcmp(flag, "--spread") == 0) {
            parsed = parse_double(value, &params.spread);
        } else if (std::strcmp(flag, "--seed") == 0) {
            parsed = parse_uint64(value, &params.seed);
        } else if (std::strcmp(flag, "--out-prefix") == 0) {
            out_prefix = value;
        } else {
            report_error("VAL-010", "unknown option '" + std::string(flag) + "'");
            usage();
            return kExitInvalidInput;
        }
        if (!parsed) {
            report_error("VAL-010", "value '" + std::string(value) + "' for option '" +
                                        std::string(flag) + "' could not be parsed");
            return kExitInvalidInput;
        }
    }

    if (!have_n || !have_d || !have_k || out_prefix.empty()) {
        report_error("VAL-010", "--n, --d, --k and --out-prefix are required");
        usage();
        return kExitInvalidInput;
    }

    // The generator's sizes obey the same VAL-001/VAL-002 limits as a clustering run, so an
    // impossible dataset is refused before any memory is touched.
    const Validation params_ok = validate_parameters(params.N, params.D, params.K, 0, 0.0);
    if (!params_ok.ok) {
        params_ok.report();
        return params_ok.exit_code;
    }
    if (!(params.spread >= 0.0) || !(params.sep >= 0.0)) {
        report_error("VAL-010", "--sep and --spread must be non-negative and not NaN");
        return kExitInvalidInput;
    }

    const Dataset data = generate(params);

    std::string err;
    const std::string x_path = out_prefix + "_X.kmd";
    const std::string c_path = out_prefix + "_C0.kmd";
    int code = write_kmd(x_path, static_cast<uint64_t>(params.N), static_cast<uint64_t>(params.D),
                         data.X.data(), &err);
    if (code != kExitOk) {
        report_error("FR-002", err);
        return code;
    }
    code = write_kmd(c_path, static_cast<uint64_t>(params.K), static_cast<uint64_t>(params.D),
                     data.C0.data(), &err);
    if (code != kExitOk) {
        report_error("FR-002", err);
        return code;
    }

    const double sigma_ratio =
        params.spread > 0.0 ? data.min_centre_distance / params.spread : 0.0;
    std::printf("{\"dataset_id\": \"%s\", \"N\": %lld, \"D\": %lld, \"K\": %lld, \"sep\": %.17g, "
                "\"spread\": %.17g, \"seed\": %llu, \"min_centre_distance\": %.17g, "
                "\"min_separation_sigma\": %.17g, \"well_separated\": %s, \"data\": \"%s\", "
                "\"centroids\": \"%s\"}\n",
                data.dataset_id.c_str(), static_cast<long long>(params.N),
                static_cast<long long>(params.D), static_cast<long long>(params.K), params.sep,
                params.spread, static_cast<unsigned long long>(params.seed),
                data.min_centre_distance, sigma_ratio, sigma_ratio >= 10.0 ? "true" : "false",
                x_path.c_str(), c_path.c_str());
    return kExitOk;
}
