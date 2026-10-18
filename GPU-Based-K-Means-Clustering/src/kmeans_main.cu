// kmeans_main.cu — the `kmeans` executable (requirements.md BLD-003a, FR-001, FR-002, FR-015,
// FR-017, FR-019, BLD-004, BLD-005).
//
// Deliberately thin. Argument parsing lives in cli_options.cpp, the §9.1 rules in validate.cpp,
// file formats in io_kmd.cpp, the §10.3 comparison in lockstep.cpp and the FR-015 rendering in
// report.cpp -- all host-only translation units that are compiled and tested with g++ on a
// machine with no GPU (CLAUDE.md). What is left here is the order of operations, which is itself a
// requirement: every VAL rule is applied before the first device allocation (FR-001), so an
// invalid configuration exits with a byte count or a parameter name instead of a CUDA error.
//
//   load (FR-002) -> VAL-001..008 -> device query and VAL-010 -> run (VAL-009 inside) ->
//   verify (FR-019) -> write outputs (FR-015)
//
// Exit codes are BLD-005: 0 success, 2 invalid input, 3 unsupported device or insufficient
// resources, 4 CUDA runtime error, 5 I/O error, 6 verification failure.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

#include "cli_options.h"
#include "common.h"
#include "cpu_reference.h"
#include "cuda_util.cuh"
#include "device_context.cuh"
#include "io_kmd.h"
#include "kmeans_gpu.h"
#include "launch_config.h"
#include "lockstep.h"
#include "report.h"
#include "validate.h"

namespace {

using namespace kmeans;

#define KMEANS_STRINGIFY_(x) #x
#define KMEANS_STRINGIFY(x) KMEANS_STRINGIFY_(x)

std::string env_or(const char *name, const char *fallback)
{
    const char *value = std::getenv(name);
    return (value != nullptr && *value != '\0') ? std::string(value) : std::string(fallback);
}

std::string os_description()
{
#if defined(__linux__) || defined(__APPLE__)
    utsname u{};
    if (uname(&u) == 0) {
        return std::string(u.sysname) + " " + u.release + " " + u.machine;
    }
    return "unknown";
#elif defined(_WIN32)
    return "Windows";
#else
    return "unknown";
#endif
}

// PERF-008. The toolchain facts are compile-time, so they describe the binary that produced the
// measurement and not whatever happens to be on PATH when it runs. The build flags and the source
// revision cannot be known from inside the compiler, so they arrive through the environment, which
// is what the notebook's configuration cell already records (KMEANS_RUN_VERSION).
BuildEnvironment build_environment(const DeviceInfo &device)
{
    BuildEnvironment env;
    env.gpu_name = device.name;
    env.compute_capability =
        std::to_string(device.cc_major) + "." + std::to_string(device.cc_minor);
    env.sm_count = device.sm_count;
    env.driver_version = device.driver_version;
    env.cuda_runtime = device.runtime_version;

#if defined(__CUDACC_VER_MAJOR__)
    env.nvcc_version = KMEANS_STRINGIFY(__CUDACC_VER_MAJOR__) "." KMEANS_STRINGIFY(
        __CUDACC_VER_MINOR__) "." KMEANS_STRINGIFY(__CUDACC_VER_BUILD__);
#else
    env.nvcc_version = "unknown";
#endif
#if defined(__VERSION__)
    env.host_compiler = __VERSION__;
#else
    env.host_compiler = "unknown";
#endif
    env.os = os_description();

    // What the binary can state about its own build without being told: the tile geometry it was
    // compiled with (SMEM-001) and whether the synchronising debug build of BLD-002 is in effect.
    std::string flags = env_or("KMEANS_BUILD_FLAGS", "unknown");
    flags += " [tile_k=" + std::to_string(kAssignTileK) + " tile_d=" +
             std::to_string(kAssignTileD);
#ifdef KMEANS_SYNC_CHECK
    flags += " KMEANS_SYNC_CHECK";
#endif
    flags += "]";
    env.build_flags = flags;

    env.git_commit = env_or("KMEANS_RUN_VERSION", "unknown");
    return env;
}

gpu::Verify to_gpu_verify(VerifyMode mode)
{
    switch (mode) {
    case VerifyMode::kNone:     return gpu::Verify::kNone;
    case VerifyMode::kFinal:    return gpu::Verify::kFinal;
    case VerifyMode::kLockstep: return gpu::Verify::kLockstep;
    }
    return gpu::Verify::kNone;
}

// FR-015: the three outputs plus the summary. Every write failure is an I/O error (exit 5), never
// a silent omission: a run whose outputs did not land is not a successful run.
int write_outputs(const CliOptions &opt, const Result &result, const RunReport &report)
{
    if (opt.out_prefix.empty()) {
        return kExitOk;
    }
    std::string err;
    const std::string a_path = opt.out_prefix + "_A.kma";
    const std::string c_path = opt.out_prefix + "_C.kmd";
    const std::string json_path = opt.out_prefix + "_summary.json";

    int rc = write_kma(a_path, result.assignments, &err);
    if (rc != kExitOk) {
        report_error("FR-015", err);
        return rc;
    }
    rc = write_kmd(c_path, static_cast<uint64_t>(report.config.K),
                   static_cast<uint64_t>(report.config.D), result.centroids.data(), &err);
    if (rc != kExitOk) {
        report_error("FR-015", err);
        return rc;
    }
    rc = write_text_file(json_path, summary_json(report), &err);
    if (rc != kExitOk) {
        report_error("FR-015", err);
        return rc;
    }
    std::printf("outputs: %s, %s, %s\n", a_path.c_str(), c_path.c_str(), json_path.c_str());
    return kExitOk;
}

}  // namespace

int main(int argc, char **argv)
{
    CliOptions opt;
    const CliParse parse = parse_cli(argc, argv, &opt);
    if (parse.usage) {
        std::printf("%s", cli_usage().c_str());
        return kExitOk;
    }
    if (!parse.ok) {
        report_error(parse.rule.c_str(), parse.message);
        std::fprintf(stderr, "%s", cli_usage().c_str());
        return parse.exit_code;
    }

    // --- VAL-010, the part that needs no device ------------------------------------------------
    // Passing 0 for the two device limits skips the checks that need them, so an option outside
    // its documented range is named as such whatever the machine has: a block size of 100 is a
    // VAL-010 violation, not a reason to report that no GPU is visible.
    Validation vranges = validate_options(opt.block_size, opt.reduce_chunks, kAssignTileK,
                                          kAssignTileD, 0, 0);
    vranges.report();
    if (!vranges.ok) {
        return vranges.exit_code;
    }

    // --- FR-002 input loading ------------------------------------------------------------------
    std::string err;
    Matrix X;
    int rc = read_kmd(opt.data_path, &X, &err);
    if (rc != kExitOk) {
        report_error("FR-002", err);
        return rc;
    }
    const int64_t N = static_cast<int64_t>(X.rows);
    const int64_t D = static_cast<int64_t>(X.cols);

    // --- VAL-001..VAL-005 ----------------------------------------------------------------------
    // Before the centroid file is read, because a K of 0 or a negative tolerance should be named
    // as such and not reported as a shape mismatch against a nonsense expectation.
    Validation v = validate_parameters(N, D, opt.K, opt.max_iter, opt.tol);
    v.report();
    if (!v.ok) {
        return v.exit_code;
    }

    // VAL-008 shape: exactly K rows of D values.
    Matrix C0;
    rc = read_kmd_expect(opt.centroids_path, static_cast<uint64_t>(opt.K),
                         static_cast<uint64_t>(D), &C0, &err);
    if (rc != kExitOk) {
        report_error("VAL-008", err);
        return rc;
    }

    // --- VAL-006, VAL-007, VAL-008 duplicates --------------------------------------------------
    double m_range = 0.0;
    Validation vdata = validate_data(X.data, N, D, C0.data, opt.K, &m_range);
    vdata.report();
    if (!vdata.ok) {
        return vdata.exit_code;
    }

    // --- CUDA-001 device query, then VAL-010 against the real device limits --------------------
    DeviceInfo device;
    try {
        device = query_device(0);
    } catch (const CudaError &e) {
        report_error("CUDA-001", e.what());
        return e.exit_code();
    }
    Validation vopt = validate_options(opt.block_size, opt.reduce_chunks, kAssignTileK,
                                       kAssignTileD, device.max_threads_per_block,
                                       static_cast<int64_t>(device.shared_mem_per_block));
    vopt.report();
    if (!vopt.ok) {
        return vopt.exit_code;
    }

    gpu::Options run_options;
    run_options.config = opt.to_config(N, D);
    run_options.m_range = m_range;
    run_options.verify = to_gpu_verify(opt.verify);
    run_options.profile_phases = opt.profile_phases;

    // FR-019 lockstep: the same comparison the test suite uses (lockstep.h), so a failure here and
    // a TEST-I10 failure mean the same thing.
    LockstepStats lockstep_stats;
    const Config config = run_options.config;
    const float *x_ptr = X.data.data();
    gpu::LockstepHook hook;
    if (opt.verify == VerifyMode::kLockstep) {
        hook = [&](const LockstepSnapshot &snap) {
            check_lockstep_iteration(snap, x_ptr, config, m_range, &lockstep_stats);
        };
    }

    std::printf("kmeans: N=%lld D=%lld K=%lld max-iter=%lld tol=%.17g strategy=%s assign=%s "
                "block=%d%s\n",
                static_cast<long long>(N), static_cast<long long>(D),
                static_cast<long long>(opt.K), static_cast<long long>(opt.max_iter), opt.tol,
                strategy_option_name(opt.strategy), assign_option_name(opt.assign_kernel),
                opt.block_size, opt.fixed_iters ? " fixed-iters" : "");
    std::printf("device: %s\n", describe_device(device).c_str());

    // --- PERF-006 repetitions ------------------------------------------------------------------
    // Without --reps this is one run and the timings are reported as they are measured. With
    // --reps R the data stays loaded and the run is repeated in the same process, which is what
    // PERF-006 asks for; the last measured repetition supplies the outputs, since every
    // repetition computes the same clustering.
    const int64_t measured = (opt.reps > 0) ? opt.reps : 1;
    const int64_t warmups = (opt.reps > 0) ? opt.warmup : 0;

    const BuildEnvironment env = build_environment(device);
    const std::string timestamp = utc_timestamp();
    const std::string dataset = dataset_id_from_path(opt.data_path);

    Result result;
    gpu::RunInfo info;
    std::vector<double> t_alg, t_e2e;
    // PERF-007: one phase sum per repetition, so a phase-timing row carries that repetition's
    // own phases rather than the last one's.
    std::vector<double> t_assign, t_accum, t_update, t_conv;
    RunReport report;

    try {
        // parse_cli rejects --reps together with --verify lockstep (FR-019), so the hook is empty
        // whenever this loop runs more than once and the statistics below describe exactly one
        // run's iterations.
        for (int64_t w = 0; w < warmups; ++w) {
            gpu::RunInfo discard;
            gpu::run(x_ptr, C0.data.data(), run_options, &discard, hook);
        }
        if (warmups > 0) {
            std::printf("warm-up: %lld run(s), not recorded (PERF-006)\n",
                        static_cast<long long>(warmups));
        }

        for (int64_t rep = 1; rep <= measured; ++rep) {
            result = gpu::run(x_ptr, C0.data.data(), run_options, &info, hook);
            t_alg.push_back(info.t_alg_ms);
            t_e2e.push_back(info.t_e2e_ms);
            t_assign.push_back(info.t_assign_ms);
            t_accum.push_back(info.t_accum_ms);
            t_update.push_back(info.t_update_ms);
            t_conv.push_back(info.t_conv_ms);

            report = RunReport{};
            report.env = env;
            report.timestamp_utc = timestamp;
            report.dataset_id = dataset;
            report.data_path = opt.data_path;
            report.centroids_path = opt.centroids_path;
            report.config = run_options.config;
            report.tile_k = kAssignTileK;
            report.tile_d = kAssignTileD;
            report.reduce_chunks_g = info.reduce_chunks;
            report.verify = opt.verify;
            report.profile_phases = opt.profile_phases;
            report.rep_index = (opt.reps > 0) ? rep : 0;
            report.iterations = result.iterations;
            report.status = result.status;
            report.centroid_consistent = result.centroid_consistent;
            report.assignment_consistent = result.assignment_consistent;
            report.last_delta = result.last_delta;
            report.has_last_delta = result.has_last_delta;
            report.t_alg_ms = info.t_alg_ms;
            report.t_e2e_ms = info.t_e2e_ms;
            report.t_alloc_ms = info.t_alloc_ms;
            report.t_h2d_ms = info.t_h2d_ms;
            report.t_d2h_ms = info.t_d2h_ms;
            report.timing_valid = info.timing_valid;
            report.t_assign_ms = info.t_assign_ms;
            report.t_accum_ms = info.t_accum_ms;
            report.t_update_ms = info.t_update_ms;
            report.t_conv_ms = info.t_conv_ms;
            report.has_phases = info.phases_measured;
            report.objective = result.objective;
            report.counts = result.counts;
            report.run_id = make_run_id(report);

            if (opt.reps > 0) {
                std::printf("  rep %lld/%lld: t_alg %.3f ms, t_e2e %.3f ms\n",
                            static_cast<long long>(rep), static_cast<long long>(measured),
                            info.t_alg_ms, info.t_e2e_ms);
            }
        }
    } catch (const CudaError &e) {
        // CUDA-008 / MEM-007: the RAII owners have already released everything on the way out.
        report_error("CUDA-008", e.what());
        return e.exit_code();
    } catch (const std::exception &e) {
        report_error("NFR-003", e.what());
        return kExitCudaError;
    }

    // --- §2.7 outcome --------------------------------------------------------------------------
    std::printf("status: %s after %lld iteration(s)", status_name(result.status),
                static_cast<long long>(result.iterations));
    if (result.has_last_delta) {
        std::printf(", last Delta %.9g", result.last_delta);
    } else {
        std::printf(", last Delta null (no update was performed)");
    }
    std::printf("\n");
    std::printf("consistency: P1 assignment_consistent=%s, P2 centroid_consistent=%s\n",
                result.assignment_consistent ? "true" : "false",
                result.centroid_consistent ? "true" : "false");
    std::printf("objective J: %.9e\n", result.objective);
    if (run_options.config.strategy == Strategy::kReduction) {
        std::printf("reduction: G = %lld (G_target %lld, G_mem %lld, %llu B of partials)\n",
                    static_cast<long long>(info.reduce_chunks),
                    static_cast<long long>(info.g_target), static_cast<long long>(info.g_memory),
                    static_cast<unsigned long long>(info.reduction_bytes));
    }
    std::printf("timing: t_alg %.3f ms, t_e2e %.3f ms, t_alloc %.3f ms, t_h2d %.3f ms, "
                "t_d2h %.3f ms%s\n",
                info.t_alg_ms, info.t_e2e_ms, info.t_alloc_ms, info.t_h2d_ms, info.t_d2h_ms,
                info.timing_valid ? "" : "  [NOT a benchmark sample: FR-019 verification]");

    // PERF-007: the phases are sums over the whole T_alg region, so what they do not account for
    // is the per-iteration 16-byte transfer and the host decision gap (CUDA-011, section 2.7).
    // Reporting that remainder rather than distributing it keeps the four numbers measurements.
    if (info.phases_measured) {
        const double sum = info.t_assign_ms + info.t_accum_ms + info.t_update_ms + info.t_conv_ms;
        std::printf("phases: assign %.3f ms, accumulate %.3f ms, update %.3f ms, "
                    "convergence %.3f ms; sum %.3f ms of t_alg %.3f ms "
                    "(%.3f ms unattributed: the 16-byte transfer and the host decision gap)\n",
                    info.t_assign_ms, info.t_accum_ms, info.t_update_ms, info.t_conv_ms, sum,
                    info.t_alg_ms, info.t_alg_ms - sum);
        std::printf("warning[PERF-007]: phase timing records extra events inside the loop, which "
                    "perturbs the totals; these rows are reported separately and are not "
                    "headline numbers\n");
    }

    if (opt.reps > 0) {
        const Statistics alg = statistics(t_alg);
        const Statistics e2e = statistics(t_e2e);
        std::printf("t_alg_ms  over %lld reps: median %.3f, mean %.3f, sd %.3f, min %.3f, "
                    "max %.3f, cv %.4f\n",
                    static_cast<long long>(alg.count), alg.median, alg.mean, alg.stddev, alg.min,
                    alg.max, alg.cv);
        std::printf("t_e2e_ms  over %lld reps: median %.3f, mean %.3f, sd %.3f, min %.3f, "
                    "max %.3f, cv %.4f\n",
                    static_cast<long long>(e2e.count), e2e.median, e2e.mean, e2e.stddev, e2e.min,
                    e2e.max, e2e.cv);
        if (alg.count < 10) {
            std::printf("warning[PERF-006]: %lld measured repetitions is below the minimum of "
                        "10; these numbers are not a benchmark result\n",
                        static_cast<long long>(alg.count));
        }
    }

    // --- FR-019 verification -------------------------------------------------------------------
    int verification_exit = kExitOk;
    if (opt.verify == VerifyMode::kFinal) {
        const gpu::FinalVerification fv =
            gpu::verify_final(x_ptr, result, run_options.config, m_range);
        std::printf("verify final: P1 hard mismatches %lld, near-tie mismatches %lld, P2 flag %s, "
                    "max |C_final - mean| %.3e\n",
                    static_cast<long long>(fv.p1_hard_mismatches),
                    static_cast<long long>(fv.p1_near_tie_mismatches),
                    fv.p2_flag_correct ? "correct" : "WRONG", fv.max_centroid_difference);
        for (const std::string &line : fv.reports) {
            std::printf("  %s\n", line.c_str());
        }
        const bool passed = fv.p1_hard_mismatches == 0 && fv.p2_flag_correct;
        report.correctness_status = passed ? "PASS" : "FAIL";
        if (!passed) {
            report_error("FR-019", "verification failed: " +
                                       std::to_string(fv.p1_hard_mismatches) +
                                       " non-near-tie P1 mismatches, P2 flag " +
                                       (fv.p2_flag_correct ? "correct" : "wrong"));
            verification_exit = kExitVerificationFailed;
        }
    } else if (opt.verify == VerifyMode::kLockstep) {
        std::printf("verify lockstep: %s\n", lockstep_summary(lockstep_stats).c_str());
        for (const std::string &note : lockstep_stats.notes) {
            std::printf("  note: %s\n", note.c_str());
        }
        const bool passed = lockstep_ok(lockstep_stats);
        report.correctness_status = passed ? "PASS" : "FAIL";
        if (!passed) {
            report_error("FR-019", "lockstep verification failed: see the notes above; every "
                                   "counter except near-tie assignment differences must be zero "
                                   "(section 10.3, NUM-005)");
            verification_exit = kExitVerificationFailed;
        }
    }

    // --- FR-015 outputs and the section 13.1 CSV ----------------------------------------------
    // Written even when verification failed, so the failing run can be inspected; the exit code
    // and correctness_status both record that it failed (PERF-001).
    rc = write_outputs(opt, result, report);
    if (rc != kExitOk) {
        return rc;
    }

    if (!opt.csv_path.empty()) {
        // One row per measured repetition. Only the last repetition's report exists in full, so
        // the rows are written from it with the per-repetition timings substituted; every other
        // column is identical across repetitions of one configuration by construction.
        for (int64_t rep = 1; rep <= measured; ++rep) {
            RunReport row = report;
            row.rep_index = rep;
            const size_t r = static_cast<size_t>(rep - 1);
            row.t_alg_ms = t_alg[r];
            row.t_e2e_ms = t_e2e[r];
            row.t_assign_ms = t_assign[r];
            row.t_accum_ms = t_accum[r];
            row.t_update_ms = t_update[r];
            row.t_conv_ms = t_conv[r];
            row.run_id = make_run_id(row);
            if (append_csv_row(opt.csv_path, row, &err) != kExitOk) {
                report_error("PERF-011", err);
                return kExitIoError;
            }
        }
        std::printf("csv: appended %lld row(s) to %s\n", static_cast<long long>(measured),
                    opt.csv_path.c_str());
    }

    if (verification_exit != kExitOk) {
        return verification_exit;
    }
    return kExitOk;
}
