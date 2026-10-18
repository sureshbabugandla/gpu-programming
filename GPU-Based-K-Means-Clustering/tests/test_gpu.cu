// test_gpu.cu — GPU test driver for milestones M2 (baseline assign, Approach A, update,
// convergence, finalization), M3 (Approach B) and M4 (assign_smem).
//
// Covers, for both accumulation strategies:
//   TEST-I01 / TEST-I02   H1-H6 and H2-a..e against the expected results of section 16.0
//   TEST-I03 / I04 / I05  iteration counting, the movement boundary and the iteration limit,
//                         which are exactly what the H2 variants encode
//   TEST-I06              the P2 flag of section 2.8 against the CPU reference
//   TEST-I07              P1 and the P2 flag on blob data (gpu::verify_final, FR-019 `final`)
//   TEST-I10 / section 10.3  lockstep: every iteration of a 100k x 16, K=8 blob run compared
//                         against the CPU reference, kernel by kernel
//   TEST-U03              the change count, exactly, from the lockstep snapshots
//   TEST-U04 / U05        accumulation (atomic; reduction over G in {1, 2, 7, N} and auto)
//   TEST-U06              poisoned P, Q, S, n must not survive into any result
//   TEST-U07              partial Stage 2 tiles: K = 1024, D in {1, 33, 65}, small chunks
//   TEST-U08 / U10        movement and update, from the lockstep snapshots
//   TEST-E16 / EDGE-16    N, D, K not multiples of the block, tile or warp size
//   SMEM-004              assign_smem produces bit-identical runs to the baseline assign on
//                         every test dataset, and TEST-U01 in lockstep for the tiled kernel
//
// One executable, non-zero exit on any failure, so a single notebook cell can gate the run
// (NFR-015). `--quick` runs only the subset the development notebook cells need; the full run
// adds the G sweep, the partial-tile cases, the non-aligned size sweep, the SMEM-004 size matrix
// and the assign_smem lockstep run.
//
// No result here is evidence of anything until an executed notebook in results/ shows it
// (CLAUDE.md): this file only defines what must hold.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "cpu_reference.h"
#include "cuda_util.cuh"
#include "gen.h"
#include "hand_data.h"
#include "kmeans_gpu.h"
#include "launch_config.h"
#include "lockstep.h"
#include "test_device.cuh"
#include "test_harness.h"
#include "validate.h"

using namespace kmeans;

namespace {

using kmeans::tests::assign_name;
using kmeans::tests::check;
using kmeans::tests::check_eq_i64;
using kmeans::tests::double_str;
using kmeans::tests::float_str;
using kmeans::tests::run_device;
using kmeans::tests::strategy_name;
using kmeans::tests::TestScope;

// --- TEST-I01 / TEST-I02: the hand-verifiable datasets on the device ---------------------------

void test_hand_cases(Strategy strategy)
{
    for (const tests::HandCase &c : tests::hand_cases()) {
        TestScope scope(strategy == Strategy::kAtomic ? "I01" : "I02",
                        std::string(strategy_name(strategy)) + "  " + c.id +
                            (c.note.empty() ? "" : "  (" + c.note + ")"));

        double m_range = 0.0;
        const Validation data = validate_data(c.X, c.N, c.D, c.C0, c.K, &m_range);
        check(data.ok, c.id + ": data accepted");

        gpu::Options options;
        options.config.N = c.N;
        options.config.D = c.D;
        options.config.K = c.K;
        options.config.max_iter = c.max_iter;
        options.config.tol = c.tol;
        options.config.strategy = strategy;
        options.m_range = m_range;
        options.verify = gpu::Verify::kFinal;

        Result r;
        gpu::RunInfo info;
        if (!run_device(c.X.data(), c.C0.data(), options, &r, &info, gpu::LockstepHook(),
                        c.id + ": device run")) {
            continue;
        }

        check_eq_i64(r.iterations, c.expected_iterations, c.id + ": iteration count (DEC-006)");
        check(r.status == c.expected_status,
              c.id + ": status: got " + status_name(r.status) + ", want " +
                  status_name(c.expected_status));
        check(r.centroid_consistent == c.expected_centroid_consistent,
              c.id + ": P2 flag: got " + (r.centroid_consistent ? "true" : "false") + ", want " +
                  (c.expected_centroid_consistent ? "true" : "false"));
        check(r.assignment_consistent, c.id + ": P1 flag true by construction");

        for (int64_t i = 0; i < c.N; ++i) {
            check_eq_i64(r.assignments[static_cast<size_t>(i)],
                         c.expected_A[static_cast<size_t>(i)],
                         c.id + ": A[" + std::to_string(i) + "]");
        }
        for (int64_t k = 0; k < c.K; ++k) {
            check_eq_i64(r.counts[static_cast<size_t>(k)], c.expected_n[static_cast<size_t>(k)],
                         c.id + ": n[" + std::to_string(k) + "]");
        }

        const double s_max = s_max_from_range(m_range);
        for (int64_t e = 0; e < c.K * c.D; ++e) {
            const float got = r.centroids[static_cast<size_t>(e)];
            const float want = c.expected_C[static_cast<size_t>(e)];
            const bool ok = c.centroids_exact ? (got == want) : centroid_agrees(got, want, s_max);
            check(ok, c.id + ": C[" + std::to_string(e) + "]: got " + float_str(got) + ", want " +
                          float_str(want) + (c.centroids_exact ? " (exact)" : " (NUM-006)"));
        }

        // DEC-003: an empty cluster keeps its initial centroid bit-exactly. True for every H case
        // because none of them ever re-populates an empty cluster.
        for (int64_t k = 0; k < c.K; ++k) {
            if (c.expected_n[static_cast<size_t>(k)] != 0) {
                continue;
            }
            for (int64_t j = 0; j < c.D; ++j) {
                const size_t e = static_cast<size_t>(k * c.D + j);
                check(r.centroids[e] == c.C0[e],
                      c.id + ": empty cluster " + std::to_string(k) + " retained C0[" +
                          std::to_string(j) + "] bit-exactly");
            }
        }

        // TEST-I06 and the CPU oracle side by side: the device must reach the same record the
        // reference does on these cases, which section 16.0 fixes by hand.
        const Result ref = cpu::run(c.X.data(), c.N, c.D, c.C0.data(), c.K, c.max_iter, c.tol,
                                    m_range, false);
        check(ref.status == r.status, c.id + ": status matches the CPU reference");
        check_eq_i64(r.iterations, ref.iterations, c.id + ": iterations match the CPU reference");
        check(r.centroid_consistent == ref.centroid_consistent,
              c.id + ": P2 flag matches the CPU reference");

        const gpu::FinalVerification v = gpu::verify_final(c.X.data(), r, options.config, m_range);
        check_eq_i64(v.p1_hard_mismatches, 0, c.id + ": P1 hard mismatches (NUM-005)");
        check(v.p2_flag_correct, c.id + ": P2 flag agrees with the reference finalization");
    }
}

// --- section 10.3 lockstep -----------------------------------------------------------------------

// LockstepStats and check_lockstep_iteration live in src/lockstep.h so that the section 10.3
// comparison is one implementation shared with the `--verify lockstep` mode of the kmeans
// executable (FR-019), and so that the numerical argument it encodes is compiled and tested
// with g++ on a machine with no GPU. What stays here is only the pass/fail wrapper.

void report_lockstep(const LockstepStats &stats, const std::string &prefix)
{
    check(stats.iterations > 0, prefix + ": at least one iteration was checked");
    check_eq_i64(stats.assign_hard_mismatches, 0, prefix + ": assign hard mismatches (NUM-005)");
    check_eq_i64(stats.count_mismatches, 0, prefix + ": count mismatches (exact)");
    check_eq_i64(stats.sum_violations, 0, prefix + ": sum violations");
    check_eq_i64(stats.centroid_violations, 0, prefix + ": centroid violations (NUM-006)");
    check_eq_i64(stats.delta_violations, 0, prefix + ": Delta violations");
    check_eq_i64(stats.change_count_mismatches, 0, prefix + ": change count mismatches");
    std::printf("      %s\n", lockstep_summary(stats).c_str());
    for (const std::string &note : stats.notes) {
        std::printf("      note: %s\n", note.c_str());
    }
}

// The 100k x 16, K = 8 blob dataset the Batch A notebook cell runs. The parameters live here
// alone, so the dataset and the sizes the test reports can never drift apart.
GenParams lockstep_params()
{
    GenParams p;
    p.N = 100000;
    p.D = 16;
    p.K = 8;
    p.sep = 10.0;
    p.spread = 1.0;
    p.seed = 20261018;
    return p;
}

void test_lockstep(Strategy strategy, AssignKernel assign, const Dataset &data,
                   const GenParams &params)
{
    TestScope scope("I10", std::string(strategy_name(strategy)) + " / " + assign_name(assign) +
                               "  lockstep vs. CPU reference on " +
                               std::to_string(params.N) + " x " + std::to_string(params.D) +
                               ", K=" + std::to_string(params.K) + " blobs (section 10.3)");

    double m_range = 0.0;
    const Validation v = validate_data(data.X, params.N, params.D, data.C0, params.K, &m_range);
    check(v.ok, "generated data is valid");

    gpu::Options options;
    options.config.N = params.N;
    options.config.D = params.D;
    options.config.K = params.K;
    options.config.max_iter = 20;
    // tol = 0 makes the movement criterion unreachable in practice, so the run converges by
    // assignment stability and every iteration up to that point is lockstep-checked.
    options.config.tol = 0.0;
    options.config.strategy = strategy;
    options.config.assign_kernel = assign;
    options.m_range = m_range;
    options.verify = gpu::Verify::kLockstep;

    LockstepStats stats;
    const Config config = options.config;
    const gpu::LockstepHook hook = [&](const gpu::LockstepSnapshot &snap) {
        check_lockstep_iteration(snap, data.X.data(), config, m_range, &stats);
    };

    Result r;
    gpu::RunInfo info;
    if (!run_device(data.X.data(), data.C0.data(), options, &r, &info, hook, "lockstep run")) {
        return;
    }

    report_lockstep(stats, std::string(strategy_name(strategy)) + "/" + assign_name(assign));
    check(!info.timing_valid,
          "a lockstep run is marked as not timing-valid (FR-019: never a benchmark sample)");
    if (strategy == Strategy::kReduction) {
        check(info.reduce_chunks >= 1 && info.reduce_chunks <= params.N,
              "the selected G is in [1, N]: " + std::to_string(info.reduce_chunks));
        std::printf("      G = %lld (G_target %lld, G_mem %lld, %llu B of partials)\n",
                    static_cast<long long>(info.reduce_chunks),
                    static_cast<long long>(info.g_target),
                    static_cast<long long>(info.g_memory),
                    static_cast<unsigned long long>(info.reduction_bytes));
    }

    // TEST-I07: the same run judged by FR-019 `final`.
    const gpu::FinalVerification fv = gpu::verify_final(data.X.data(), r, config, m_range);
    check_eq_i64(fv.p1_hard_mismatches, 0, "P1 hard mismatches after the run");
    check(fv.p2_flag_correct, "the P2 flag agrees with the reference finalization");
    check(is_converged(r.status) || r.status == Status::ITERATION_LIMIT,
          std::string("status is a non-error status: ") + status_name(r.status));
    check(std::isfinite(r.objective) && r.objective > 0.0,
          "the objective J is finite and positive: " + double_str(r.objective));
    std::printf("      status %s, t = %lld, J = %.9e, T_alg %.3f ms (not a benchmark sample)\n",
                status_name(r.status), static_cast<long long>(r.iterations), r.objective,
                info.t_alg_ms);
}

// --- TEST-U06: poisoned buffers must not survive ------------------------------------------------

void test_poisoning(Strategy strategy)
{
    TestScope scope("U06", std::string(strategy_name(strategy)) +
                               "  poisoned P, Q, S and n do not reach any result (MEM-006)");

    GenParams p;
    p.N = 10000;
    p.D = 8;
    p.K = 5;
    p.sep = 10.0;
    p.spread = 1.0;
    p.seed = 606;
    const Dataset data = generate(p);

    double m_range = 0.0;
    const Validation v = validate_data(data.X, p.N, p.D, data.C0, p.K, &m_range);
    check(v.ok, "generated data is valid");

    gpu::Options clean;
    clean.config.N = p.N;
    clean.config.D = p.D;
    clean.config.K = p.K;
    // Three accumulations at least: two in the loop plus the finalization pass, which is what
    // "call twice" in TEST-U06 asks for.
    clean.config.max_iter = 2;
    clean.config.tol = 0.0;
    clean.config.fixed_iters = true;
    clean.config.strategy = strategy;
    clean.m_range = m_range;

    gpu::Options poisoned = clean;
    poisoned.poison_buffers = true;

    Result r_clean, r_poisoned;
    gpu::RunInfo info_clean, info_poisoned;
    if (!run_device(data.X.data(), data.C0.data(), clean, &r_clean, &info_clean,
                    gpu::LockstepHook(), "unpoisoned run")) {
        return;
    }
    if (!run_device(data.X.data(), data.C0.data(), poisoned, &r_poisoned, &info_poisoned,
                    gpu::LockstepHook(), "poisoned run")) {
        return;
    }

    check_eq_i64(r_poisoned.iterations, r_clean.iterations, "iteration count unchanged");
    check(r_poisoned.status == r_clean.status, "status unchanged");

    int64_t assignment_diffs = 0, count_diffs = 0, non_finite = 0, centroid_diffs = 0;
    double worst = 0.0;
    for (int64_t i = 0; i < p.N; ++i) {
        if (r_poisoned.assignments[static_cast<size_t>(i)] !=
            r_clean.assignments[static_cast<size_t>(i)]) {
            ++assignment_diffs;
        }
    }
    for (int64_t k = 0; k < p.K; ++k) {
        if (r_poisoned.counts[static_cast<size_t>(k)] != r_clean.counts[static_cast<size_t>(k)]) {
            ++count_diffs;
        }
    }
    const double s_max = s_max_from_range(m_range);
    for (int64_t e = 0; e < p.K * p.D; ++e) {
        const float got = r_poisoned.centroids[static_cast<size_t>(e)];
        const float want = r_clean.centroids[static_cast<size_t>(e)];
        if (!std::isfinite(got)) {
            ++non_finite;
            continue;
        }
        const double diff = std::fabs(static_cast<double>(got) - static_cast<double>(want));
        if (diff > worst) {
            worst = diff;
        }
        if (strategy == Strategy::kReduction) {
            // RED-009: a fixed summation order, so two runs of the same configuration are
            // expected to agree bit for bit; a surviving poison would show up here first.
            if (got != want) {
                ++centroid_diffs;
            }
        } else if (!centroid_agrees(got, want, s_max)) {
            // ATOM-007: Approach A fixes no summation order, so the comparison is NUM-006.
            ++centroid_diffs;
        }
    }

    check_eq_i64(non_finite, 0, "every centroid is finite (no NaN survived)");
    check_eq_i64(assignment_diffs, 0, "assignments identical to the unpoisoned run");
    check_eq_i64(count_diffs, 0, "counts identical to the unpoisoned run");
    check_eq_i64(centroid_diffs, 0,
                 std::string("centroids ") +
                     (strategy == Strategy::kReduction ? "bitwise identical (RED-009)"
                                                       : "within NUM-006 (ATOM-007)"));
    check(std::isfinite(r_poisoned.objective), "the objective J is finite");
    std::printf("      max |dC| between the poisoned and clean runs: %.3e\n", worst);
}

// --- TEST-U05: the G sweep ----------------------------------------------------------------------

// Accumulation is compared through the lockstep snapshot of a single iteration, which is the only
// place the device's S and n are visible to a test (ARC-07 outputs).
void test_chunk_sweep()
{
    GenParams p;
    p.N = 20000;
    p.D = 6;
    p.K = 7;
    p.sep = 10.0;
    p.spread = 1.0;
    p.seed = 505;
    const Dataset data = generate(p);

    double m_range = 0.0;
    const Validation v = validate_data(data.X, p.N, p.D, data.C0, p.K, &m_range);

    const int64_t chunk_counts[] = {1, 2, 7, p.N, 0};  // 0 selects automatically (RED-006)
    for (int64_t user_g : chunk_counts) {
        TestScope scope("U05", "reduction accumulation with G = " +
                                   (user_g == 0 ? std::string("auto") : std::to_string(user_g)));
        check(v.ok, "generated data is valid");

        gpu::Options options;
        options.config.N = p.N;
        options.config.D = p.D;
        options.config.K = p.K;
        options.config.max_iter = 1;
        options.config.tol = 0.0;
        options.config.fixed_iters = true;
        options.config.strategy = Strategy::kReduction;
        options.config.reduce_chunks = user_g;
        options.m_range = m_range;
        options.verify = gpu::Verify::kLockstep;

        LockstepStats stats;
        const Config config = options.config;
        const gpu::LockstepHook hook = [&](const gpu::LockstepSnapshot &snap) {
            check_lockstep_iteration(snap, data.X.data(), config, m_range, &stats);
        };

        Result r;
        gpu::RunInfo info;
        if (!run_device(data.X.data(), data.C0.data(), options, &r, &info, hook, "run")) {
            continue;
        }
        if (user_g != 0) {
            check_eq_i64(info.reduce_chunks, user_g, "the override was honoured");
        }
        report_lockstep(stats, "G=" + std::to_string(info.reduce_chunks));
    }
}

// --- TEST-U07: partial Stage 2 tiles -------------------------------------------------------------

void test_partial_tiles()
{
    // K = 1024 with only a few thousand points means most (chunk, cluster) pairs are empty, which
    // is the EDGE-35 case; D in {1, 33, 65} makes the last 32-dimension tile of Stage 2 partial.
    const int64_t dims[] = {1, 33, 65};
    for (int64_t D : dims) {
        TestScope scope("U07", "reduction with K = 1024, D = " + std::to_string(D) +
                                   ", G = 7: partial tiles and empty chunks (EDGE-35)");

        GenParams p;
        p.N = 5000;
        p.D = D;
        p.K = 1024;
        p.sep = 10.0;
        p.spread = 1.0;
        p.seed = 707;
        const Dataset data = generate(p);

        double m_range = 0.0;
        const Validation v = validate_data(data.X, p.N, p.D, data.C0, p.K, &m_range);
        check(v.ok, "generated data is valid");

        gpu::Options options;
        options.config.N = p.N;
        options.config.D = p.D;
        options.config.K = p.K;
        options.config.max_iter = 1;
        options.config.tol = 0.0;
        options.config.fixed_iters = true;
        options.config.strategy = Strategy::kReduction;
        options.config.reduce_chunks = 7;
        options.m_range = m_range;
        options.verify = gpu::Verify::kLockstep;

        LockstepStats stats;
        const Config config = options.config;
        std::vector<double> column_sums(static_cast<size_t>(D), 0.0);
        std::vector<double> device_column_sums(static_cast<size_t>(D), 0.0);
        const gpu::LockstepHook hook = [&](const gpu::LockstepSnapshot &snap) {
            check_lockstep_iteration(snap, data.X.data(), config, m_range, &stats);
            // RED-005: summing S over all clusters must reproduce the column sums of X, because
            // every (i, j) contributed exactly once.
            for (int64_t k = 0; k < config.K; ++k) {
                for (int64_t j = 0; j < config.D; ++j) {
                    device_column_sums[static_cast<size_t>(j)] +=
                        snap.sums[static_cast<size_t>(k * config.D + j)];
                }
            }
        };

        for (int64_t i = 0; i < p.N; ++i) {
            for (int64_t j = 0; j < D; ++j) {
                column_sums[static_cast<size_t>(j)] +=
                    static_cast<double>(data.X[static_cast<size_t>(i * D + j)]);
            }
        }

        Result r;
        gpu::RunInfo info;
        if (!run_device(data.X.data(), data.C0.data(), options, &r, &info, hook, "run")) {
            continue;
        }
        report_lockstep(stats, "K=1024, D=" + std::to_string(D));

        int64_t column_violations = 0;
        double worst = 0.0;
        const double s_max = s_max_from_range(m_range);
        for (int64_t j = 0; j < D; ++j) {
            const double want = column_sums[static_cast<size_t>(j)];
            const double got = device_column_sums[static_cast<size_t>(j)];
            const double diff = std::fabs(got - want);
            if (diff > worst) {
                worst = diff;
            }
            const double scale = static_cast<double>(p.N) * s_max;
            if (!(diff <= 1e-10 * (scale > s_max ? scale : s_max))) {
                ++column_violations;
            }
        }
        check_eq_i64(column_violations, 0, "column sums of X preserved by the reduction");
        std::printf("      max column-sum difference %.3e over %lld dimensions\n", worst,
                    static_cast<long long>(D));
    }
}

// --- TEST-E16 / EDGE-16: non-aligned sizes -------------------------------------------------------

void test_non_aligned(Strategy strategy)
{
    const int64_t n_values[] = {1, 31, 33, 255, 257, 1023};
    const int64_t d_values[] = {1, 3, 31, 33, 65};
    const int64_t k_values[] = {1, 3, 7, 9, 17};

    TestScope scope("E16", std::string(strategy_name(strategy)) +
                               "  N, D, K not multiples of the block, tile or warp size");

    int64_t configurations = 0, mismatches = 0;
    for (int64_t N : n_values) {
        for (int64_t D : d_values) {
            for (int64_t K : k_values) {
                GenParams p;
                p.N = N;
                p.D = D;
                p.K = K;
                // Section 10.3 method 2: separation far above 10 sigma, so no near-tie is
                // expected and device and reference must agree exactly.
                p.sep = 50.0;
                p.spread = 0.1;
                p.seed = 1600 + static_cast<uint64_t>(N * 1000 + D * 10 + K);
                const Dataset data = generate(p);

                double m_range = 0.0;
                const Validation v = validate_data(data.X, N, D, data.C0, K, &m_range);
                if (!v.ok) {
                    check(false, "N=" + std::to_string(N) + " D=" + std::to_string(D) +
                                     " K=" + std::to_string(K) + ": generated data rejected: " +
                                     v.message);
                    continue;
                }

                gpu::Options options;
                options.config.N = N;
                options.config.D = D;
                options.config.K = K;
                options.config.max_iter = 25;
                options.config.tol = 0.0;
                options.config.strategy = strategy;
                options.m_range = m_range;

                Result r;
                gpu::RunInfo info;
                if (!run_device(data.X.data(), data.C0.data(), options, &r, &info,
                                gpu::LockstepHook(),
                                "N=" + std::to_string(N) + " D=" + std::to_string(D) +
                                    " K=" + std::to_string(K))) {
                    continue;
                }
                ++configurations;

                const Result ref = cpu::run(data.X.data(), N, D, data.C0.data(), K, 25, 0.0,
                                            m_range, false);
                const std::string tag = "N=" + std::to_string(N) + " D=" + std::to_string(D) +
                                        " K=" + std::to_string(K);
                bool equal = r.status == ref.status && r.iterations == ref.iterations &&
                             r.centroid_consistent == ref.centroid_consistent;
                for (int64_t i = 0; equal && i < N; ++i) {
                    equal = r.assignments[static_cast<size_t>(i)] ==
                            ref.assignments[static_cast<size_t>(i)];
                }
                for (int64_t k = 0; equal && k < K; ++k) {
                    equal = r.counts[static_cast<size_t>(k)] == ref.counts[static_cast<size_t>(k)];
                }
                const double s_max = s_max_from_range(m_range);
                for (int64_t e = 0; equal && e < K * D; ++e) {
                    equal = centroid_agrees(r.centroids[static_cast<size_t>(e)],
                                            ref.centroids[static_cast<size_t>(e)], s_max);
                }
                if (!equal) {
                    ++mismatches;
                    if (mismatches <= 5) {
                        const gpu::FinalVerification fv =
                            gpu::verify_final(data.X.data(), r, options.config, m_range);
                        check(false, tag + ": device and reference disagree (status " +
                                         status_name(r.status) + " vs " + status_name(ref.status) +
                                         ", t " + std::to_string(r.iterations) + " vs " +
                                         std::to_string(ref.iterations) + ", P1 hard mismatches " +
                                         std::to_string(fv.p1_hard_mismatches) + ")");
                    }
                }
            }
        }
    }
    check_eq_i64(mismatches, 0, "every non-aligned configuration matched the CPU reference");
    std::printf("      %lld configurations run\n", static_cast<long long>(configurations));
}

// --- SMEM-004: assign_smem must agree with the baseline exactly ---------------------------------
//
// Both kernels accumulate the same binary32 terms in the same ascending-j order and compare
// candidates in the same ascending-k order with the same strict-less-than rule, so the computed
// distances are bit-identical and the whole run must be too -- assignments, counts, centroids,
// iteration count, status and Delta. That is a stronger statement than "within NUM-006", and it
// is the one SMEM-004 asks for when both kernels are compiled with the same FMA settings. A
// difference would mean the tiled kernel reordered arithmetic (NUM-002) or that ptxas contracted
// a multiply-add in one kernel and not the other, which SMEM-004 requires to be explained.

// One (dataset, parameters) pair run with both assign kernels and compared. Returns false when
// either run failed on the device, so the caller can keep counting configurations honestly.
bool compare_assign_kernels(const float *X, const float *C0, const Config &base, double m_range,
                            const std::string &tag)
{
    gpu::Options options;
    options.config = base;
    options.m_range = m_range;

    Result baseline;
    gpu::RunInfo info;
    options.config.assign_kernel = AssignKernel::kBaseline;
    if (!run_device(X, C0, options, &baseline, &info, gpu::LockstepHook(), tag + " baseline")) {
        return false;
    }

    Result smem;
    options.config.assign_kernel = AssignKernel::kSharedMemory;
    if (!run_device(X, C0, options, &smem, &info, gpu::LockstepHook(), tag + " smem")) {
        return false;
    }

    std::string why;
    check(tests::results_bit_equal(baseline, smem, &why),
          tag + ": assign_smem differs from assign (" + why + ")");
    return true;
}

// H1-H6 and H2-a..e: tiny, hand-derived, and every one of them has K <= 3 and D <= 2, so they
// exercise the partial-tile paths of the loader -- T_K = 8 and T_D = 32 both overhang.
void test_smem_hand_cases(Strategy strategy)
{
    for (const tests::HandCase &c : tests::hand_cases()) {
        TestScope scope("SMEM-004", std::string(strategy_name(strategy)) + "  " + c.id +
                                        (c.note.empty() ? "" : "  (" + c.note + ")"));
        double m_range = 0.0;
        const Validation v = validate_data(c.X, c.N, c.D, c.C0, c.K, &m_range);
        check(v.ok, c.id + ": the hand dataset is valid");
        if (!v.ok) {
            continue;
        }

        Config config;
        config.N = c.N;
        config.D = c.D;
        config.K = c.K;
        config.max_iter = c.max_iter;
        config.tol = c.tol;
        config.strategy = strategy;
        compare_assign_kernels(c.X.data(), c.C0.data(), config, m_range, c.id);
    }
}

// Blob datasets. The sizes are chosen around the tile geometry rather than around the warp: D
// below, at and above T_D = 32, K below, at and above T_K = 8, and N that is not a multiple of
// the block size so the final block is partial (SMEM-002, EDGE-17).
void test_smem_datasets(bool quick)
{
    struct Case {
        int64_t N, D, K;
        const char *why;
    };
    const Case quick_cases[] = {
        {1, 1, 1, "N = D = K = 1: one partial tile in both directions"},
        {257, 33, 7, "D and K just over one tile, partial final block"},
        {1023, 32, 8, "D and K exactly one tile"},
        {4096, 65, 9, "three dimension tiles, two centroid tiles, both overhanging"},
    };
    const Case full_cases[] = {
        {50000, 128, 100, "many tiles in both directions (13 k-tiles x 4 j-tiles)"},
        {10000, 512, 64, "D = 512: 16 dimension tiles per centroid tile"},
        {10000, 2, 1024, "K = 1024 with D below one dimension tile"},
    };

    const Strategy strategies[] = {Strategy::kAtomic, Strategy::kReduction};
    int64_t configurations = 0;

    auto run_case = [&](const Case &c) {
        GenParams p;
        p.N = c.N;
        p.D = c.D;
        p.K = c.K;
        // Section 10.3 method 2: far above 10 sigma, so the two kernels cannot be let off the
        // hook by a near-tie that happens to fall the same way in both.
        p.sep = 50.0;
        p.spread = 0.1;
        p.seed = 4000 + static_cast<uint64_t>(c.N * 1000 + c.D * 10 + c.K);
        const Dataset data = generate(p);

        double m_range = 0.0;
        const Validation v = validate_data(data.X, c.N, c.D, data.C0, c.K, &m_range);
        if (!v.ok) {
            check(false, std::string(c.why) + ": generated data rejected: " + v.message);
            return;
        }

        for (Strategy strategy : strategies) {
            TestScope scope("SMEM-004", std::string(strategy_name(strategy)) + "  N=" +
                                            std::to_string(c.N) + " D=" + std::to_string(c.D) +
                                            " K=" + std::to_string(c.K) + "  " + c.why);
            Config config;
            config.N = c.N;
            config.D = c.D;
            config.K = c.K;
            config.max_iter = 15;
            config.tol = 0.0;
            config.strategy = strategy;
            if (compare_assign_kernels(data.X.data(), data.C0.data(), config, m_range,
                                       std::string(c.why))) {
                ++configurations;
            }
        }
    };

    for (const Case &c : quick_cases) {
        run_case(c);
    }
    if (!quick) {
        for (const Case &c : full_cases) {
            run_case(c);
        }
    }
    std::printf("      %lld configurations compared kernel against kernel\n",
                static_cast<long long>(configurations));
}

// The TEST-E16 size matrix, run through both assign kernels. This is the "all test datasets" part
// of SMEM-004: the non-aligned sweep is where a tile loader that mishandles an overhang shows up.
void test_smem_non_aligned()
{
    const int64_t n_values[] = {1, 31, 33, 255, 257, 1023};
    const int64_t d_values[] = {1, 3, 31, 33, 65};
    const int64_t k_values[] = {1, 3, 7, 9, 17};

    TestScope scope("SMEM-004", "assign vs. assign_smem over the TEST-E16 size matrix");
    int64_t configurations = 0;
    for (int64_t N : n_values) {
        for (int64_t D : d_values) {
            for (int64_t K : k_values) {
                GenParams p;
                p.N = N;
                p.D = D;
                p.K = K;
                p.sep = 50.0;
                p.spread = 0.1;
                // The same seeds as test_non_aligned, so the two tests see the same datasets.
                p.seed = 1600 + static_cast<uint64_t>(N * 1000 + D * 10 + K);
                const Dataset data = generate(p);

                double m_range = 0.0;
                const Validation v = validate_data(data.X, N, D, data.C0, K, &m_range);
                if (!v.ok) {
                    check(false, "generated data rejected: " + v.message);
                    continue;
                }

                Config config;
                config.N = N;
                config.D = D;
                config.K = K;
                config.max_iter = 25;
                config.tol = 0.0;
                config.strategy = Strategy::kReduction;  // the deterministic one (RED-009)
                if (compare_assign_kernels(data.X.data(), data.C0.data(), config, m_range,
                                           "N=" + std::to_string(N) + " D=" + std::to_string(D) +
                                               " K=" + std::to_string(K))) {
                    ++configurations;
                }
            }
        }
    }
    std::printf("      %lld configurations compared kernel against kernel\n",
                static_cast<long long>(configurations));
}

}  // namespace

int main(int argc, char **argv)
{
    // The notebook runs the hand cases and the kernel checks as separate cells (section 15.3
    // parts 7 and 8), so the sections are selectable; --quick drops the sweeps that the Batch A
    // cells do not need.
    bool quick = false;
    bool run_hand = true;
    bool run_kernels = true;
    bool run_smem = true;
    const char *selection = "all sections";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            quick = true;
        } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            const char *what = argv[++i];
            if (std::strcmp(what, "hand") == 0) {
                run_kernels = run_smem = false;
                selection = "hand cases only";
            } else if (std::strcmp(what, "kernels") == 0) {
                run_hand = run_smem = false;
                selection = "kernel checks only";
            } else if (std::strcmp(what, "smem") == 0) {
                run_hand = run_kernels = false;
                selection = "shared-memory assignment only";
            } else if (std::strcmp(what, "all") != 0) {
                std::fprintf(stderr,
                             "error[VAL-010]: --only expects hand, kernels, smem or all\n");
                return kExitInvalidInput;
            }
        } else {
            std::fprintf(stderr,
                         "usage: test_gpu [--only {all|hand|kernels|smem}] [--quick]\n");
            return kExitInvalidInput;
        }
    }

    std::printf("M2-M4 GPU test suite (%s, %s)\n", quick ? "quick" : "full", selection);
    std::printf("requirements.md sections 6, 7, 8, 16.0, 16.1, 16.2, 10.3\n\n");

    // The device is reported once, from the same query path the driver uses (CUDA-001).
    try {
        const DeviceInfo info = query_device(0);
        std::printf("device: %s\n", describe_device(info).c_str());
        std::printf("        CUDA runtime %d, driver %d, native atomicAdd(double): %s\n\n",
                    info.runtime_version, info.driver_version,
                    info.has_native_fp64_atomics() ? "yes" : "no");
    } catch (const CudaError &err) {
        std::fprintf(stderr, "error[CUDA-001]: %s\n", err.what());
        return err.exit_code();
    }

    const Strategy strategies[] = {Strategy::kAtomic, Strategy::kReduction};

    if (run_hand) {
        std::printf("TEST-I01 / TEST-I02: H1-H6 and H2-a..e (section 16.0)\n");
        for (Strategy strategy : strategies) {
            test_hand_cases(strategy);
        }
    }

    if (run_kernels) {
        std::printf("\nTEST-U06: buffer initialisation under poisoning\n");
        for (Strategy strategy : strategies) {
            test_poisoning(strategy);
        }

        std::printf("\nTEST-I10 / section 10.3: lockstep against the CPU reference\n");
        {
            const GenParams params = lockstep_params();
            const Dataset data = generate(params);
            std::printf("  dataset %s, minimum centre distance %.3f (%.1f sigma)\n",
                        data.dataset_id.c_str(), data.min_centre_distance,
                        data.spread > 0.0 ? data.min_centre_distance / data.spread : 0.0);
            for (Strategy strategy : strategies) {
                test_lockstep(strategy, AssignKernel::kBaseline, data, params);
            }
        }

        if (!quick) {
            std::printf("\nTEST-U05: reduction accumulation over the G sweep (RED-006)\n");
            test_chunk_sweep();

            std::printf("\nTEST-U07: partial Stage 2 tiles and empty chunks (RED-003, EDGE-35)\n");
            test_partial_tiles();

            std::printf("\nTEST-E16: non-aligned N, D, K (CUDA-012, EDGE-16)\n");
            for (Strategy strategy : strategies) {
                test_non_aligned(strategy);
            }
        } else {
            std::printf("\nskipped in --quick: TEST-U05 (G sweep), TEST-U07 (partial tiles), "
                        "TEST-E16 (non-aligned sizes)\n");
        }
    }

    if (run_smem) {
        std::printf("\nSMEM-004: assign_smem against the baseline, T_K = %d, T_D = %d, "
                    "%lld B of shared memory per block (section 8)\n",
                    kAssignTileK, kAssignTileD,
                    static_cast<long long>(assign_smem_shared_bytes(kAssignTileK, kAssignTileD,
                                                                    Config().block_size)));
        for (Strategy strategy : strategies) {
            test_smem_hand_cases(strategy);
        }
        test_smem_datasets(quick);

        if (!quick) {
            test_smem_non_aligned();

            std::printf("\nTEST-U01 / section 10.3: assign_smem in lockstep against the CPU "
                        "reference\n");
            const GenParams params = lockstep_params();
            const Dataset data = generate(params);
            for (Strategy strategy : strategies) {
                test_lockstep(strategy, AssignKernel::kSharedMemory, data, params);
            }
        } else {
            std::printf("\nskipped in --quick: the SMEM-004 size matrix and the assign_smem "
                        "lockstep run\n");
        }
    }

    return tests::report_totals("all GPU-level tests in this run passed");
}
