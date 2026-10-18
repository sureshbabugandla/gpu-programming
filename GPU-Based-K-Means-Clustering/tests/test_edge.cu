// test_edge.cu — the §9.2 corner-case matrix, cross-strategy comparison and reproducibility.
//
// The cases that need a device and a CPU reference live here; the ones that are decided before any
// device work (EDGE-01..03, 20, 21 and the fault-injection paths EDGE-27/28) are exit-code checks
// against the `kmeans` executable and are driven from the notebook's test-suite cell, because what
// they assert is the process exit code and the BLD-006 message, not an in-process value.
//
//   TEST-E04..E06, E08..E12, E15   §9.2 structural cases, device against the CPU reference
//   TEST-E18 / E19                 magnitudes near the VAL-007 bound; near-tie resolution
//   TEST-E26 / E29 / E30           VAL-009 and the overflow-checked size arithmetic, exit 3 / 2
//   TEST-E34                       a tile configuration over the per-block limit, exit 2
//   TEST-U02                       tie-breaking on the device, both assignment kernels
//   TEST-I09                       Approach A against Approach B, iteration by iteration
//   TEST-I11                       well-separated data end to end, exact against the reference
//   TEST-R01 / TEST-R02            repeated execution: bitwise for Approach B, reported for A
//
// One executable, non-zero exit on any failure (NFR-015). No result here is evidence of anything
// until an executed notebook in results/ shows it (CLAUDE.md).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "cpu_reference.h"
#include "cuda_util.cuh"
#include "device_context.cuh"
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
using kmeans::tests::check_rejected;
using kmeans::tests::double_str;
using kmeans::tests::float_str;
using kmeans::tests::run_device;
using kmeans::tests::strategy_name;
using kmeans::tests::TestScope;

const Strategy kStrategies[] = {Strategy::kAtomic, Strategy::kReduction};
const AssignKernel kAssignKernels[] = {AssignKernel::kBaseline, AssignKernel::kSharedMemory};

// --- a corner case and the comparison every one of them gets -----------------------------------

struct EdgeInput {
    std::string tag;
    int64_t N = 0;
    int64_t D = 0;
    int64_t K = 0;
    int64_t max_iter = 10;
    double tol = 0.0;
    std::vector<float> X;   // N x D row-major
    std::vector<float> C0;  // K x D row-major
};

struct EdgeOutcome {
    bool ran = false;
    Result device;
    Result reference;
    double m_range = 0.0;
    Validation validation;
};

// Runs one case on the device and on the CPU reference with identical inputs and checks that they
// agree: assignments, counts, iteration count, status and the P2 flag exactly, centroids within
// NUM-006. §10.3 method 2 is what makes exactness the right criterion for these cases -- they are
// constructed, not sampled, so no near-tie is present unless the case is about near-ties.
EdgeOutcome run_and_compare(const EdgeInput &in, Strategy strategy, AssignKernel assign)
{
    EdgeOutcome out;
    out.validation = validate_data(in.X, in.N, in.D, in.C0, in.K, &out.m_range);
    check(out.validation.ok,
          in.tag + ": data accepted" + (out.validation.ok ? "" : ": " + out.validation.message));
    if (!out.validation.ok) {
        return out;
    }

    gpu::Options options;
    options.config.N = in.N;
    options.config.D = in.D;
    options.config.K = in.K;
    options.config.max_iter = in.max_iter;
    options.config.tol = in.tol;
    options.config.strategy = strategy;
    options.config.assign_kernel = assign;
    options.m_range = out.m_range;
    options.verify = gpu::Verify::kFinal;

    gpu::RunInfo info;
    if (!run_device(in.X.data(), in.C0.data(), options, &out.device, &info, gpu::LockstepHook(),
                    in.tag + ": device run")) {
        return out;
    }
    out.reference = cpu::run(in.X.data(), in.N, in.D, in.C0.data(), in.K, in.max_iter, in.tol,
                             out.m_range, false);
    out.ran = true;

    check(out.device.status == out.reference.status,
          in.tag + ": status: device " + status_name(out.device.status) + ", reference " +
              status_name(out.reference.status));
    check_eq_i64(out.device.iterations, out.reference.iterations, in.tag + ": iteration count");
    check(out.device.centroid_consistent == out.reference.centroid_consistent,
          in.tag + ": P2 flag matches the reference");
    check(out.device.assignment_consistent, in.tag + ": P1 flag true by construction");

    for (int64_t i = 0; i < in.N; ++i) {
        const size_t e = static_cast<size_t>(i);
        if (out.device.assignments[e] != out.reference.assignments[e]) {
            check(false, in.tag + ": A[" + std::to_string(i) + "]: device " +
                             std::to_string(out.device.assignments[e]) + ", reference " +
                             std::to_string(out.reference.assignments[e]));
            break;  // one named point is enough to locate the fault
        }
    }
    for (int64_t k = 0; k < in.K; ++k) {
        check_eq_i64(out.device.counts[static_cast<size_t>(k)],
                     out.reference.counts[static_cast<size_t>(k)],
                     in.tag + ": n[" + std::to_string(k) + "]");
    }
    const double s_max = s_max_from_range(out.m_range);
    for (int64_t e = 0; e < in.K * in.D; ++e) {
        const size_t idx = static_cast<size_t>(e);
        check(centroid_agrees(out.device.centroids[idx], out.reference.centroids[idx], s_max),
              in.tag + ": C[" + std::to_string(e) + "]: device " +
                  float_str(out.device.centroids[idx]) + ", reference " +
                  float_str(out.reference.centroids[idx]) + " (NUM-006)");
        check(std::isfinite(out.device.centroids[idx]),
              in.tag + ": C[" + std::to_string(e) + "] is finite");
    }

    // FR-009: every point lands in exactly one cluster.
    uint64_t total = 0;
    for (uint32_t n : out.device.counts) {
        total += n;
    }
    check(total == static_cast<uint64_t>(in.N),
          in.tag + ": counts sum to N: got " + std::to_string(total) + ", want " +
              std::to_string(in.N));
    return out;
}

// Every empty cluster kept its initial centroid bit-exactly (DEC-003). Only sound for cases in
// which an empty cluster is never re-populated, which is true of all the constructed cases below.
void check_empty_clusters_retained(const EdgeInput &in, const Result &r)
{
    for (int64_t k = 0; k < in.K; ++k) {
        if (r.counts[static_cast<size_t>(k)] != 0u) {
            continue;
        }
        for (int64_t j = 0; j < in.D; ++j) {
            const size_t e = static_cast<size_t>(k * in.D + j);
            check(std::memcmp(&r.centroids[e], &in.C0[e], sizeof(float)) == 0,
                  in.tag + ": empty cluster " + std::to_string(k) + " retained C0[" +
                      std::to_string(j) + "] bit-exactly (DEC-003)");
        }
    }
}

// Runs a case through every (strategy, assign kernel) pair. The §9.2 behaviour must not depend on
// either choice: FR-018 puts assignment, update, convergence and finalization on one code path,
// and SMEM-004 makes the two assignment kernels interchangeable.
template <typename Extra>
void for_each_variant(const EdgeInput &in, const char *id, const char *title, Extra &&extra)
{
    for (Strategy strategy : kStrategies) {
        for (AssignKernel assign : kAssignKernels) {
            TestScope scope(id, std::string(strategy_name(strategy)) + " / " +
                                    assign_name(assign) + "  " + title);
            const EdgeOutcome out = run_and_compare(in, strategy, assign);
            if (out.ran) {
                extra(in, out);
            }
        }
    }
}

// --- EDGE-04: K > N ----------------------------------------------------------------------------

void test_e04_k_greater_than_n()
{
    EdgeInput in;
    in.tag = "E04";
    in.N = 3;
    in.D = 2;
    in.K = 7;
    in.X = {0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 10.0f};
    // Seven distinct centroids for three points: at least four must stay empty.
    in.C0 = {0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 10.0f, 20.0f, 20.0f,
             30.0f, 30.0f, 40.0f, 40.0f, 50.0f, 50.0f};

    // VAL-003 is a parameter rule, so it is checked once rather than per variant.
    {
        const Validation v = validate_parameters(in.N, in.D, in.K, in.max_iter, in.tol);
        TestScope scope("E04", "VAL-003 accepts K > N with the W-KGTN warning");
        check(v.ok, "K > N is accepted, not rejected");
        check(v.has_warning("W-KGTN"), "the W-KGTN warning is raised");
    }

    for_each_variant(in, "E04", "K > N: at least K - N clusters stay empty and retain C0",
                     [](const EdgeInput &input, const EdgeOutcome &out) {
                         int64_t empty = 0;
                         for (uint32_t n : out.device.counts) {
                             if (n == 0u) {
                                 ++empty;
                             }
                         }
                         check(empty >= input.K - input.N,
                               "at least K - N = " + std::to_string(input.K - input.N) +
                                   " clusters are empty: got " + std::to_string(empty));
                         check_empty_clusters_retained(input, out.device);
                     });
}

// --- EDGE-05: N = K with C0 = X ----------------------------------------------------------------

void test_e05_n_equals_k()
{
    EdgeInput in;
    in.tag = "E05";
    in.N = 5;
    in.D = 3;
    in.K = 5;
    // Distinct points, well apart, and C0 = X: every point is already its own nearest centroid,
    // so the first update leaves the centroids where they are and §2.7 stops at t = 1.
    in.X = {0.0f,  0.0f,  0.0f,  10.0f, 0.0f,  0.0f, 0.0f, 10.0f, 0.0f,
            0.0f,  0.0f,  10.0f, 10.0f, 10.0f, 10.0f};
    in.C0 = in.X;

    for_each_variant(in, "E05", "N = K with C0 = X: each point its own cluster at t = 1",
                     [](const EdgeInput &input, const EdgeOutcome &out) {
                         check(out.device.status == Status::CONVERGED_ASSIGNMENTS,
                               std::string("status is CONVERGED_ASSIGNMENTS, got ") +
                                   status_name(out.device.status));
                         check_eq_i64(out.device.iterations, 1, "t = 1");
                         for (int64_t k = 0; k < input.K; ++k) {
                             check_eq_i64(out.device.counts[static_cast<size_t>(k)], 1,
                                          "n[" + std::to_string(k) + "] = 1");
                         }
                         // The centroid of a one-point cluster is that point exactly: the sum is
                         // one binary32 value and the division is by 1 (EDGE-15).
                         for (int64_t e = 0; e < input.K * input.D; ++e) {
                             const size_t idx = static_cast<size_t>(e);
                             check(out.device.centroids[idx] == input.X[idx],
                                   "C[" + std::to_string(e) + "] equals the point exactly");
                         }
                     });
}

// --- EDGE-06: N = 1, and EDGE-15 as its degenerate case ----------------------------------------

void test_e06_single_point()
{
    EdgeInput in;
    in.tag = "E06";
    in.N = 1;
    in.D = 4;
    in.K = 3;
    in.X = {1.0f, 2.0f, 3.0f, 4.0f};
    in.C0 = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 5.0f, 100.0f, 100.0f, 100.0f, 100.0f};

    for_each_variant(in, "E06", "N = 1: the point joins its nearest centroid at t = 1",
                     [](const EdgeInput &input, const EdgeOutcome &out) {
                         check_eq_i64(out.device.iterations, 1, "t = 1");
                         int64_t ones = 0, zeros = 0;
                         for (uint32_t n : out.device.counts) {
                             ones += (n == 1u) ? 1 : 0;
                             zeros += (n == 0u) ? 1 : 0;
                         }
                         check_eq_i64(ones, 1, "exactly one cluster holds the point");
                         check_eq_i64(zeros, input.K - 1, "every other cluster is empty");
                         check(out.device.centroid_consistent,
                               "P2 holds: a single-point cluster's centroid is the point");
                         // EDGE-15: bit-exact, because the division is by 1.
                         const int32_t k = out.device.assignments[0];
                         for (int64_t j = 0; j < input.D; ++j) {
                             const size_t e = static_cast<size_t>(k * input.D + j);
                             check(out.device.centroids[e] == input.X[static_cast<size_t>(j)],
                                   "C[" + std::to_string(k) + "][" + std::to_string(j) +
                                       "] is the point, bit-exactly (EDGE-15)");
                         }
                         check_empty_clusters_retained(input, out.device);
                     });
}

// --- EDGE-08: K = 1 ----------------------------------------------------------------------------

void test_e08_single_cluster()
{
    EdgeInput in;
    in.tag = "E08";
    in.N = 8;
    in.D = 2;
    in.K = 1;
    // Coordinates whose mean is exactly representable: the sum is 36 and 36/8 = 4.5.
    in.X = {1.0f, 8.0f, 2.0f, 7.0f, 3.0f, 6.0f, 4.0f, 5.0f,
            5.0f, 4.0f, 6.0f, 3.0f, 7.0f, 2.0f, 12.0f, 1.0f};
    in.C0 = {0.0f, 0.0f};

    for_each_variant(in, "E08", "K = 1: every point in cluster 0, centroid = global mean",
                     [](const EdgeInput &input, const EdgeOutcome &out) {
                         for (int64_t i = 0; i < input.N; ++i) {
                             check_eq_i64(out.device.assignments[static_cast<size_t>(i)], 0,
                                          "A[" + std::to_string(i) + "] = 0");
                         }
                         check_eq_i64(out.device.counts[0], input.N, "n[0] = N");
                         check(out.device.status == Status::CONVERGED_ASSIGNMENTS,
                               std::string("status is CONVERGED_ASSIGNMENTS, got ") +
                                   status_name(out.device.status));
                         check_eq_i64(out.device.iterations, 1, "t = 1");
                         // The mean computed independently here, in binary64, as §2.5 does.
                         for (int64_t j = 0; j < input.D; ++j) {
                             double sum = 0.0;
                             for (int64_t i = 0; i < input.N; ++i) {
                                 sum += static_cast<double>(
                                     input.X[static_cast<size_t>(i * input.D + j)]);
                             }
                             const float mean =
                                 static_cast<float>(sum / static_cast<double>(input.N));
                             check(centroid_agrees(out.device.centroids[static_cast<size_t>(j)],
                                                   mean, s_max_from_range(out.m_range)),
                                   "C[0][" + std::to_string(j) + "] is the global mean: got " +
                                       float_str(out.device.centroids[static_cast<size_t>(j)]) +
                                       ", want " + float_str(mean));
                         }
                     });
}

// --- EDGE-09: duplicate input points -----------------------------------------------------------

void test_e09_duplicate_points()
{
    EdgeInput in;
    in.tag = "E09";
    in.N = 8;
    in.D = 2;
    in.K = 3;
    // Four distinct coordinates, each appearing twice, in an order that puts duplicates in
    // different warps of nothing in particular -- the point is only that they are not adjacent.
    const std::vector<float> p[] = {{0.0f, 0.0f}, {9.0f, 1.0f}, {1.0f, 9.0f}, {9.0f, 9.0f}};
    const int order[] = {0, 1, 2, 3, 3, 2, 1, 0};
    for (int idx : order) {
        in.X.push_back(p[idx][0]);
        in.X.push_back(p[idx][1]);
    }
    in.C0 = {0.0f, 0.0f, 9.0f, 1.0f, 5.0f, 9.0f};

    for_each_variant(in, "E09", "duplicate points always share a cluster",
                     [](const EdgeInput &input, const EdgeOutcome &out) {
                         for (int64_t i = 0; i < input.N; ++i) {
                             for (int64_t j = i + 1; j < input.N; ++j) {
                                 const bool same_point =
                                     std::memcmp(&input.X[static_cast<size_t>(i * input.D)],
                                                 &input.X[static_cast<size_t>(j * input.D)],
                                                 static_cast<size_t>(input.D) * sizeof(float)) == 0;
                                 if (!same_point) {
                                     continue;
                                 }
                                 check_eq_i64(out.device.assignments[static_cast<size_t>(i)],
                                              out.device.assignments[static_cast<size_t>(j)],
                                              "points " + std::to_string(i) + " and " +
                                                  std::to_string(j) +
                                                  " are identical and must share a cluster");
                             }
                         }
                     });
}

// --- EDGE-10: duplicate initial centroids ------------------------------------------------------

void test_e10_duplicate_centroids()
{
    EdgeInput in;
    in.tag = "E10";
    in.N = 4;
    in.D = 2;
    in.K = 3;
    in.X = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    // Centroids 0 and 1 are identical. While they stay identical the tie rule sends every point
    // of that pair to index 0, so centroid 1 receives nothing and retains its value (DEC-005).
    in.C0 = {0.0f, 0.0f, 0.0f, 0.0f, 100.0f, 100.0f};

    {
        double m_range = 0.0;
        const Validation v = validate_data(in.X, in.N, in.D, in.C0, in.K, &m_range);
        TestScope scope("E10", "VAL-008 accepts duplicate centroids with the W-DUPC warning");
        check(v.ok, "duplicate centroids are accepted, not rejected");
        check(v.has_warning("W-DUPC"), "the W-DUPC warning is raised");
    }

    // The §9.2 criterion is about the state while the two centroids are still identical, which is
    // the initial assignment: as soon as cluster 0 is updated to the mean of its points the two
    // are no longer identical and cluster 1 may legitimately start receiving points. max_iter = 0
    // reports A(0) (§2.7), so this judges the tie rule and nothing else.
    in.max_iter = 0;
    for_each_variant(in, "E10", "duplicate C0 at t = 0: the lower index takes the points",
                     [](const EdgeInput &input, const EdgeOutcome &out) {
                         check_eq_i64(out.device.counts[1], 0,
                                      "the higher-index duplicate receives no points");
                         check(out.device.counts[0] > 0,
                               "the lower-index duplicate receives them: n[0] = " +
                                   std::to_string(out.device.counts[0]));
                         check_empty_clusters_retained(input, out.device);
                     });

    // And the full run must still track the CPU reference once the duplicates diverge, which is
    // the other half of the §9.2 acceptance criterion ("matches CPU").
    in.max_iter = 10;
    for_each_variant(in, "E10", "duplicate C0 over a full run: matches the CPU reference",
                     [](const EdgeInput &, const EdgeOutcome &) {});
}

// --- EDGE-11: all input points identical -------------------------------------------------------

void test_e11_identical_points()
{
    EdgeInput in;
    in.tag = "E11";
    in.N = 64;
    in.D = 3;
    in.K = 4;
    for (int64_t i = 0; i < in.N; ++i) {
        in.X.insert(in.X.end(), {2.0f, 2.0f, 2.0f});
    }
    // Centroids 1 and 2 are equidistant from the point; the lowest index of the two must win.
    in.C0 = {10.0f, 10.0f, 10.0f, 0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 4.0f, 20.0f, 20.0f, 20.0f};

    for_each_variant(in, "E11", "all points identical: all to the lowest-index nearest centroid",
                     [](const EdgeInput &input, const EdgeOutcome &out) {
                         const int32_t first = out.device.assignments[0];
                         check_eq_i64(first, 1,
                                      "the tie between centroids 1 and 2 goes to index 1");
                         for (int64_t i = 1; i < input.N; ++i) {
                             check_eq_i64(out.device.assignments[static_cast<size_t>(i)], first,
                                          "A[" + std::to_string(i) + "] equals A[0]");
                         }
                         check_eq_i64(out.device.counts[static_cast<size_t>(first)], input.N,
                                      "the winning cluster holds every point");
                         check_eq_i64(out.device.iterations, 1, "t = 1");
                         check_empty_clusters_retained(input, out.device);
                     });
}

// --- EDGE-12: all centroids identical ----------------------------------------------------------

void test_e12_identical_centroids()
{
    EdgeInput in;
    in.tag = "E12";
    in.N = 33;  // not a multiple of the warp or block size, on purpose
    in.D = 2;
    in.K = 5;
    for (int64_t i = 0; i < in.N; ++i) {
        in.X.push_back(static_cast<float>(i));
        in.X.push_back(static_cast<float>(-i));
    }
    for (int64_t k = 0; k < in.K; ++k) {
        in.C0.insert(in.C0.end(), {7.0f, -7.0f});
    }

    // As for EDGE-10, the criterion A == 0 and n_0 == N describes the state while the centroids
    // are still identical. max_iter = 0 reports exactly that state.
    in.max_iter = 0;
    for_each_variant(in, "E12", "all centroids identical at t = 0: everything to cluster 0",
                     [](const EdgeInput &input, const EdgeOutcome &out) {
                         for (int64_t i = 0; i < input.N; ++i) {
                             check_eq_i64(out.device.assignments[static_cast<size_t>(i)], 0,
                                          "A[" + std::to_string(i) + "] = 0");
                         }
                         check_eq_i64(out.device.counts[0], input.N, "n[0] = N");
                         for (int64_t k = 1; k < input.K; ++k) {
                             check_eq_i64(out.device.counts[static_cast<size_t>(k)], 0,
                                          "n[" + std::to_string(k) + "] = 0");
                         }
                         check_empty_clusters_retained(input, out.device);
                     });

    in.max_iter = 10;
    for_each_variant(in, "E12",
                     "all centroids identical over a full run: matches the CPU reference",
                     [](const EdgeInput &, const EdgeOutcome &) {});
}

// --- EDGE-18: magnitudes near the VAL-007 bound ------------------------------------------------

void test_e18_large_magnitudes()
{
    // VAL-007 requires 4 D M^2 <= 2^126. With D = 8 that is M <= 2^(123/2) ~= 1.0e18; the
    // generator below produces coordinates up to roughly 4 sigma of sep, so sep = 1e17 lands
    // close under the bound without crossing it, which is what EDGE-18 asks for. The point of the
    // case is that no intermediate binary32 distance overflows to infinity.
    GenParams p;
    p.N = 20000;
    p.D = 8;
    p.K = 6;
    p.sep = 1.0e17;
    p.spread = 1.0e15;
    p.seed = 180018;
    const Dataset data = generate(p);

    EdgeInput in;
    in.tag = "E18";
    in.N = p.N;
    in.D = p.D;
    in.K = p.K;
    in.max_iter = 10;
    in.X = data.X;
    in.C0 = data.C0;

    {
        double m_range = 0.0;
        const Validation v = validate_data(in.X, in.N, in.D, in.C0, in.K, &m_range);
        TestScope scope("E18", "VAL-007 accepts data close under the 4 D M^2 <= 2^126 bound");
        check(v.ok, "accepted: " + v.message);
        check(m_range > 1.0e17,
              "M is genuinely large: " + double_str(m_range));
        // 2^126, the bound VAL-007 evaluates in binary64.
        const double bound = 85070591730234615865843651857942052864.0;
        check(4.0 * static_cast<double>(in.D) * m_range * m_range <= bound,
              "4 D M^2 = " + double_str(4.0 * static_cast<double>(in.D) * m_range * m_range) +
                  " is within 2^126");
    }

    for (Strategy strategy : kStrategies) {
        TestScope scope("E18", std::string(strategy_name(strategy)) +
                                   "  magnitudes near the VAL-007 bound: no overflow");
        const EdgeOutcome out = run_and_compare(in, strategy, AssignKernel::kBaseline);
        if (!out.ran) {
            continue;
        }
        check(std::isfinite(out.device.objective) && out.device.objective > 0.0,
              "the objective is finite: " + double_str(out.device.objective));
        check(std::isfinite(out.device.last_delta), "Delta is finite");
    }
}

// --- EDGE-19: coordinate differences below binary32 resolution ---------------------------------

void test_e19_tiny_differences()
{
    // Two centroids 1e-6 apart, and points on the midpoint perturbed by ~1e-14 -- below the
    // binary32 ulp of 5e-7, which is about 3e-14. The perturbations are therefore partly invisible
    // to the binary32 distance path, which is exactly EDGE-19: such points are near-ties by
    // construction, and NUM-005 permits the device and the FP64 oracle to disagree on them while
    // forbidding any disagreement elsewhere.
    EdgeInput in;
    in.tag = "E19";
    in.N = 2048;
    in.D = 1;
    in.K = 2;
    in.max_iter = 5;
    Rng rng(190019);
    for (int64_t i = 0; i < in.N; ++i) {
        const double jitter = (rng.uniform_open() - 0.5) * 4.0e-14;
        in.X.push_back(static_cast<float>(5.0e-7 + jitter));
    }
    in.C0 = {0.0f, 1.0e-6f};

    for (Strategy strategy : kStrategies) {
        for (AssignKernel assign : kAssignKernels) {
            TestScope scope("E19", std::string(strategy_name(strategy)) + " / " +
                                       assign_name(assign) +
                                       "  differences below binary32 resolution");
            double m_range = 0.0;
            const Validation v = validate_data(in.X, in.N, in.D, in.C0, in.K, &m_range);
            check(v.ok, "data accepted");
            if (!v.ok) {
                continue;
            }

            gpu::Options options;
            options.config.N = in.N;
            options.config.D = in.D;
            options.config.K = in.K;
            options.config.max_iter = in.max_iter;
            options.config.tol = in.tol;
            options.config.strategy = strategy;
            options.config.assign_kernel = assign;
            options.m_range = m_range;
            options.verify = gpu::Verify::kFinal;

            Result r;
            gpu::RunInfo info;
            if (!run_device(in.X.data(), in.C0.data(), options, &r, &info, gpu::LockstepHook(),
                            "E19 device run")) {
                continue;
            }

            // The FR-019 `final` check is exactly the NUM-005 rule: every reported assignment is
            // the nearest reported centroid, judged with the binary32 path, and disagreements with
            // the FP64 oracle are failures unless the point is a near-tie.
            const gpu::FinalVerification fv =
                gpu::verify_final(in.X.data(), r, options.config, m_range);
            check_eq_i64(fv.p1_hard_mismatches, 0,
                         "no non-near-tie P1 mismatch (NUM-005)");
            check(fv.p2_flag_correct, "the P2 flag agrees with the reference finalization");
            std::printf("      near-tie points %lld of %lld, threshold %.3e; "
                        "status %s, t = %lld\n",
                        static_cast<long long>(fv.p1_near_tie_mismatches),
                        static_cast<long long>(in.N), near_tie_threshold(in.D),
                        status_name(r.status), static_cast<long long>(r.iterations));
        }
    }
}

// --- EDGE-26, EDGE-29, EDGE-30: the pre-checks that run before any allocation -------------------

void test_pre_allocation_rejections(const DeviceInfo &device)
{
    {
        TestScope scope("E26", "VAL-009 rejects a configuration larger than free memory, exit 3");
        Config cfg;
        // 8ND alone is about 512 GB, which no current device has.
        cfg.N = 2000000000;
        cfg.D = 32;
        cfg.K = 64;
        cfg.max_iter = 1;
        // The substring is the one VAL-009 promises: the message states the required and the
        // available byte counts, not merely that the configuration does not fit.
        check_rejected([&]() { check_capacity(device, cfg, 0); }, kExitUnsupportedDevice,
                       "an oversized configuration is rejected", "B exceeds the usable");
    }

    {
        TestScope scope("E29", "a G override beyond N is rejected before allocation, exit 3");
        Config cfg;
        cfg.N = 100000;
        cfg.D = 16;
        cfg.K = 8;
        cfg.strategy = Strategy::kReduction;
        cfg.reduce_chunks = 100001;  // DEC-001 requires 1 <= G <= N
        check_rejected([&]() { select_chunks(device, cfg, 0); }, kExitUnsupportedDevice,
                       "G > N is rejected", "RED-007");

        // And a G that fits DEC-001 but not the MEM-003 budget.
        Config big;
        big.N = 1000000;
        big.D = 512;
        big.K = 100000;
        big.strategy = Strategy::kReduction;
        big.reduce_chunks = 1000;  // 8GKD = 4 x 10^14 bytes
        check_rejected([&]() { select_chunks(device, big, 0); }, kExitUnsupportedDevice,
                       "a G beyond the MEM-003 budget is rejected", "RED-007");
    }

    {
        TestScope scope("E30", "overflow in the size arithmetic is detected, not wrapped");
        Config cfg;
        // N*D = 2^62, so 4ND overflows 64 bits and must be reported rather than wrapped into a
        // small, plausible-looking byte count (CUDA-006).
        cfg.N = 2147483647;
        cfg.D = 2147483647;
        cfg.K = 2;
        check_rejected([&]() { check_capacity(device, cfg, 0); }, kExitInvalidInput,
                       "an overflowing configuration is rejected", "overflows");
    }

    {
        TestScope scope("E34", "a tile configuration over the per-block limit is rejected, exit 2");
        // 4 T_K T_D with T_K = 4096 is 512 KiB, far past any device's per-block limit.
        check_rejected([&]() { check_shared_memory(device, 256, 4096, kAssignTileD); },
                       kExitInvalidInput, "an oversized tile is rejected at startup", "SMEM-003");
        // VAL-010 reaches the same verdict without a device, which is what the host-side test in
        // test_cpu.cpp checks; here it is checked against the real limit.
        const Validation v =
            validate_options(256, 0, 4096, kAssignTileD, device.max_threads_per_block,
                             static_cast<int64_t>(device.shared_mem_per_block));
        check(!v.ok, "VAL-010 also rejects it");
        check_eq_i64(v.exit_code, kExitInvalidInput, "with exit code 2");

        // The configuration this binary was built with must of course pass.
        const Validation ok =
            validate_options(256, 0, kAssignTileK, kAssignTileD, device.max_threads_per_block,
                             static_cast<int64_t>(device.shared_mem_per_block));
        check(ok.ok, "the configured tile T_K = " + std::to_string(kAssignTileK) + ", T_D = " +
                         std::to_string(kAssignTileD) + " is accepted: " + ok.message);
    }
}

// --- EDGE-29 second half: a large configuration that does run ----------------------------------

void test_e29_large_run()
{
    GenParams p;
    p.N = 2000000;
    p.D = 32;
    p.K = 64;
    p.sep = 30.0;
    p.spread = 1.0;
    p.seed = 290029;
    const Dataset data = generate(p);

    double m_range = 0.0;
    const Validation v = validate_data(data.X, p.N, p.D, data.C0, p.K, &m_range);

    for (Strategy strategy : kStrategies) {
        TestScope scope("E29", std::string(strategy_name(strategy)) + "  N = 2e6, D = 32, K = 64 "
                                                                      "runs and verifies");
        check(v.ok, "data accepted");
        if (!v.ok) {
            continue;
        }
        gpu::Options options;
        options.config.N = p.N;
        options.config.D = p.D;
        options.config.K = p.K;
        options.config.max_iter = 10;
        options.config.strategy = strategy;
        options.m_range = m_range;
        options.verify = gpu::Verify::kFinal;

        Result r;
        gpu::RunInfo info;
        if (!run_device(data.X.data(), data.C0.data(), options, &r, &info, gpu::LockstepHook(),
                        "E29 large run")) {
            continue;
        }
        const gpu::FinalVerification fv =
            gpu::verify_final(data.X.data(), r, options.config, m_range);
        check_eq_i64(fv.p1_hard_mismatches, 0, "P1 holds (NUM-005)");
        check(fv.p2_flag_correct, "the P2 flag is correct");
        std::printf("      status %s, t = %lld, T_alg %.1f ms, %llu B of device memory\n",
                    status_name(r.status), static_cast<long long>(r.iterations), info.t_alg_ms,
                    static_cast<unsigned long long>(
                        device_memory_bytes(p.N, p.D, p.K, info.reduce_chunks).total));
    }
}

// --- TEST-U02: tie-breaking on the device ------------------------------------------------------

void test_u02_tie_breaking()
{
    struct TieCase {
        std::string tag;
        int64_t N, D, K;
        std::vector<float> X;
        std::vector<float> C0;
        std::vector<int32_t> expected_A;
    };

    std::vector<TieCase> cases;

    // H3: the single point sits exactly between the two centroids (§16.0).
    cases.push_back({"H3", 1, 1, 2, {1.0f}, {0.0f, 2.0f}, {0}});

    // Points on the perpendicular bisector of two integer centroids. Every coordinate here is
    // exactly representable in binary32 and the two squared distances are formed from the same
    // terms, so the computed values are exactly equal and DEC-005 must decide.
    {
        TieCase c;
        c.tag = "bisector x = 1 between (0,0) and (2,0)";
        c.N = 5;
        c.D = 2;
        c.K = 2;
        const float ys[] = {0.0f, 1.0f, -1.0f, 4.0f, -16.0f};
        for (float y : ys) {
            c.X.push_back(1.0f);
            c.X.push_back(y);
        }
        c.C0 = {0.0f, 0.0f, 2.0f, 0.0f};
        c.expected_A.assign(static_cast<size_t>(c.N), 0);
        cases.push_back(c);
    }

    // A point equidistant from all four corners of a square: the lowest of four tied indices.
    cases.push_back({"centre of four corners", 1, 2, 4, {1.0f, 1.0f},
                     {0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 2.0f, 2.0f}, {0}});

    // Duplicate centroids: the tie is between two identical centroids, so the rule has to pick the
    // lower index for every point (EDGE-10 at the kernel level).
    {
        TieCase c;
        c.tag = "duplicate centroids";
        c.N = 4;
        c.D = 2;
        c.K = 4;
        c.X = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f};
        c.C0 = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
        c.expected_A.assign(static_cast<size_t>(c.N), 0);
        cases.push_back(c);
    }

    for (const TieCase &c : cases) {
        for (AssignKernel assign : kAssignKernels) {
            TestScope scope("U02", std::string(assign_name(assign)) + "  " + c.tag +
                                       ": the lowest tied index wins (DEC-005)");
            // max_iter = 0: §2.7 reports A(0), so this judges the assignment kernel alone, with
            // no update, no convergence test and no finalization in the way.
            gpu::Options options;
            options.config.N = c.N;
            options.config.D = c.D;
            options.config.K = c.K;
            options.config.max_iter = 0;
            options.config.assign_kernel = assign;
            options.m_range = 0.0;

            double m_range = 0.0;
            const Validation v = validate_data(c.X, c.N, c.D, c.C0, c.K, &m_range);
            check(v.ok, "data accepted");
            options.m_range = m_range;

            Result r;
            gpu::RunInfo info;
            if (!run_device(c.X.data(), c.C0.data(), options, &r, &info, gpu::LockstepHook(),
                            c.tag + ": device run")) {
                continue;
            }
            check_eq_i64(r.iterations, 0, "t = 0: the initial assignment pass only");
            for (int64_t i = 0; i < c.N; ++i) {
                check_eq_i64(r.assignments[static_cast<size_t>(i)],
                             c.expected_A[static_cast<size_t>(i)],
                             "A[" + std::to_string(i) + "]");
            }
            // The CPU reference applies the same rule to the same values.
            std::vector<int32_t> a_ref(static_cast<size_t>(c.N), -1);
            cpu::assign(c.X.data(), c.N, c.D, c.C0.data(), c.K, a_ref.data(), nullptr);
            for (int64_t i = 0; i < c.N; ++i) {
                check_eq_i64(r.assignments[static_cast<size_t>(i)],
                             a_ref[static_cast<size_t>(i)],
                             "A[" + std::to_string(i) + "] matches the CPU reference");
            }
        }
    }
}

// --- TEST-I09: Approach A against Approach B ---------------------------------------------------

// Both strategies are run with a lockstep hook that records every snapshot. The comparison then
// walks the two sequences while they were fed the same A(t): that is the "lockstep accumulation"
// of the §16.7 matrix, and it is the only way to attribute a difference to the accumulation rather
// than to two trajectories that have already diverged (EDGE-32).
void test_i09_cross_strategy()
{
    GenParams p;
    p.N = 50000;
    p.D = 16;
    p.K = 32;
    p.sep = 10.0;
    p.spread = 1.0;
    p.seed = 90009;
    const Dataset data = generate(p);

    TestScope scope("I09", "Approach A against Approach B, accumulation by accumulation "
                           "(ATOM-007, RED-009)");

    double m_range = 0.0;
    const Validation v = validate_data(data.X, p.N, p.D, data.C0, p.K, &m_range);
    check(v.ok, "data accepted");
    if (!v.ok) {
        return;
    }

    std::vector<LockstepSnapshot> snapshots[2];
    for (int s = 0; s < 2; ++s) {
        gpu::Options options;
        options.config.N = p.N;
        options.config.D = p.D;
        options.config.K = p.K;
        options.config.max_iter = 10;
        options.config.strategy = kStrategies[s];
        options.m_range = m_range;
        options.verify = gpu::Verify::kLockstep;

        std::vector<LockstepSnapshot> &into = snapshots[s];
        const gpu::LockstepHook hook = [&into](const LockstepSnapshot &snap) {
            into.push_back(snap);
        };

        Result r;
        gpu::RunInfo info;
        if (!run_device(data.X.data(), data.C0.data(), options, &r, &info, hook,
                        std::string(strategy_name(kStrategies[s])) + " run")) {
            return;
        }
    }

    check(!snapshots[0].empty() && !snapshots[1].empty(), "both runs produced snapshots");
    const size_t common = std::min(snapshots[0].size(), snapshots[1].size());
    const double s_max = s_max_from_range(m_range);

    double max_abs_sum = 0.0, max_rel_sum = 0.0, max_abs_centroid = 0.0;
    size_t compared = 0;
    int64_t sum_violations = 0, centroid_violations = 0;
    size_t diverged_at = common;

    for (size_t t = 0; t < common; ++t) {
        const LockstepSnapshot &a = snapshots[0][t];
        const LockstepSnapshot &b = snapshots[1][t];
        if (a.a_in != b.a_in) {
            // From here on the two runs are accumulating different partitions, so a difference
            // would say nothing about the accumulation (EDGE-32). Stop and report where.
            diverged_at = t;
            break;
        }
        ++compared;
        for (int64_t k = 0; k < p.K; ++k) {
            const double scale = static_cast<double>(a.counts[static_cast<size_t>(k)]) * s_max;
            const double tol = 1e-12 * (scale > s_max ? scale : s_max);
            for (int64_t j = 0; j < p.D; ++j) {
                const size_t e = static_cast<size_t>(k * p.D + j);
                const double diff = std::fabs(a.sums[e] - b.sums[e]);
                const double denom = std::fabs(b.sums[e]);
                if (diff > max_abs_sum) {
                    max_abs_sum = diff;
                }
                if (denom > 0.0 && diff / denom > max_rel_sum) {
                    max_rel_sum = diff / denom;
                }
                if (!(diff <= tol)) {
                    ++sum_violations;
                }
            }
        }
        for (int64_t e = 0; e < p.K * p.D; ++e) {
            const size_t idx = static_cast<size_t>(e);
            const double diff = std::fabs(static_cast<double>(a.c_out[idx]) -
                                          static_cast<double>(b.c_out[idx]));
            if (diff > max_abs_centroid) {
                max_abs_centroid = diff;
            }
            if (!centroid_agrees(a.c_out[idx], b.c_out[idx], s_max)) {
                ++centroid_violations;
            }
        }
        // The counts are integers in both strategies and must agree exactly.
        for (int64_t k = 0; k < p.K; ++k) {
            check_eq_i64(a.counts[static_cast<size_t>(k)], b.counts[static_cast<size_t>(k)],
                         "iteration " + std::to_string(t) + ": n[" + std::to_string(k) + "]");
        }
    }

    check(compared > 0, "at least one accumulation was compared on identical input");
    check_eq_i64(sum_violations, 0, "S agrees within 1e-12 relative (TEST-U04/U05 criterion)");
    check_eq_i64(centroid_violations, 0, "the resulting centroids agree within NUM-006");
    const std::string divergence =
        (diverged_at < common) ? "; trajectories diverged at iteration " +
                                     std::to_string(diverged_at) + " (EDGE-32, permitted)"
                               : std::string();
    std::printf("      %llu of %llu iterations compared on identical A(t)%s\n",
                static_cast<unsigned long long>(compared),
                static_cast<unsigned long long>(common), divergence.c_str());
    std::printf("      max |S_A - S_B| %.3e absolute, %.3e relative; "
                "max |C_A - C_B| %.3e (NUM-006 scale S_max = %.3e)\n",
                max_abs_sum, max_rel_sum, max_abs_centroid, s_max);
}

// --- TEST-I11: well-separated data end to end ---------------------------------------------------

void test_i11_well_separated()
{
    GenParams p;
    p.N = 200000;
    p.D = 16;
    p.K = 16;
    // Section 10.3 method 2: separation far above 10 sigma, so the device and the reference must
    // agree exactly on A, n, t and status rather than within a tolerance.
    p.sep = 50.0;
    p.spread = 0.1;
    p.seed = 110011;
    const Dataset data = generate(p);

    double m_range = 0.0;
    const Validation v = validate_data(data.X, p.N, p.D, data.C0, p.K, &m_range);
    const Result ref = cpu::run(data.X.data(), p.N, p.D, data.C0.data(), p.K, 30, 0.0, m_range,
                                false);

    for (Strategy strategy : kStrategies) {
        for (AssignKernel assign : kAssignKernels) {
            TestScope scope("I11", std::string(strategy_name(strategy)) + " / " +
                                       assign_name(assign) + "  " + std::to_string(p.N) + " x " +
                                       std::to_string(p.D) + ", K = " + std::to_string(p.K) +
                                       " blobs at " +
                                       std::to_string(static_cast<int>(data.min_centre_distance /
                                                                       data.spread)) +
                                       " sigma");
            check(v.ok, "data accepted");
            if (!v.ok) {
                continue;
            }
            gpu::Options options;
            options.config.N = p.N;
            options.config.D = p.D;
            options.config.K = p.K;
            options.config.max_iter = 30;
            options.config.strategy = strategy;
            options.config.assign_kernel = assign;
            options.m_range = m_range;

            Result r;
            gpu::RunInfo info;
            if (!run_device(data.X.data(), data.C0.data(), options, &r, &info,
                            gpu::LockstepHook(), "I11 device run")) {
                continue;
            }

            check(r.status == ref.status, std::string("status: device ") + status_name(r.status) +
                                              ", reference " + status_name(ref.status));
            check_eq_i64(r.iterations, ref.iterations, "iteration count matches exactly");
            int64_t assignment_mismatches = 0;
            for (int64_t i = 0; i < p.N; ++i) {
                if (r.assignments[static_cast<size_t>(i)] != ref.assignments[static_cast<size_t>(i)]) {
                    ++assignment_mismatches;
                }
            }
            check_eq_i64(assignment_mismatches, 0, "every assignment matches exactly (NUM-004)");
            for (int64_t k = 0; k < p.K; ++k) {
                check_eq_i64(r.counts[static_cast<size_t>(k)],
                             ref.counts[static_cast<size_t>(k)],
                             "n[" + std::to_string(k) + "]");
            }
            const double s_max = s_max_from_range(m_range);
            int64_t centroid_violations = 0;
            double worst = 0.0;
            for (int64_t e = 0; e < p.K * p.D; ++e) {
                const size_t idx = static_cast<size_t>(e);
                const double diff = std::fabs(static_cast<double>(r.centroids[idx]) -
                                              static_cast<double>(ref.centroids[idx]));
                if (diff > worst) {
                    worst = diff;
                }
                if (!centroid_agrees(r.centroids[idx], ref.centroids[idx], s_max)) {
                    ++centroid_violations;
                }
            }
            check_eq_i64(centroid_violations, 0, "every centroid within NUM-006");
            std::printf("      status %s, t = %lld, max |dC| %.3e, J = %.9e\n",
                        status_name(r.status), static_cast<long long>(r.iterations), worst,
                        r.objective);
        }
    }
}

// --- TEST-R01 / TEST-R02: repeated execution ----------------------------------------------------

// A dataset on which binary64 accumulation of binary32 addends is genuinely order-dependent.
//
// This matters because the blob datasets do not test what TEST-R02 is about. A binary32 value has
// a 24-bit significand and a binary64 accumulator has 53 bits, so adding a binary32 addend to a
// binary64 running sum is *exact* as long as their exponents differ by less than 53 - 24 = 29
// bits, i.e. by less than a factor of about 5.4e8. Blob coordinates are all within a few sigma of
// each other, far inside that window, so on blob data every partial sum is exact, the result does
// not depend on the summation order, and Approach A comes out bitwise reproducible by accident --
// which proves nothing about ATOM-007.
//
// Here the coordinates are log-uniform over magnitudes 1e-6 to 1e6, a span of about 2^40, well
// past the 2^29 window. Small addends are then rounded off against a large running sum by an
// amount that depends on when they arrive, so the hardware's atomic ordering becomes visible in
// the last bits of S -- and that is the property TEST-R02 reports on.
//
// VAL-007 is satisfied with room to spare: M = 1e6 and D = 4 give 4 D M^2 = 1.6e13, far below
// 2^126. The dataset is built here rather than by the generator because FR-003 fixes the
// generator to Gaussian blobs; this is a test fixture, not a new generator mode.
Dataset make_wide_magnitude_dataset(int64_t N, int64_t D, int64_t K, uint64_t seed)
{
    Dataset d;
    d.dataset_id = "wide-magnitude-n" + std::to_string(N) + "_d" + std::to_string(D) + "_k" +
                   std::to_string(K) + "_seed" + std::to_string(seed);
    d.spread = 1.0;
    d.X.reserve(static_cast<size_t>(N * D));

    Rng rng(seed);
    for (int64_t e = 0; e < N * D; ++e) {
        const double exponent = -6.0 + 12.0 * rng.uniform_open();  // magnitudes 1e-6 .. 1e6
        const double sign = (rng.uniform_open() < 0.5) ? -1.0 : 1.0;
        d.X.push_back(static_cast<float>(sign * std::pow(10.0, exponent)));
    }
    // Initial centroids: K points spread through the array, as FR-003 does for the blob sets.
    d.C0.reserve(static_cast<size_t>(K * D));
    for (int64_t k = 0; k < K; ++k) {
        const int64_t i = (k * N) / K;
        for (int64_t j = 0; j < D; ++j) {
            d.C0.push_back(d.X[static_cast<size_t>(i * D + j)]);
        }
    }
    return d;
}

struct ReproDataset {
    std::string name;
    std::string why;
    int64_t N = 0, D = 0, K = 0;
    std::vector<float> X, C0;
};

void test_reproducibility(int64_t runs)
{
    std::vector<ReproDataset> datasets;
    {
        GenParams p;
        p.N = 100000;
        p.D = 16;
        p.K = 8;
        p.sep = 10.0;
        p.spread = 1.0;
        p.seed = 20261018;
        const Dataset data = generate(p);
        datasets.push_back({"blobs", "every binary32 addend is within 2^29 of the running sum, so "
                                     "the binary64 partial sums are exact and order-independent",
                            p.N, p.D, p.K, data.X, data.C0});
    }
    {
        const int64_t N = 2000000, D = 4, K = 8;
        const Dataset data = make_wide_magnitude_dataset(N, D, K, 20261019);
        datasets.push_back({"wide-magnitude",
                            "magnitudes span about 2^40, past the 2^29 window in which a binary32 "
                            "addend is absorbed exactly, so the binary64 sums depend on the order",
                            N, D, K, data.X, data.C0});
    }

    for (const ReproDataset &ds : datasets) {
        double m_range = 0.0;
        const Validation v = validate_data(ds.X, ds.N, ds.D, ds.C0, ds.K, &m_range);

        for (Strategy strategy : kStrategies) {
            const bool reduction = strategy == Strategy::kReduction;
            TestScope scope(reduction ? "R01" : "R02",
                            std::string(strategy_name(strategy)) + " on " + ds.name + ", " +
                                std::to_string(runs) + " runs");
            check(v.ok, "data accepted");
            if (!v.ok) {
                continue;
            }
            std::printf("      dataset: %s\n", ds.why.c_str());

            gpu::Options options;
            options.config.N = ds.N;
            options.config.D = ds.D;
            options.config.K = ds.K;
            options.config.max_iter = 5;
            // FR-020: the same number of updates in every run, so the comparison is about
            // floating-point accumulation and not about one run stopping a step earlier.
            options.config.fixed_iters = true;
            options.config.strategy = strategy;
            // §16.5 says "fixed config", and G is part of the configuration: RED-009 makes
            // Approach B bitwise reproducible for a *fixed* (N, D, K, G), and RED-006 picks G from
            // the free memory cudaMemGetInfo reports, which another process on the machine can
            // move between runs. Pinning G is what makes TEST-R01 a test of the kernels rather
            // than of the runtime's memory accounting.
            options.config.reduce_chunks = 1024;
            options.m_range = m_range;

            Result first;
            gpu::RunInfo info;
            if (!run_device(ds.X.data(), ds.C0.data(), options, &first, &info,
                            gpu::LockstepHook(), "run 1")) {
                continue;
            }

            const double s_max = s_max_from_range(m_range);
            int64_t differing = 0, assignment_differing = 0, out_of_tolerance = 0;
            double worst_centroid = 0.0;
            std::string first_difference;
            for (int64_t run = 2; run <= runs; ++run) {
                Result again;
                if (!run_device(ds.X.data(), ds.C0.data(), options, &again, &info,
                                gpu::LockstepHook(), "run " + std::to_string(run))) {
                    continue;
                }
                std::string why;
                if (!tests::results_bit_equal(first, again, &why)) {
                    ++differing;
                    if (first_difference.empty()) {
                        first_difference = why;
                    }
                }
                for (size_t i = 0; i < first.assignments.size(); ++i) {
                    if (first.assignments[i] != again.assignments[i]) {
                        ++assignment_differing;
                        break;
                    }
                }
                for (size_t e = 0; e < first.centroids.size(); ++e) {
                    const double diff = std::fabs(static_cast<double>(first.centroids[e]) -
                                                  static_cast<double>(again.centroids[e]));
                    if (diff > worst_centroid) {
                        worst_centroid = diff;
                    }
                    // Each run's centroids must agree with the first within NUM-006. That is the
                    // per-run guarantee of the §16.5 entry for Approach A; it says nothing about
                    // the runs being identical, which ATOM-007 does not promise.
                    if (!centroid_agrees(again.centroids[e], first.centroids[e], s_max)) {
                        ++out_of_tolerance;
                    }
                }
                check(again.status == first.status, "run " + std::to_string(run) +
                                                        ": same status");
                check_eq_i64(again.iterations, first.iterations,
                             "run " + std::to_string(run) + ": same iteration count");
            }

            if (reduction) {
                // TEST-R01: RED-004 gives every element of P and Q exactly one writer and RED-009
                // fixes the summation order, so repeated runs of Approach B must be bitwise
                // identical. This is a pass/fail criterion, and it holds whatever the magnitudes
                // are -- which is why the wide-magnitude dataset is included here too.
                check_eq_i64(differing, 0, "every run is bitwise identical to the first" +
                                               (first_difference.empty()
                                                    ? std::string()
                                                    : "; first difference: " + first_difference));
                std::printf("      G = %lld; %lld of %lld runs differed; max |dC| %.3e\n",
                            static_cast<long long>(info.reduce_chunks),
                            static_cast<long long>(differing), static_cast<long long>(runs - 1),
                            worst_centroid);
            } else {
                // TEST-R02 is a report, not a threshold: ATOM-007 fixes no summation order, so
                // Approach A may differ run to run. What is checked is only that each run stays
                // within NUM-006 of the first, which is the per-run guarantee the matrix names.
                check_eq_i64(out_of_tolerance, 0,
                             "every centroid of every run is within NUM-006 of the first run");
                std::printf("      %lld of %lld repeat runs differed from the first; "
                            "%lld differed in their assignments; max |dC| %.3e "
                            "(NUM-006 scale S_max = %.3e)\n",
                            static_cast<long long>(differing),
                            static_cast<long long>(runs - 1),
                            static_cast<long long>(assignment_differing), worst_centroid,
                            s_max);
                if (!first_difference.empty()) {
                    std::printf("      first difference observed: %s\n",
                                first_difference.c_str());
                }
                if (differing == 0) {
                    std::printf("      no difference observed: on this dataset the binary64 "
                                "partial sums are exact, so the atomic order does not change the "
                                "result. That is expected here, not a guarantee (ATOM-007).\n");
                }
            }
        }
    }
}

}  // namespace

int main(int argc, char **argv)
{
    bool quick = false;
    bool run_edges = true;
    bool run_cross = true;
    bool run_repro = true;
    const char *selection = "all sections";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            quick = true;
        } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            const char *what = argv[++i];
            if (std::strcmp(what, "edges") == 0) {
                run_cross = run_repro = false;
                selection = "corner cases only";
            } else if (std::strcmp(what, "cross") == 0) {
                run_edges = run_repro = false;
                selection = "cross-strategy and well-separated only";
            } else if (std::strcmp(what, "repro") == 0) {
                run_edges = run_cross = false;
                selection = "reproducibility only";
            } else if (std::strcmp(what, "all") != 0) {
                std::fprintf(stderr,
                             "error[VAL-010]: --only expects edges, cross, repro or all\n");
                return kExitInvalidInput;
            }
        } else {
            std::fprintf(stderr,
                         "usage: test_edge [--only {all|edges|cross|repro}] [--quick]\n");
            return kExitInvalidInput;
        }
    }

    std::printf("M5 corner-case, cross-strategy and reproducibility suite (%s, %s)\n",
                quick ? "quick" : "full", selection);
    std::printf("requirements.md sections 9.2, 16.1, 16.2, 16.5\n\n");

    DeviceInfo device;
    try {
        device = query_device(0);
    } catch (const CudaError &err) {
        std::fprintf(stderr, "error[CUDA-001]: %s\n", err.what());
        return err.exit_code();
    }
    std::printf("device: %s\n", describe_device(device).c_str());
    std::printf("        tiles T_K = %d, T_D = %d, %lld B of shared memory per 256-thread block\n\n",
                kAssignTileK, kAssignTileD,
                static_cast<long long>(assign_smem_shared_bytes(kAssignTileK, kAssignTileD, 256)));

    if (run_edges) {
        std::printf("TEST-E04..E12, E15: structural corner cases (section 9.2)\n");
        test_e04_k_greater_than_n();
        test_e05_n_equals_k();
        test_e06_single_point();
        test_e08_single_cluster();
        test_e09_duplicate_points();
        test_e10_duplicate_centroids();
        test_e11_identical_points();
        test_e12_identical_centroids();

        std::printf("\nTEST-E18 / E19: magnitudes near the VAL-007 bound, and near-ties\n");
        test_e18_large_magnitudes();
        test_e19_tiny_differences();

        std::printf("\nTEST-E26 / E29 / E30 / E34: the pre-checks that run before any "
                    "allocation\n");
        test_pre_allocation_rejections(device);

        std::printf("\nTEST-U02: tie-breaking on the device (DEC-005)\n");
        test_u02_tie_breaking();
    }

    if (run_cross) {
        if (!quick) {
            std::printf("\nTEST-E29: a large configuration that does run\n");
            test_e29_large_run();
        }
        std::printf("\nTEST-I09: Approach A against Approach B\n");
        test_i09_cross_strategy();
        std::printf("\nTEST-I11: well-separated data end to end\n");
        test_i11_well_separated();
    }

    if (run_repro) {
        // Section 16.5 asks for 10 runs. A quick run does 3, which can still show a difference but
        // is not the §16.5 evidence, and says so.
        const int64_t runs = quick ? 3 : 10;
        std::printf("\nTEST-R01 / TEST-R02: repeated execution with identical inputs "
                    "(section 16.5, EDGE-33)\n");
        if (quick) {
            std::printf("  note: 3 runs per configuration in quick mode; section 16.5 requires "
                        "10, so this is not the acceptance evidence\n");
        }
        test_reproducibility(runs);
    }

    return tests::report_totals("all corner-case, cross-strategy and reproducibility tests "
                                "in this run passed");
}
