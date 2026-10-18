// kmeans_gpu.h — the device driver loop and finalization (ARC-09, section 2.7, section 2.8).
//
// The entry point mirrors cpu::run so the two can be compared field by field (section 10.3,
// ARC-03/ARC-09 interface contract): same inputs, same Result record. Everything that differs
// between Approach A and Approach B is inside the accumulation step, which is what FR-018
// requires -- assignment, update, convergence and finalization are one code path.
//
// Errors are raised as CudaError (cuda_util.cuh) carrying the BLD-005 exit code, so every early
// return unwinds through the RAII buffer owners and releases device memory (MEM-007).

#ifndef KMEANS_KMEANS_GPU_H
#define KMEANS_KMEANS_GPU_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common.h"
#include "device_context.cuh"
#include "lockstep.h"

namespace kmeans {
namespace gpu {

// FR-019 verification modes.
enum class Verify {
    kNone,
    kFinal,     // P1 and P2 checked after the run (verify_final below)
    kLockstep   // every iteration compared against the CPU reference (section 10.3)
};

struct Options {
    Config config;         // N, D, K, max_iter, tol, strategy, assign kernel, block size, G
    double m_range = 0.0;  // the M of VAL-007, which fixes the NUM-006 scale
    int device = 0;
    uint64_t reduction_cap_bytes = 0;  // MEM-003 user cap on B_red; 0 means no explicit cap
    Verify verify = Verify::kNone;

    // PERF-007: record an event pair around each phase of the loop and sum them across
    // iterations. Off by default because the extra events perturb the totals, which is why
    // PERF-007 keeps phase runs out of the headline numbers.
    bool profile_phases = false;

    // TEST-U06: fill P, Q, S and n with 0xFF bytes (NaN and 0xFFFFFFFF) after allocation. A
    // correct Approach B overwrites every element of P and Q in Stage 1 and every element of S
    // and n in Stage 2, so the poison must not survive into any result.
    bool poison_buffers = false;

    // TEST-R02: copy the finalization sums S to the host alongside the result. S is the binary64
    // accumulator the section 2.8 step 1 pass fills, and M = fl32(S/n) is formed from it; the
    // reproducibility report needs S itself because rounding to binary32 hides last-bit
    // differences. The copy happens after the T_alg end event, so it costs the timings nothing.
    bool capture_sums = false;
};

// What the run reports about itself beyond the clustering Result.
struct RunInfo {
    DeviceInfo device;
    int64_t reduce_chunks = 0;   // the G of RED-006, 0 for Approach A; reported per RED-007
    int64_t g_memory = 0;        // G_mem, for the record
    int64_t g_target = 0;        // max(1, ceil(T_target / D))
    uint64_t reduction_bytes = 0;  // 8GKD + 4GK actually allocated
    int assign_grid = 0;
    int accum_grid = 0;
    int movement_grid = 0;

    // PERF-005 timings in milliseconds.
    double t_alg_ms = 0.0;
    double t_e2e_ms = 0.0;
    double t_alloc_ms = 0.0;
    double t_h2d_ms = 0.0;
    double t_d2h_ms = 0.0;
    // FR-019: lockstep downloads sit inside the measured region, so a lockstep run's timings are
    // not comparable and must never be used as a benchmark sample.
    bool timing_valid = true;

    // PERF-007 phase sums in milliseconds, filled only when options.profile_phases is set. They
    // cover the same region as t_alg_ms and sum to less than it: the per-iteration 16-byte
    // transfer and the host decision gap belong to no phase and are left unattributed rather
    // than folded into one.
    double t_assign_ms = 0.0;
    double t_accum_ms = 0.0;
    double t_update_ms = 0.0;
    double t_conv_ms = 0.0;
    bool phases_measured = false;

    double max_centroid_difference = 0.0;  // worst |C_final - M| from the P2 check (section 2.8)

    // TEST-R02, filled only when options.capture_sums is set: K*D binary64 finalization sums.
    std::vector<double> final_sums;
};

// The snapshot record and the comparison that judges it live in lockstep.h, free of CUDA, so the
// §10.3 logic is shared by the test driver and the `--verify lockstep` mode of the executable and
// is compiled without a GPU. The alias keeps `gpu::LockstepSnapshot` readable at the call sites,
// where the snapshot is unambiguously device state.
using kmeans::LockstepSnapshot;

using LockstepHook = std::function<void(const LockstepSnapshot &)>;

// Section 2.7 on the device. X and C0 are host arrays in row-major binary32 (section 2.1).
// `info` may be null. `hook` is called once per iteration when options.verify == kLockstep.
Result run(const float *X, const float *C0, const Options &options, RunInfo *info,
           const LockstepHook &hook);

inline Result run(const float *X, const float *C0, const Options &options, RunInfo *info)
{
    return run(X, C0, options, info, LockstepHook());
}

// --- FR-019 `final` verification ---------------------------------------------------------------

// PERF-001 (specification v1.9): P1 costs N*K*D binary32 distance evaluations on the host, which
// at the top of the PERF-003 matrix exceeds the cost of the run it guards. Above this many terms
// the check is made on a fixed seeded sample instead, and the mode is reported with the result.
// P2 is O(N*D) and is never sampled.
constexpr int64_t kP1FullCheckTerms = 20000000000LL;  // 2e10
constexpr int64_t kP1SampleSize = 100000;
constexpr uint64_t kP1SampleSeed = 0x5EEDC0DEULL;  // fixed, so the sample is reproducible

struct FinalVerification {
    int64_t p1_hard_mismatches = 0;       // non-near-tie disagreements: failures (NUM-005)
    int64_t p1_near_tie_mismatches = 0;   // near-tie disagreements: reported, not failures
    bool p2_flag_correct = false;         // the device's P2 flag equals the reference's
    bool reference_p2 = false;
    double max_centroid_difference = 0.0;
    std::vector<std::string> reports;     // one line per near-tie or hard mismatch

    // PERF-001 mode. `p1_points_checked` is N for a full check and kP1SampleSize for a sampled
    // one; `p1_mode` is the §13.1 `p1_check_mode` value, "full" or "sampled:<size>".
    bool p1_sampled = false;
    int64_t p1_points_checked = 0;
    std::string p1_mode = "full";
};

// P1: every reported assignment is the nearest reported centroid, judged with the CPU reference's
// binary32 distance path and the near-tie rule of NUM-005. P2: the reference's finalization pass
// agrees with the device's centroid_consistent flag (section 2.8 step 2).
FinalVerification verify_final(const float *X, const Result &result, const Config &config,
                               double m_range);

}  // namespace gpu
}  // namespace kmeans

#endif  // KMEANS_KMEANS_GPU_H
