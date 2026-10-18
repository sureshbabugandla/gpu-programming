// lockstep.h — the §10.3 iteration-by-iteration comparison against the CPU reference.
//
// One implementation, two users: the GPU test driver (TEST-I10) and the `--verify lockstep` mode
// of the `kmeans` executable (FR-019). Both feed it the same snapshots, so a failure in the CLI
// and a failure in the test suite mean exactly the same thing.
//
// Free of CUDA constructs, so the comparison logic — which is where the numerical argument of
// §10.3 actually lives — is compiled and tested with g++ on a machine with no GPU (CLAUDE.md).
// The snapshot is a plain record of host copies; the driver fills it, this header judges it.

#ifndef KMEANS_LOCKSTEP_H
#define KMEANS_LOCKSTEP_H

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

namespace kmeans {

// One iteration of device state. The snapshot is taken while A(t) still exists, before the
// assignment pass overwrites it in place (FR-007).
struct LockstepSnapshot {
    int64_t iteration = 0;        // t before this update, so the update produces C(t+1)
    std::vector<int32_t> a_in;    // A(t), the input of the accumulation
    std::vector<float> c_in;      // C(t)
    std::vector<uint32_t> counts; // device n from the accumulation
    std::vector<double> sums;     // device S from the accumulation
    std::vector<float> c_out;     // C(t+1) from the device update
    std::vector<int32_t> a_out;   // A(t+1) from the device assignment pass
    double delta = 0.0;           // device Delta
    int64_t changed = 0;          // device change count
};

// Counters over a whole run. Everything except `assign_near_tie_mismatches` must be zero for the
// run to pass: near-tie disagreements are permitted by NUM-005 and are reported, not failed.
struct LockstepStats {
    int64_t iterations = 0;
    int64_t assign_hard_mismatches = 0;
    int64_t assign_near_tie_mismatches = 0;
    int64_t count_mismatches = 0;
    int64_t sum_violations = 0;
    int64_t centroid_violations = 0;
    int64_t change_count_mismatches = 0;
    int64_t delta_violations = 0;
    double max_sum_diff = 0.0;
    double max_centroid_diff = 0.0;
    double max_delta_diff = 0.0;
    std::vector<std::string> notes;  // first ten diagnostics, each naming the quantity
};

// One iteration of §10.3 item 1. Every kernel of the iteration is judged separately, so a failure
// names the kernel rather than only the end result:
//   accumulation  device n exactly, device S within 1e-12 * n_k * S_max (TEST-U04/U05 criterion)
//   update        device C(t+1) within NUM-006 of the reference update of the same (S, n)
//   movement      device Delta against the reference movement of the device's own centroids
//   assign        device A(t+1) against the reference assignment of the device's own C(t+1),
//                 with disagreements allowed only on near-ties (NUM-005)
//   change count  exactly, against the reference count of differences between A(t) and A(t+1)
void check_lockstep_iteration(const LockstepSnapshot &snap, const float *X, const Config &config,
                              double m_range, LockstepStats *stats);

// True when the run satisfies §10.3: at least one iteration checked and every hard counter zero.
bool lockstep_ok(const LockstepStats &stats);

// One line of the counters and the worst differences, for a test log or the CLI output.
std::string lockstep_summary(const LockstepStats &stats);

}  // namespace kmeans

#endif  // KMEANS_LOCKSTEP_H
