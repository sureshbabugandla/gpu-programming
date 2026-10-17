// cpu_reference.h — single-threaded CPU oracle (requirements.md NUM-003, ARC-03).
//
// Implements §2 exactly: the binary32 distance path of §2.2 with ascending-j accumulation, a
// binary64 path used only for near-tie diagnosis (NUM-005), binary64 sums in ascending i (§2.4),
// the lowest-index tie rule (DEC-005), empty-cluster retention (DEC-003), the convergence rule
// (DEC-002), iteration counting (DEC-006) and finalization (§2.8).
//
// The step functions are exposed individually, not only through run(), because §10.3 lockstep
// verification feeds the device's own state into each step and compares the result.
//
// Must be compiled with -ffp-contract=off and without fast-math: contracting a multiply-add in
// distance2_fp32 would change the oracle's rounding and make near-tie diagnosis meaningless.

#ifndef KMEANS_CPU_REFERENCE_H
#define KMEANS_CPU_REFERENCE_H

#include <cstdint>
#include <vector>

#include "common.h"

namespace kmeans {
namespace cpu {

// §2.2 in binary32: terms and accumulation in binary32, ascending j, never reordered.
float distance2_fp32(const float *x, const float *c, int64_t D);

// The same quantity in binary64, for NUM-005 margins and for the objective (§2.9). Not used for
// assignment: that must see exactly what the device sees.
double distance2_fp64(const float *x, const float *c, int64_t D);

// §2.3 for one point: nearest centroid by binary32 distance, lowest index among exactly equal
// computed values (strict-less-than replacement in ascending k).
int32_t nearest_centroid(const float *x, int64_t D, const float *C, int64_t K);

// §2.3 for all points, writing A. Returns the number of points whose assignment differs from
// A_prev (FR-007). With A_prev == nullptr this is the initial pass and the return value is N,
// which the driver ignores because changes are only counted inside the loop.
int64_t assign(const float *X, int64_t N, int64_t D, const float *C, int64_t K, int32_t *A,
               const int32_t *A_prev);

// NUM-005 diagnosis for one point, in binary64.
struct TieInfo {
    double d1 = 0.0;      // smallest binary64 distance
    double d2 = 0.0;      // second smallest
    double margin = 0.0;  // (d2 - d1) / max(d2, FLT_MIN)
    int32_t k1 = 0;       // index attaining d1 (lowest index on ties)
    int32_t k2 = 0;       // index attaining d2
    bool near_tie = false;
};
TieInfo tie_info(const float *x, int64_t D, const float *C, int64_t K);

// §2.4: counts exactly, sums in binary64 in ascending i. S has K*D entries, n has K.
void accumulate(const float *X, int64_t N, int64_t D, const int32_t *A, int64_t K, double *S,
                uint32_t *n);

// §2.5: C_next[k] = fl32(S[k]/n[k]) for n[k] > 0, else C_cur[k] retained bit-exactly. No division
// is performed when n[k] == 0. C_next must differ from C_cur (double buffering, FR-010).
void update(const double *S, const uint32_t *n, int64_t K, int64_t D, const float *C_cur,
            float *C_next);

// §2.6: max over clusters with n[k] > 0 of the binary64 Euclidean displacement. Empty clusters
// contribute delta_k = 0 by construction and are excluded from the maximum.
double movement(const float *C_cur, const float *C_next, const uint32_t *n, int64_t K, int64_t D);

// §2.9: J in binary64, computed after the run and outside any timed region (FR-016).
double objective(const float *X, int64_t N, int64_t D, const int32_t *A, const float *C);

// §2.8 steps 1 and 2: one accumulation pass over A_final, then the P2 check against NUM-006.
// C_final is never modified. `m_range` is the M of VAL-007, which fixes the NUM-006 scale.
struct Finalization {
    std::vector<uint32_t> counts;       // n_final
    std::vector<float> means;           // fl32(S/n) for non-empty k, C_final retained otherwise
    bool centroid_consistent = false;   // P2
    double max_centroid_difference = 0.0;
};
Finalization finalize(const float *X, int64_t N, int64_t D, const int32_t *A_final,
                      const float *C_final, int64_t K, double m_range);

// §2.7 driver. `fixed_iters` (FR-020) disables both convergence criteria so exactly max_iter
// updates run and the status is ITERATION_LIMIT.
Result run(const float *X, int64_t N, int64_t D, const float *C0, int64_t K, int64_t max_iter,
           double tol, double m_range, bool fixed_iters);

}  // namespace cpu
}  // namespace kmeans

#endif  // KMEANS_CPU_REFERENCE_H
