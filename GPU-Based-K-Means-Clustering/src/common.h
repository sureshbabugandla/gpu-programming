// common.h — shared types, statuses, exit codes, messages and size arithmetic.
//
// requirements.md: Status (§2.7), ExitCode (BLD-005), message format (BLD-006),
// overflow-checked size arithmetic (CUDA-006) and the capacity model (MEM-002).
//
// Header-only and free of CUDA constructs, so the M2 .cu translation units include it unchanged.

#ifndef KMEANS_COMMON_H
#define KMEANS_COMMON_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace kmeans {

// --- §2.7 termination statuses -----------------------------------------------------------------

enum class Status {
    CONVERGED_ASSIGNMENTS,  // A(t) == A(t-1) exactly
    CONVERGED_MOVEMENT,     // delta <= epsilon, assignments still changing
    ITERATION_LIMIT,        // t == max_iter and neither criterion held (includes max_iter == 0)
    ERROR                   // validation, resource, CUDA or I/O failure; no clustering output
};

inline const char *status_name(Status s)
{
    switch (s) {
    case Status::CONVERGED_ASSIGNMENTS: return "CONVERGED_ASSIGNMENTS";
    case Status::CONVERGED_MOVEMENT:    return "CONVERGED_MOVEMENT";
    case Status::ITERATION_LIMIT:       return "ITERATION_LIMIT";
    case Status::ERROR:                 return "ERROR";
    }
    return "UNKNOWN";
}

// §2.7: termination is never reported as convergence.
inline bool is_converged(Status s)
{
    return s == Status::CONVERGED_ASSIGNMENTS || s == Status::CONVERGED_MOVEMENT;
}

// --- BLD-005 exit codes ------------------------------------------------------------------------

enum ExitCode {
    kExitOk = 0,                  // any non-error status
    kExitInvalidInput = 2,        // VAL rule violated
    kExitUnsupportedDevice = 3,   // unsupported device or insufficient resources
    kExitCudaError = 4,           // CUDA runtime error
    kExitIoError = 5,             // I/O error
    kExitVerificationFailed = 6   // only with --verify
};

// --- BLD-006 messages --------------------------------------------------------------------------
// Single-line prefix naming the rule that was violated, then the cause and the offending value.

inline void report_error(const char *rule_id, const std::string &what)
{
    std::fprintf(stderr, "error[%s]: %s\n", rule_id, what.c_str());
}

inline void report_warning(const char *warning_id, const std::string &what)
{
    std::fprintf(stderr, "warning[%s]: %s\n", warning_id, what.c_str());
}

// --- numerical constants (§0.1, NUM-005, NUM-006) ----------------------------------------------

// u32 = 2^-24, the unit roundoff of IEEE-754 binary32.
constexpr double kUnitRoundoffF32 = 5.9604644775390625e-08;

// NUM-005: a point is a near-tie when its relative margin is at most 4 (D + 2) u32.
inline double near_tie_threshold(int64_t D)
{
    return 4.0 * static_cast<double>(D + 2) * kUnitRoundoffF32;
}

// NUM-006: |a - b| <= 4 u32 |b| + 4 u32 S_max 2^-10, with a the value under test and b the
// reference. Both sides round one binary64 quotient to binary32, so they differ by at most about
// one binary32 ulp; the absolute term covers values near zero.
inline double centroid_tolerance(float reference, double s_max)
{
    const double abs_ref = std::fabs(static_cast<double>(reference));
    return 4.0 * kUnitRoundoffF32 * abs_ref +
           4.0 * kUnitRoundoffF32 * s_max * 0.0009765625;  // 2^-10
}

inline bool centroid_agrees(float test, float reference, double s_max)
{
    const double diff = std::fabs(static_cast<double>(test) - static_cast<double>(reference));
    return diff <= centroid_tolerance(reference, s_max);
}

// NUM-006: S_max = max(M, 1e-30), with M the range bound of VAL-007.
inline double s_max_from_range(double m_range)
{
    return m_range > 1e-30 ? m_range : 1e-30;
}

// --- CUDA-006 / MEM-002 size arithmetic --------------------------------------------------------
// Every element offset and allocation size is computed in unsigned 64-bit arithmetic with an
// explicit overflow check, so an oversized configuration is rejected before any allocation
// instead of wrapping into a small, plausible-looking byte count.

inline bool checked_mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a != 0 && b > UINT64_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

inline bool checked_add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a > UINT64_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

// Byte counts of MEM-001, in the order of that table. `reduce_chunks` is the Approach B chunk
// count G; pass 0 for Approach A, whose run allocates no P/Q buffers.
struct MemoryModel {
    uint64_t x_col = 0;        // 4ND
    uint64_t x_row = 0;        // 4ND
    uint64_t centroids = 0;    // 8KD   (C_cur and C_next)
    uint64_t assignments = 0;  // 4N
    uint64_t sums = 0;         // 8KD
    uint64_t counts = 0;       // 4K
    uint64_t partials = 0;     // 8GKD
    uint64_t partial_counts = 0;  // 4GK
    uint64_t scalars = 0;      // movement block maxima, delta, change counter, pinned scalars
    uint64_t total = 0;
    bool overflow = false;     // true when any product or sum exceeded 64 bits
};

// MEM-002: required bytes = 8ND + 4N + 16KD + 4K + [8GKD + 4GK] + small.
inline MemoryModel device_memory_bytes(int64_t N, int64_t D, int64_t K, int64_t reduce_chunks,
                                       int64_t movement_blocks = 1024)
{
    MemoryModel m;
    const uint64_t n = static_cast<uint64_t>(N);
    const uint64_t d = static_cast<uint64_t>(D);
    const uint64_t k = static_cast<uint64_t>(K);
    const uint64_t g = static_cast<uint64_t>(reduce_chunks);

    uint64_t nd = 0, kd = 0, gkd = 0, gk = 0;
    bool ok = checked_mul(n, d, &nd) && checked_mul(k, d, &kd);
    ok = ok && checked_mul(g, kd, &gkd) && checked_mul(g, k, &gk);
    if (!ok) {
        m.overflow = true;
        return m;
    }

    ok = ok && checked_mul(nd, 4, &m.x_col);
    ok = ok && checked_mul(nd, 4, &m.x_row);
    ok = ok && checked_mul(kd, 8, &m.centroids);
    ok = ok && checked_mul(n, 4, &m.assignments);
    ok = ok && checked_mul(kd, 8, &m.sums);
    ok = ok && checked_mul(k, 4, &m.counts);
    ok = ok && checked_mul(gkd, 8, &m.partials);
    ok = ok && checked_mul(gk, 4, &m.partial_counts);
    ok = ok && checked_mul(static_cast<uint64_t>(movement_blocks), 8, &m.scalars);
    ok = ok && checked_add(m.scalars, 32, &m.scalars);  // delta, change counter, pinned copies
    if (!ok) {
        m.overflow = true;
        return m;
    }

    const uint64_t parts[] = {m.x_col,  m.x_row, m.centroids, m.assignments,  m.sums,
                              m.counts, m.partials, m.partial_counts, m.scalars};
    uint64_t total = 0;
    for (uint64_t part : parts) {
        if (!checked_add(total, part, &total)) {
            m.overflow = true;
            return m;
        }
    }
    m.total = total;
    return m;
}

// --- run configuration and result --------------------------------------------------------------

enum class Strategy { kAtomic, kReduction };
enum class AssignKernel { kBaseline, kSharedMemory };

struct Config {
    int64_t N = 0;
    int64_t D = 0;
    int64_t K = 0;
    int64_t max_iter = 0;
    double tol = 0.0;
    bool fixed_iters = false;  // FR-020: disable both convergence criteria
    Strategy strategy = Strategy::kAtomic;
    AssignKernel assign_kernel = AssignKernel::kBaseline;
    int block_size = 256;          // CUDA-007 default
    int64_t reduce_chunks = 0;     // 0 = choose automatically (RED-006)
};

// §2.8 step 3 reports exactly these fields; the CPU reference and the device driver fill the same
// record so the two can be compared field by field (§10.3).
struct Result {
    std::vector<int32_t> assignments;  // A_final
    std::vector<float> centroids;      // C_final, K x D row-major, unchanged by finalization
    std::vector<uint32_t> counts;      // n_final from the finalization accumulation pass
    int64_t iterations = 0;            // DEC-006: completed centroid updates
    Status status = Status::ERROR;
    double last_delta = 0.0;           // delta of the last update
    bool has_last_delta = false;       // false when t == 0 (reported as null, FR-015)
    bool centroid_consistent = false;  // P2, verified per NUM-006
    bool assignment_consistent = true; // P1, true by construction (§2.8)
    double objective = 0.0;            // J (§2.9), binary64, outside timed regions
};

}  // namespace kmeans

#endif  // KMEANS_COMMON_H
