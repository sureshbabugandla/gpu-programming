#include "validate.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "launch_config.h"

namespace kmeans {
namespace {

constexpr int64_t kInt32Max = 2147483647;  // 2^31 - 1 (VAL-002, VAL-004)

Validation fail(const char *rule, const std::string &message, int exit_code)
{
    Validation v;
    v.ok = false;
    v.rule = rule;
    v.message = message;
    v.exit_code = exit_code;
    return v;
}

// Distinct names rather than overloads: an overloaded num(int) would be ambiguous between the integral and
// floating-point conversions, and every message below states exactly which kind of value it holds.
std::string istr(int64_t value)
{
    return std::to_string(value);
}

std::string ustr(uint64_t value)
{
    return std::to_string(value);
}

std::string dstr(double value)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", value);
    return std::string(buf);
}

}  // namespace

bool Validation::has_warning(const std::string &id) const
{
    return std::find(warnings.begin(), warnings.end(), id) != warnings.end();
}

void Validation::report() const
{
    for (const std::string &w : warnings) {
        if (w == "W-KGTN") {
            report_warning(w.c_str(), "K > N: at least K - N clusters stay empty and retain their "
                                      "initial centroids (VAL-003, DEC-003)");
        } else if (w == "W-DUPC") {
            report_warning(w.c_str(), "duplicate initial centroids: the higher-index duplicate "
                                      "receives no points while identical (VAL-008, DEC-005)");
        } else {
            report_warning(w.c_str(), "see requirements.md");
        }
    }
    if (!ok) {
        report_error(rule.c_str(), message);
    }
}

Validation validate_parameters(int64_t N, int64_t D, int64_t K, int64_t max_iter, double tol)
{
    // VAL-001
    if (N < 1) {
        return fail("VAL-001", "N must be at least 1, got N=" + istr(N), kExitInvalidInput);
    }
    if (D < 1) {
        return fail("VAL-001", "D must be at least 1, got D=" + istr(D), kExitInvalidInput);
    }
    if (K < 1) {
        return fail("VAL-001", "K must be at least 1, got K=" + istr(K), kExitInvalidInput);
    }

    // VAL-002: assignments must fit int32 and counts uint32.
    if (N > kInt32Max) {
        return fail("VAL-002", "N must be at most 2^31-1, got N=" + istr(N), kExitInvalidInput);
    }
    if (D > kInt32Max) {
        return fail("VAL-002", "D must be at most 2^31-1, got D=" + istr(D), kExitInvalidInput);
    }
    if (K > kInt32Max) {
        return fail("VAL-002", "K must be at most 2^31-1, got K=" + istr(K), kExitInvalidInput);
    }

    // VAL-004: max_iter = 0 is accepted and means "initial assignment only" (§2.7).
    if (max_iter < 0 || max_iter > kInt32Max) {
        return fail("VAL-004", "max-iter must be in [0, 2^31-1], got " + istr(max_iter),
                    kExitInvalidInput);
    }

    // VAL-005: epsilon finite and non-negative; NaN, +-Inf and negative values are rejected.
    if (std::isnan(tol)) {
        return fail("VAL-005", "tol must be a finite number, got NaN", kExitInvalidInput);
    }
    if (std::isinf(tol)) {
        return fail("VAL-005", "tol must be finite, got " + dstr(tol), kExitInvalidInput);
    }
    if (tol < 0.0) {
        return fail("VAL-005", "tol must be non-negative, got " + dstr(tol), kExitInvalidInput);
    }

    Validation v;
    // VAL-003: K > N is accepted with a warning; at least K - N clusters remain empty (DEC-003).
    if (K > N) {
        v.warnings.push_back("W-KGTN");
    }
    return v;
}

Validation validate_options(int block_size, int64_t reduce_chunks, int tile_k, int tile_d,
                            int device_max_threads_per_block, int64_t device_shared_mem_per_block)
{
    // CUDA-007: block size is a multiple of 32 in [32, 1024] and within the device limit.
    if (block_size < 32 || block_size > 1024 || block_size % 32 != 0) {
        return fail("VAL-010",
                    "block-size must be a multiple of 32 in [32, 1024], got " + istr(block_size),
                    kExitInvalidInput);
    }
    if (device_max_threads_per_block > 0 && block_size > device_max_threads_per_block) {
        return fail("VAL-010", "block-size " + istr(block_size) + " exceeds the device limit of " +
                                   istr(device_max_threads_per_block),
                    kExitInvalidInput);
    }

    // RED-006: G is chosen automatically when 0, otherwise it must be at least 1.
    if (reduce_chunks < 0) {
        return fail("VAL-010", "reduce-chunks must be at least 1 (or 0 for automatic), got " +
                                   istr(reduce_chunks),
                    kExitInvalidInput);
    }

    // SMEM-001/SMEM-003: tile sizes positive, T_D a multiple of 16, 4 T_K T_D bytes of shared
    // memory within the per-block limit (EDGE-34 rejects oversized configurations).
    if (tile_k < 1 || tile_d < 1) {
        return fail("VAL-010", "tile sizes must be at least 1, got tile_k=" + istr(tile_k) +
                                   " tile_d=" + istr(tile_d),
                    kExitInvalidInput);
    }
    // A multiple of 16, not of 32: the documented range of T_D is the PERF-003 tile sweep,
    // {16, 32}, and the kernel needs only an even T_D (its static_assert) so that the T_K x T_D
    // tile fills whole 8-byte words and the change-count reduction after it starts aligned. 16
    // floats is still exactly one 64-byte segment per tile row, which is the coalescing
    // property the warp-multiple form of this rule was protecting; widths that are not a whole
    // number of segments, such as 8 or 24, stay rejected.
    if (tile_d % 16 != 0) {
        return fail("VAL-010", "tile_d must be a multiple of 16, got " + istr(tile_d),
                    kExitInvalidInput);
    }
    uint64_t tile_bytes = 0;
    if (!checked_mul(static_cast<uint64_t>(tile_k) * 4, static_cast<uint64_t>(tile_d),
                     &tile_bytes)) {
        return fail("VAL-010", "tile sizes overflow the shared-memory calculation",
                    kExitInvalidInput);
    }
    // What assign_smem actually requests is the tile plus its change-count reduction, so that sum
    // is what has to fit; checking only the tile would let a configuration through that then
    // failed at the launch (EDGE-34).
    uint64_t shared_bytes = 0;
    if (!checked_add(tile_bytes, static_cast<uint64_t>(block_reduction_bytes(block_size)),
                     &shared_bytes)) {
        return fail("VAL-010", "tile sizes overflow the shared-memory calculation",
                    kExitInvalidInput);
    }
    if (device_shared_mem_per_block > 0 &&
        shared_bytes > static_cast<uint64_t>(device_shared_mem_per_block)) {
        return fail("VAL-010", "assign_smem needs " + ustr(shared_bytes) +
                                   " B of shared memory per block (" + ustr(tile_bytes) +
                                   " B tile + " + istr(block_reduction_bytes(block_size)) +
                                   " B change-count reduction), device allows " +
                                   istr(device_shared_mem_per_block) + " B per block",
                    kExitInvalidInput);
    }
    return Validation{};
}

double range_bound(const std::vector<float> &X, const std::vector<float> &C0)
{
    double m = -1.0;
    for (float value : X) {
        const double a = std::fabs(static_cast<double>(value));
        if (a > m) {
            m = a;
        }
    }
    for (float value : C0) {
        const double a = std::fabs(static_cast<double>(value));
        if (a > m) {
            m = a;
        }
    }
    return m;
}

Validation validate_data(const std::vector<float> &X, int64_t N, int64_t D,
                         const std::vector<float> &C0, int64_t K, double *m_range)
{
    Validation v;

    // VAL-008: the centroid file has exactly K rows of D values; the data file N rows of D.
    const uint64_t expect_x = static_cast<uint64_t>(N) * static_cast<uint64_t>(D);
    const uint64_t expect_c = static_cast<uint64_t>(K) * static_cast<uint64_t>(D);
    if (X.size() != expect_x) {
        return fail("VAL-008", "data has " + ustr(X.size()) +
                                   " values, expected N*D = " + ustr(expect_x),
                    kExitInvalidInput);
    }
    if (C0.size() != expect_c) {
        return fail("VAL-008", "centroids have " + ustr(C0.size()) +
                                   " values, expected K*D = " + ustr(expect_c),
                    kExitInvalidInput);
    }

    // VAL-006: every coordinate finite, reporting the first offending (row, column).
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < D; ++j) {
            const float value = X[static_cast<size_t>(i * D + j)];
            if (!std::isfinite(value)) {
                return fail("VAL-006", "data value at (row " + istr(i) + ", column " + istr(j) +
                                           ") is not finite: " + dstr(static_cast<double>(value)),
                            kExitInvalidInput);
            }
        }
    }
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t j = 0; j < D; ++j) {
            const float value = C0[static_cast<size_t>(k * D + j)];
            if (!std::isfinite(value)) {
                return fail("VAL-006", "initial centroid at (row " + istr(k) + ", column " + istr(j) +
                                           ") is not finite: " + dstr(static_cast<double>(value)),
                            kExitInvalidInput);
            }
        }
    }

    // VAL-007: with M the largest magnitude in X and C0, require 4 D M^2 <= 2^126 so that no
    // binary32 distance can overflow during the run. Later centroids are means of points or
    // retained initial centroids, so they stay within [-M, M] up to rounding.
    const double m = range_bound(X, C0);
    const double bound = std::ldexp(1.0, 126);
    const double value = 4.0 * static_cast<double>(D) * m * m;
    if (!(value <= bound)) {
        return fail("VAL-007", "range bound exceeded: M=" + dstr(m) + ", 4*D*M^2=" + dstr(value) +
                                   " > 2^126=" + dstr(bound),
                    kExitInvalidInput);
    }
    if (m_range != nullptr) {
        *m_range = m;
    }

    // VAL-008: duplicate initial centroids are accepted with a warning. Compared bit-wise on the
    // float values, which is what the tie rule of DEC-005 sees.
    std::map<std::string, int64_t> seen;
    for (int64_t k = 0; k < K; ++k) {
        std::string key(reinterpret_cast<const char *>(&C0[static_cast<size_t>(k * D)]),
                        static_cast<size_t>(D) * sizeof(float));
        if (seen.count(key) != 0) {
            v.warnings.push_back("W-DUPC");
            break;
        }
        seen.emplace(std::move(key), k);
    }
    return v;
}

Validation validate_capacity(int64_t N, int64_t D, int64_t K, int64_t reduce_chunks,
                             uint64_t free_bytes)
{
    const MemoryModel m = device_memory_bytes(N, D, K, reduce_chunks);
    if (m.overflow) {
        // EDGE-30: index products near or above 2^63 are rejected by the checked arithmetic.
        return fail("VAL-009",
                    "required device memory overflows 64-bit arithmetic for N=" + istr(N) +
                        " D=" + istr(D) + " K=" + istr(K) + " G=" + istr(reduce_chunks),
                    kExitInvalidInput);
    }
    // A 5 % safety margin keeps allocation metadata and the context out of the way.
    const uint64_t budget = static_cast<uint64_t>(static_cast<double>(free_bytes) * 0.95);
    if (m.total > budget) {
        return fail("VAL-009",
                    "required " + ustr(m.total) + " B exceeds the usable " +
                        ustr(budget) + " B (95 % of " +
                        ustr(free_bytes) + " B free)",
                    kExitUnsupportedDevice);
    }
    return Validation{};
}

}  // namespace kmeans
