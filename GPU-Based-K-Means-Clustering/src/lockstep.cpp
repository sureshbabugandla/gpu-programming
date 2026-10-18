#include "lockstep.h"

#include <cmath>
#include <cstdio>

#include "cpu_reference.h"

namespace kmeans {
namespace {

// %.17g round-trips a binary64, so a diagnostic can be pasted back into a test.
std::string double_str(double v)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

std::string float_str(float v)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    return std::string(buf);
}

}  // namespace

// The contract is in lockstep.h; what each block below adds is why its criterion is the right
// one for the kernel it judges.
void check_lockstep_iteration(const LockstepSnapshot &snap, const float *X,
                              const Config &config, double m_range, LockstepStats *stats)
{
    const int64_t N = config.N;
    const int64_t D = config.D;
    const int64_t K = config.K;
    const double s_max = s_max_from_range(m_range);
    ++stats->iterations;

    // The very first snapshot also pins down the initial assignment pass of section 2.7, which no
    // later snapshot covers: for t > 0, A(t) was the previous iteration's A(t+1).
    if (snap.iteration == 0) {
        std::vector<int32_t> a_ref(static_cast<size_t>(N), 0);
        cpu::assign(X, N, D, snap.c_in.data(), K, a_ref.data(), nullptr);
        for (int64_t i = 0; i < N; ++i) {
            if (a_ref[static_cast<size_t>(i)] == snap.a_in[static_cast<size_t>(i)]) {
                continue;
            }
            const cpu::TieInfo tie = cpu::tie_info(X + i * D, D, snap.c_in.data(), K);
            if (tie.near_tie) {
                ++stats->assign_near_tie_mismatches;
            } else {
                ++stats->assign_hard_mismatches;
                if (stats->notes.size() < 10) {
                    stats->notes.push_back("initial assign: point " + std::to_string(i) +
                                           " device " +
                                           std::to_string(snap.a_in[static_cast<size_t>(i)]) +
                                           ", reference " +
                                           std::to_string(a_ref[static_cast<size_t>(i)]) +
                                           ", margin " + double_str(tie.margin));
                }
            }
        }
    }

    // Accumulation: the reference is given the device's own A(t).
    std::vector<double> sums_ref(static_cast<size_t>(K * D), 0.0);
    std::vector<uint32_t> counts_ref(static_cast<size_t>(K), 0);
    cpu::accumulate(X, N, D, snap.a_in.data(), K, sums_ref.data(), counts_ref.data());

    uint64_t total_counted = 0;
    for (int64_t k = 0; k < K; ++k) {
        const uint32_t got = snap.counts[static_cast<size_t>(k)];
        total_counted += got;
        if (got != counts_ref[static_cast<size_t>(k)]) {
            ++stats->count_mismatches;
            if (stats->notes.size() < 10) {
                stats->notes.push_back("n[" + std::to_string(k) + "] device " +
                                       std::to_string(got) + ", reference " +
                                       std::to_string(counts_ref[static_cast<size_t>(k)]));
            }
        }
    }
    // RED-005 / FR-009: every point contributes to exactly one cluster, so the counts sum to N.
    if (total_counted != static_cast<uint64_t>(N)) {
        ++stats->count_mismatches;
        if (stats->notes.size() < 10) {
            stats->notes.push_back("counts sum to " + std::to_string(total_counted) +
                                   ", want N = " + std::to_string(N));
        }
    }

    for (int64_t k = 0; k < K; ++k) {
        // |S_kj| <= n_k * M, and binary64 accumulation of n_k terms loses far less than 1e-12 of
        // that, so the bound is proportional to the largest value the sum could take.
        const double scale = static_cast<double>(counts_ref[static_cast<size_t>(k)]) * s_max;
        const double tol = 1e-12 * (scale > s_max ? scale : s_max);
        for (int64_t j = 0; j < D; ++j) {
            const size_t e = static_cast<size_t>(k * D + j);
            const double diff = std::fabs(snap.sums[e] - sums_ref[e]);
            if (diff > stats->max_sum_diff) {
                stats->max_sum_diff = diff;
            }
            if (!(diff <= tol)) {  // negated so a NaN counts as a violation
                ++stats->sum_violations;
                if (stats->notes.size() < 10) {
                    stats->notes.push_back("S[" + std::to_string(k) + "][" + std::to_string(j) +
                                           "] device " + double_str(snap.sums[e]) +
                                           ", reference " + double_str(sums_ref[e]) +
                                           ", tolerance " + double_str(tol));
                }
            }
        }
    }

    // Update: the reference update of the same (S, n) must reproduce the device's C(t+1). The
    // reference is fed its own sums, so this checks the division and the DEC-003 retention, and
    // tolerates only the one-ulp difference NUM-006 allows.
    std::vector<float> c_ref(static_cast<size_t>(K * D), 0.0f);
    cpu::update(sums_ref.data(), counts_ref.data(), K, D, snap.c_in.data(), c_ref.data());
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t j = 0; j < D; ++j) {
            const size_t e = static_cast<size_t>(k * D + j);
            const double diff = std::fabs(static_cast<double>(snap.c_out[e]) -
                                          static_cast<double>(c_ref[e]));
            if (diff > stats->max_centroid_diff) {
                stats->max_centroid_diff = diff;
            }
            if (!centroid_agrees(snap.c_out[e], c_ref[e], s_max)) {
                ++stats->centroid_violations;
                if (stats->notes.size() < 10) {
                    stats->notes.push_back("C[" + std::to_string(k) + "][" + std::to_string(j) +
                                           "] device " + float_str(snap.c_out[e]) +
                                           ", reference " + float_str(c_ref[e]));
                }
            }
            if (counts_ref[static_cast<size_t>(k)] == 0u && snap.c_out[e] != snap.c_in[e]) {
                ++stats->centroid_violations;  // DEC-003: retained bit-exactly
                if (stats->notes.size() < 10) {
                    stats->notes.push_back("empty cluster " + std::to_string(k) +
                                           " was not retained bit-exactly");
                }
            }
        }
    }

    // Movement: judged against the device's own centroids and counts, so this isolates the
    // movement and max_reduce kernels from any difference in the centroids themselves.
    const double delta_ref = cpu::movement(snap.c_in.data(), snap.c_out.data(),
                                           snap.counts.data(), K, D);
    const double delta_diff = std::fabs(delta_ref - snap.delta);
    if (delta_diff > stats->max_delta_diff) {
        stats->max_delta_diff = delta_diff;
    }
    const double delta_tol = 1e-12 * (delta_ref > 1.0 ? delta_ref : 1.0);
    if (!(delta_diff <= delta_tol)) {
        ++stats->delta_violations;
        if (stats->notes.size() < 10) {
            stats->notes.push_back("Delta device " + double_str(snap.delta) + ", reference " +
                                   double_str(delta_ref));
        }
    }

    // Assignment: the reference is given the device's own C(t+1).
    const int64_t near_ties_before = stats->assign_near_tie_mismatches;
    std::vector<int32_t> a_ref(static_cast<size_t>(N), 0);
    const int64_t changed_ref =
        cpu::assign(X, N, D, snap.c_out.data(), K, a_ref.data(), snap.a_in.data());
    for (int64_t i = 0; i < N; ++i) {
        if (a_ref[static_cast<size_t>(i)] == snap.a_out[static_cast<size_t>(i)]) {
            continue;
        }
        const cpu::TieInfo tie = cpu::tie_info(X + i * D, D, snap.c_out.data(), K);
        if (tie.near_tie) {
            ++stats->assign_near_tie_mismatches;
        } else {
            ++stats->assign_hard_mismatches;
            if (stats->notes.size() < 10) {
                stats->notes.push_back("assign: point " + std::to_string(i) + " device " +
                                       std::to_string(snap.a_out[static_cast<size_t>(i)]) +
                                       ", reference " +
                                       std::to_string(a_ref[static_cast<size_t>(i)]) +
                                       ", margin " + double_str(tie.margin) + " > threshold " +
                                       double_str(near_tie_threshold(D)));
            }
        }
    }

    // TEST-U03: the change count is an integer reduction and must be exact. The reference counts
    // differences between the device's A(t) and its own A(t+1); where a near-tie made the two
    // assignments differ in this iteration the counts may legitimately differ, so that case is
    // reported, not failed. The comparison is per iteration: a near-tie seen earlier in the run
    // must not excuse a wrong count here.
    if (changed_ref != snap.changed) {
        if (stats->assign_near_tie_mismatches == near_ties_before) {
            ++stats->change_count_mismatches;
            if (stats->notes.size() < 10) {
                stats->notes.push_back("change count device " + std::to_string(snap.changed) +
                                       ", reference " + std::to_string(changed_ref));
            }
        } else if (stats->notes.size() < 10) {
            stats->notes.push_back("change count differs by " +
                                   std::to_string(changed_ref - snap.changed) +
                                   " with near-ties present (NUM-005), reported not failed");
        }
    }
}

bool lockstep_ok(const LockstepStats &stats)
{
    return stats.iterations > 0 && stats.assign_hard_mismatches == 0 &&
           stats.count_mismatches == 0 && stats.sum_violations == 0 &&
           stats.centroid_violations == 0 && stats.delta_violations == 0 &&
           stats.change_count_mismatches == 0;
}

std::string lockstep_summary(const LockstepStats &stats)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%lld iterations checked; near-tie assignment differences %lld; "
                  "max |dS| %.3e, max |dC| %.3e, max |dDelta| %.3e",
                  static_cast<long long>(stats.iterations),
                  static_cast<long long>(stats.assign_near_tie_mismatches), stats.max_sum_diff,
                  stats.max_centroid_diff, stats.max_delta_diff);
    return std::string(buf);
}

}  // namespace kmeans
