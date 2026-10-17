#include "cpu_reference.h"

#include <cfloat>
#include <cmath>

namespace kmeans {
namespace cpu {

float distance2_fp32(const float *x, const float *c, int64_t D)
{
    // NUM-002: each term is formed in binary32 and accumulated in binary32 in ascending j,
    // starting from 0. The order is never changed, and -ffp-contract=off keeps the compiler from
    // fusing the multiply and add, so this is the exact value the device must reproduce.
    float acc = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
        const float diff = x[j] - c[j];
        acc += diff * diff;
    }
    return acc;
}

double distance2_fp64(const float *x, const float *c, int64_t D)
{
    double acc = 0.0;
    for (int64_t j = 0; j < D; ++j) {
        const double diff = static_cast<double>(x[j]) - static_cast<double>(c[j]);
        acc += diff * diff;
    }
    return acc;
}

int32_t nearest_centroid(const float *x, int64_t D, const float *C, int64_t K)
{
    // DEC-005: start at k = 0 and replace only on strictly smaller distance, so exactly equal
    // computed binary32 distances keep the lowest index.
    int32_t best_k = 0;
    float best_d = distance2_fp32(x, C, D);
    for (int64_t k = 1; k < K; ++k) {
        const float d = distance2_fp32(x, C + k * D, D);
        if (d < best_d) {
            best_d = d;
            best_k = static_cast<int32_t>(k);
        }
    }
    return best_k;
}

int64_t assign(const float *X, int64_t N, int64_t D, const float *C, int64_t K, int32_t *A,
               const int32_t *A_prev)
{
    int64_t changed = 0;
    for (int64_t i = 0; i < N; ++i) {
        const int32_t k = nearest_centroid(X + i * D, D, C, K);
        if (A_prev != nullptr && A_prev[i] != k) {
            ++changed;
        }
        A[i] = k;
    }
    return A_prev == nullptr ? N : changed;
}

TieInfo tie_info(const float *x, int64_t D, const float *C, int64_t K)
{
    TieInfo info;
    info.d1 = HUGE_VAL;
    info.d2 = HUGE_VAL;
    info.k1 = 0;
    info.k2 = K > 1 ? 1 : 0;

    for (int64_t k = 0; k < K; ++k) {
        const double d = distance2_fp64(x, C + k * D, D);
        if (d < info.d1) {
            info.d2 = info.d1;
            info.k2 = info.k1;
            info.d1 = d;
            info.k1 = static_cast<int32_t>(k);
        } else if (d < info.d2) {
            info.d2 = d;
            info.k2 = static_cast<int32_t>(k);
        }
    }

    if (K < 2) {
        // With one centroid there is no competitor, so no point can be a near-tie.
        info.d2 = info.d1;
        info.margin = HUGE_VAL;
        info.near_tie = false;
        return info;
    }
    const double denom = info.d2 > static_cast<double>(FLT_MIN) ? info.d2
                                                                : static_cast<double>(FLT_MIN);
    info.margin = (info.d2 - info.d1) / denom;
    info.near_tie = info.margin <= near_tie_threshold(D);
    return info;
}

void accumulate(const float *X, int64_t N, int64_t D, const int32_t *A, int64_t K, double *S,
                uint32_t *n)
{
    for (int64_t e = 0; e < K * D; ++e) {
        S[e] = 0.0;
    }
    for (int64_t k = 0; k < K; ++k) {
        n[k] = 0;
    }
    // §2.4: ascending i, binary64 sums, exact integer counts. The reference fixes this order so
    // that Approach B (which also has a fixed order) can be compared against it meaningfully.
    for (int64_t i = 0; i < N; ++i) {
        const int64_t k = A[i];
        n[k] += 1;
        const float *row = X + i * D;
        double *sums = S + k * D;
        for (int64_t j = 0; j < D; ++j) {
            sums[j] += static_cast<double>(row[j]);
        }
    }
}

void update(const double *S, const uint32_t *n, int64_t K, int64_t D, const float *C_cur,
            float *C_next)
{
    for (int64_t k = 0; k < K; ++k) {
        const float *cur = C_cur + k * D;
        float *next = C_next + k * D;
        if (n[k] == 0) {
            // DEC-003: retain the previous centroid bit-exactly; no division is performed.
            for (int64_t j = 0; j < D; ++j) {
                next[j] = cur[j];
            }
            continue;
        }
        const double count = static_cast<double>(n[k]);
        const double *sums = S + k * D;
        for (int64_t j = 0; j < D; ++j) {
            // §2.5: divide in binary64 and round the quotient once to binary32.
            next[j] = static_cast<float>(sums[j] / count);
        }
    }
}

double movement(const float *C_cur, const float *C_next, const uint32_t *n, int64_t K, int64_t D)
{
    double max_delta = 0.0;
    for (int64_t k = 0; k < K; ++k) {
        if (n[k] == 0) {
            continue;  // §2.6: empty clusters have delta_k = 0 and are excluded from the maximum
        }
        const float *cur = C_cur + k * D;
        const float *next = C_next + k * D;
        double sum = 0.0;
        for (int64_t j = 0; j < D; ++j) {
            const double diff = static_cast<double>(next[j]) - static_cast<double>(cur[j]);
            sum += diff * diff;
        }
        const double delta = std::sqrt(sum);
        if (delta > max_delta) {
            max_delta = delta;
        }
    }
    return max_delta;
}

double objective(const float *X, int64_t N, int64_t D, const int32_t *A, const float *C)
{
    double j_total = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        j_total += distance2_fp64(X + i * D, C + static_cast<int64_t>(A[i]) * D, D);
    }
    return j_total;
}

Finalization finalize(const float *X, int64_t N, int64_t D, const int32_t *A_final,
                      const float *C_final, int64_t K, double m_range)
{
    Finalization out;
    out.counts.assign(static_cast<size_t>(K), 0);
    out.means.assign(static_cast<size_t>(K * D), 0.0f);

    std::vector<double> S(static_cast<size_t>(K * D), 0.0);
    accumulate(X, N, D, A_final, K, S.data(), out.counts.data());
    update(S.data(), out.counts.data(), K, D, C_final, out.means.data());

    // §2.8 step 2: P2 holds when every non-empty cluster's reported centroid agrees with the mean
    // of its assigned points within NUM-006. Empty clusters are not evaluated: their centroid is
    // retained by DEC-003 and is not a mean of anything.
    const double s_max = s_max_from_range(m_range);
    bool consistent = true;
    double worst = 0.0;
    for (int64_t k = 0; k < K; ++k) {
        if (out.counts[static_cast<size_t>(k)] == 0) {
            continue;
        }
        for (int64_t j = 0; j < D; ++j) {
            const size_t e = static_cast<size_t>(k * D + j);
            const float reported = C_final[e];
            const float mean = out.means[e];
            const double diff =
                std::fabs(static_cast<double>(reported) - static_cast<double>(mean));
            if (diff > worst) {
                worst = diff;
            }
            if (!centroid_agrees(reported, mean, s_max)) {
                consistent = false;
            }
        }
    }
    out.centroid_consistent = consistent;
    out.max_centroid_difference = worst;
    return out;
}

Result run(const float *X, int64_t N, int64_t D, const float *C0, int64_t K, int64_t max_iter,
           double tol, double m_range, bool fixed_iters)
{
    Result result;
    std::vector<float> c_cur(C0, C0 + K * D);
    std::vector<float> c_next(static_cast<size_t>(K * D), 0.0f);
    std::vector<int32_t> a_cur(static_cast<size_t>(N), 0);
    std::vector<int32_t> a_prev(static_cast<size_t>(N), 0);
    std::vector<double> sums(static_cast<size_t>(K * D), 0.0);
    std::vector<uint32_t> counts(static_cast<size_t>(K), 0);

    // §2.7: one assignment pass before the loop, so A is always the assignment of the current C.
    assign(X, N, D, c_cur.data(), K, a_cur.data(), nullptr);

    int64_t t = 0;
    Status status = Status::ITERATION_LIMIT;
    double last_delta = 0.0;
    bool has_last_delta = false;

    while (t < max_iter) {
        a_prev = a_cur;

        accumulate(X, N, D, a_cur.data(), K, sums.data(), counts.data());
        update(sums.data(), counts.data(), K, D, c_cur.data(), c_next.data());
        const double delta = movement(c_cur.data(), c_next.data(), counts.data(), K, D);
        c_cur.swap(c_next);  // FR-010: double buffering, pointers swapped after the update

        ++t;  // DEC-006: the iteration count is the number of completed centroid updates
        last_delta = delta;
        has_last_delta = true;

        const int64_t changed = assign(X, N, D, c_cur.data(), K, a_cur.data(), a_prev.data());

        if (fixed_iters) {
            continue;  // FR-020: both criteria disabled, so exactly max_iter updates run
        }
        // DEC-002: assignment stability is checked first and takes reporting precedence.
        if (changed == 0) {
            status = Status::CONVERGED_ASSIGNMENTS;
            break;
        }
        if (delta <= tol) {
            status = Status::CONVERGED_MOVEMENT;  // inclusive comparison (ENV-10)
            break;
        }
    }

    const Finalization fin = finalize(X, N, D, a_cur.data(), c_cur.data(), K, m_range);

    result.assignments = a_cur;
    result.centroids = c_cur;  // §2.8: finalization never replaces C_final
    result.counts = fin.counts;
    result.iterations = t;
    result.status = status;
    result.last_delta = last_delta;
    result.has_last_delta = has_last_delta;
    result.centroid_consistent = fin.centroid_consistent;
    result.assignment_consistent = true;  // P1 holds by construction
    result.objective = objective(X, N, D, a_cur.data(), c_cur.data());
    return result;
}

}  // namespace cpu
}  // namespace kmeans
