#include "gen.h"

#include <cmath>
#include <cstdio>

namespace kmeans {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// FNV-1a over the parameter string, so a dataset ID is short but still parameter-specific.
uint64_t fnv1a(const std::string &text)
{
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char c : text) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ull;
    }
    return hash;
}

}  // namespace

double Rng::uniform_open()
{
    // 53 significant bits is the most a binary64 can hold; adding half a step moves the value off
    // both endpoints without biasing the distribution's mean.
    const uint64_t raw = engine_();
    return (static_cast<double>(raw >> 11) + 0.5) * (1.0 / 9007199254740992.0);  // 2^-53
}

double Rng::normal()
{
    if (has_cached_normal_) {
        has_cached_normal_ = false;
        return cached_normal_;
    }
    const double u1 = uniform_open();
    const double u2 = uniform_open();
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = kTwoPi * u2;
    cached_normal_ = radius * std::sin(angle);
    has_cached_normal_ = true;
    return radius * std::cos(angle);
}

uint64_t Rng::bounded(uint64_t bound)
{
    if (bound <= 1) {
        return 0;
    }
    // Rejection sampling: discard the incomplete final block of the 2^64 range so every value in
    // [0, bound) is equally likely. std::uniform_int_distribution would do this too, but its
    // algorithm is implementation-defined and FR-003 needs a fixed one.
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % bound) - 1;
    uint64_t raw = engine_();
    while (raw > limit) {
        raw = engine_();
    }
    return raw % bound;
}

std::string dataset_id(const GenParams &p)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), "n%lld_d%lld_k%lld_sep%.6g_spread%.6g_seed%llu",
                  static_cast<long long>(p.N), static_cast<long long>(p.D),
                  static_cast<long long>(p.K), p.sep, p.spread,
                  static_cast<unsigned long long>(p.seed));
    const std::string base(buf);
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "-%016llx",
                  static_cast<unsigned long long>(fnv1a(base)));
    return base + suffix;
}

Dataset generate(const GenParams &p)
{
    Dataset out;
    out.dataset_id = dataset_id(p);
    out.spread = p.spread;
    const int64_t N = p.N;
    const int64_t D = p.D;
    const int64_t K = p.K;

    Rng rng(p.seed);

    // 1. Cluster centres. Drawing them from the same normal stream keeps the generator simple and
    // deterministic; the realised separation is measured below rather than assumed, because
    // §10.3 calls a dataset well-separated only at separation >= 10 * spread.
    std::vector<double> centres(static_cast<size_t>(K * D), 0.0);
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t j = 0; j < D; ++j) {
            centres[static_cast<size_t>(k * D + j)] = p.sep * rng.normal();
        }
    }

    // 2. Points: point i belongs to centre i % K, which spreads the clusters evenly without
    // consuming randomness for the membership choice.
    out.X.assign(static_cast<size_t>(N * D), 0.0f);
    for (int64_t i = 0; i < N; ++i) {
        const int64_t k = i % K;
        for (int64_t j = 0; j < D; ++j) {
            const double value = centres[static_cast<size_t>(k * D + j)] + p.spread * rng.normal();
            out.X[static_cast<size_t>(i * D + j)] = static_cast<float>(value);
        }
    }

    // 3. Fisher-Yates permutation of the point indices, descending as in the classical algorithm.
    std::vector<int64_t> perm(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        perm[static_cast<size_t>(i)] = i;
    }
    for (int64_t i = N - 1; i > 0; --i) {
        const uint64_t j = rng.bounded(static_cast<uint64_t>(i) + 1);
        const int64_t tmp = perm[static_cast<size_t>(i)];
        perm[static_cast<size_t>(i)] = perm[static_cast<size_t>(j)];
        perm[static_cast<size_t>(j)] = tmp;
    }

    // 4. FR-003: the initial centroids are the points at permutation indices floor(q*N/K). With
    // K > N the same point can be chosen more than once, which VAL-008 then flags as W-DUPC.
    out.C0.assign(static_cast<size_t>(K * D), 0.0f);
    for (int64_t q = 0; q < K; ++q) {
        const int64_t slot = static_cast<int64_t>((static_cast<uint64_t>(q) *
                                                   static_cast<uint64_t>(N)) /
                                                  static_cast<uint64_t>(K));
        const int64_t point = perm[static_cast<size_t>(slot % N)];
        for (int64_t j = 0; j < D; ++j) {
            out.C0[static_cast<size_t>(q * D + j)] = out.X[static_cast<size_t>(point * D + j)];
        }
    }

    // Measured minimum pairwise centre distance, in the generated binary32 values' own units.
    double min_sq = -1.0;
    for (int64_t a = 0; a < K; ++a) {
        for (int64_t b = a + 1; b < K; ++b) {
            double sum = 0.0;
            for (int64_t j = 0; j < D; ++j) {
                const double diff = centres[static_cast<size_t>(a * D + j)] -
                                    centres[static_cast<size_t>(b * D + j)];
                sum += diff * diff;
            }
            if (min_sq < 0.0 || sum < min_sq) {
                min_sq = sum;
            }
        }
    }
    out.min_centre_distance = min_sq < 0.0 ? 0.0 : std::sqrt(min_sq);
    return out;
}

}  // namespace kmeans
