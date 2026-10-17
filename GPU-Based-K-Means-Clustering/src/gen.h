// gen.h — deterministic Gaussian-blob dataset generator (requirements.md FR-003, ARC-02).
//
// Uses the raw output of std::mt19937_64 plus a self-implemented Box-Muller transform. Standard
// library distributions (std::normal_distribution, std::uniform_int_distribution) are deliberately
// not used: their algorithms are unspecified, so the same seed can yield different numbers on a
// different standard library, and FR-003 requires identical parameters to reproduce bit-identical
// files.

#ifndef KMEANS_GEN_H
#define KMEANS_GEN_H

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace kmeans {

struct GenParams {
    int64_t N = 0;
    int64_t D = 0;
    int64_t K = 0;        // K_true = K (PERF-003)
    double sep = 10.0;    // scale of the cluster-centre spread
    double spread = 1.0;  // standard deviation of the points around their centre
    uint64_t seed = 0;
};

// Raw-output wrapper. Every draw below consumes a documented, fixed number of 64-bit words, so the
// generation order fixes the stream and the output is reproducible.
class Rng {
public:
    explicit Rng(uint64_t seed) : engine_(seed) {}

    // Uniform in the open interval (0, 1): the top 53 bits plus a half-step, so the result is
    // never exactly 0 (which would make std::log in Box-Muller return -inf) and never exactly 1.
    double uniform_open();

    // Standard normal via Box-Muller. Both values of each pair are used, so two normals cost one
    // pair of uniforms; the second is cached.
    double normal();

    // Uniform integer in [0, bound) by rejection, which is unbiased for every bound.
    uint64_t bounded(uint64_t bound);

private:
    std::mt19937_64 engine_;
    bool has_cached_normal_ = false;
    double cached_normal_ = 0.0;
};

struct Dataset {
    std::vector<float> X;    // N x D row-major
    std::vector<float> C0;   // K x D row-major
    std::string dataset_id;
    double min_centre_distance = 0.0;  // smallest pairwise centre distance, measured not assumed
    double spread = 0.0;               // echoed so the caller can report separation in sigma units
};

// Identifier derived from every generator parameter (FR-003), so a dataset is traceable from a
// benchmark CSV row back to the exact inputs that produced it.
std::string dataset_id(const GenParams &p);

// Generation order, fixed by FR-003 and by this function:
//   1. K centre vectors, each coordinate sep * normal()
//   2. N points: point i belongs to centre i % K and gets spread * normal() per coordinate
//   3. a seeded Fisher-Yates permutation of [0, N)
//   4. initial centroids: the points at permutation indices floor(q*N/K), q in [0, K)
Dataset generate(const GenParams &p);

}  // namespace kmeans

#endif  // KMEANS_GEN_H
