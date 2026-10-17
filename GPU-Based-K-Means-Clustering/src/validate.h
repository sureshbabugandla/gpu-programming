// validate.h — input validation rules VAL-001..VAL-010 (requirements.md §9.1, FR-001).
//
// Every rule is applied before any device allocation, and the first violation names the parameter
// and the rule ID (BLD-006) and maps to an exit code (BLD-005). Rules that accept the input with a
// warning (VAL-003 K > N, VAL-008 duplicate centroids) record the warning ID instead of failing.

#ifndef KMEANS_VALIDATE_H
#define KMEANS_VALIDATE_H

#include <string>
#include <vector>

#include "common.h"

namespace kmeans {

struct Validation {
    bool ok = true;
    std::string rule;     // failing rule ID, empty when ok
    std::string message;  // cause and offending value
    int exit_code = kExitOk;
    std::vector<std::string> warnings;  // e.g. "W-KGTN", "W-DUPC", in the order detected

    bool has_warning(const std::string &id) const;
    // Emit the failure and every warning in BLD-006 format.
    void report() const;
};

// VAL-001, VAL-002, VAL-003, VAL-004, VAL-005. Sizes arrive as 64-bit integers (§2.1) and are
// rejected unless they fit the int32 assignment and uint32 count types.
Validation validate_parameters(int64_t N, int64_t D, int64_t K, int64_t max_iter, double tol);

// VAL-010: option values within their documented ranges and the device limits. Device limits are
// passed in so the rule is testable without a GPU; pass 0 for a limit that is not yet known.
Validation validate_options(int block_size, int64_t reduce_chunks, int tile_k, int tile_d,
                            int device_max_threads_per_block, int64_t device_shared_mem_per_block);

// VAL-007 range bound: M = max(max|x_ij|, max|c0_kj|). Returns M, which NUM-006 also needs as the
// scale S_max. Returns a negative value only if the arrays are empty.
double range_bound(const std::vector<float> &X, const std::vector<float> &C0);

// VAL-006 (finite values, first offending row and column reported), VAL-007 (4 D M^2 <= 2^126
// evaluated in binary64) and VAL-008 (shape of the centroid file, duplicate centroids warned).
// `m_range` receives the M of VAL-007 when validation succeeds.
Validation validate_data(const std::vector<float> &X, int64_t N, int64_t D,
                         const std::vector<float> &C0, int64_t K, double *m_range);

// VAL-009: total required device memory (MEM-002) must fit in the free device memory less a 5 %
// safety margin. `free_bytes` comes from cudaMemGetInfo in M2 and from the test in M1.
Validation validate_capacity(int64_t N, int64_t D, int64_t K, int64_t reduce_chunks,
                             uint64_t free_bytes);

}  // namespace kmeans

#endif  // KMEANS_VALIDATE_H
