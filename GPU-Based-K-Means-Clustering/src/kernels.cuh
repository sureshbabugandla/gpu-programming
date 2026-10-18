// kernels.cuh — host-side launch interface for the CUDA-002 kernel set (ARC-05, ARC-06, ARC-07,
// ARC-08).
//
// The kernels themselves stay inside kernels.cu. The driver only ever calls the launchers below,
// which keeps every launch configuration in one place (CUDA-007) and means no translation unit
// other than kernels.cu contains a kernel launch, so the build needs no relocatable device code.
//
// Every launcher is stream-ordered on the single stream of DEC-010 and checks cudaGetLastError
// after the launch (CUDA-008).

#ifndef KMEANS_KERNELS_CUH
#define KMEANS_KERNELS_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "common.h"
#include "launch_config.h"

namespace kmeans {
namespace gpu {

// The 16 bytes copied to pinned host memory once per iteration (CUDA-011, MEM-001). Keeping both
// scalars in one device object is what makes it a single 16-byte transfer, which is what the
// transfer-discipline check of TEST-P03 looks for in the Nsight Systems trace.
struct ConvergenceScalars {
    double delta;                // Delta of the update just performed (section 2.6)
    unsigned long long changed;  // number of points whose assignment differed (FR-007)
};

// FR-004..FR-007, DEC-008: one thread per point, grid-stride, ascending k and j.
// `count_changes` is false for the initial pass of section 2.7, which has no previous assignment
// to compare against and therefore never reads A.
void launch_assign(const float *x_col, int64_t N, int64_t D, const float *centroids, int64_t K,
                   int32_t *assignments, unsigned long long *changed, bool count_changes,
                   int block, int sm_count, cudaStream_t stream);

// Section 8, SMEM-001..SMEM-004: the same contract as launch_assign with the centroid tile staged
// in shared memory. T_K and T_D are the compile-time constants of launch_config.h, so selecting
// other tile sizes is a rebuild with -DKMEANS_TILE_K / -DKMEANS_TILE_D (SMEM-001).
void launch_assign_smem(const float *x_col, int64_t N, int64_t D, const float *centroids,
                        int64_t K, int32_t *assignments, unsigned long long *changed,
                        bool count_changes, int block, int sm_count, cudaStream_t stream);

// Section 6, ATOM-002..ATOM-004: one thread per element of X, grid-stride over N*D.
void launch_accum_atomic(const float *x_col, int64_t N, int64_t D, const int32_t *assignments,
                         int64_t K, double *sums, unsigned int *counts, int block, int sm_count,
                         cudaStream_t stream);

// RED-002: one logical worker per (g, j), grid-stride over G*D.
void launch_reduce_stage1(const float *x_row, int64_t N, int64_t D, const int32_t *assignments,
                          int64_t K, int64_t G, double *partials, unsigned int *partial_counts,
                          int block, int sm_count, cudaStream_t stream);

// RED-003: one 32x8 block per (cluster, 32-dimension tile), grid-stride over tiles.
void launch_reduce_stage2(const double *partials, const unsigned int *partial_counts, int64_t D,
                          int64_t K, int64_t G, double *sums, unsigned int *counts, int sm_count,
                          cudaStream_t stream);

// Section 2.5 / FR-010 / FR-011: grid-stride over K*D into the next centroid buffer.
void launch_update(const double *sums, const unsigned int *counts, int64_t K, int64_t D,
                   const float *centroids_cur, float *centroids_next, int block, int sm_count,
                   cudaStream_t stream);

// Section 2.6 / FR-012: delta_k^2 per cluster, then one maximum per block.
void launch_movement(const float *centroids_cur, const float *centroids_next,
                     const unsigned int *counts, int64_t K, int64_t D, double *block_max,
                     int block, int sm_count, cudaStream_t stream);

// Final maximum over the per-block maxima in a single block, with the one square root.
void launch_max_reduce(const double *block_max, int block_count, double *delta,
                       cudaStream_t stream);

}  // namespace gpu
}  // namespace kmeans

#endif  // KMEANS_KERNELS_CUH
