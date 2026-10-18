// kernels.cu — the CUDA-002 kernel set.
//
// Thread mapping, data ownership and synchronisation points for each kernel are documented at the
// kernel and summarised in the notebook (tools/cells/05_kernel_contracts.md and
// 05_reduction_contracts.md). Three rules hold throughout:
//
//   * Every global index is guarded against its range (CUDA-005), and threads that fall outside
//     skip work but still reach every __syncthreads().
//   * Every element offset is formed in unsigned 64-bit arithmetic (CUDA-006).
//   * No kernel reads data written by another block of the same launch; phase boundaries are
//     kernel boundaries on one stream (CUDA-004, DEC-010).

#include "kernels.cuh"

#include "cuda_util.cuh"

namespace kmeans {
namespace gpu {
namespace {

// Grid-stride step, in 64-bit (CUDA-006).
__device__ inline uint64_t grid_stride()
{
    return static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
}

__device__ inline uint64_t global_thread_index()
{
    return static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
           static_cast<uint64_t>(threadIdx.x);
}

// CUDA-007 validates the block size as a multiple of 32, which is not necessarily a power of two
// (96, 160, 192 are all legal). The block reductions below therefore start from the next power of
// two at or above blockDim.x and guard the partner index, which is correct for any block size.
__device__ inline unsigned reduction_start_stride(unsigned block_dim)
{
    unsigned stride = 1;
    while (stride < block_dim) {
        stride <<= 1;
    }
    return stride >> 1;
}

// --- FR-004..FR-007: assignment ----------------------------------------------------------------
//
// Thread mapping   one thread per point i, 1-D grid-stride (DEC-008).
// Ownership        thread i is the only reader and writer of assignments[i], so the in-place
//                  update of FR-007 needs no synchronisation. x_col and centroids are read-only.
// Synchronisation  none for the assignment itself. The change count is reduced through shared
//                  memory, so every thread of the block reaches every barrier, including threads
//                  whose grid-stride loop ran zero times (CUDA-005).
// Numerics         NUM-002: terms in binary32, accumulated in binary32 in ascending j from 0, and
//                  never reordered. DEC-005: k ascends from 0 and the best candidate is replaced
//                  only on a strictly smaller distance, so exactly equal computed distances keep
//                  the lowest index.
// Coalescing       x_col is dimension-major, so for fixed j consecutive threads read consecutive
//                  addresses; the centroid read is uniform across the warp and broadcasts
//                  (CUDA-009).
__global__ void assign_kernel(const float *__restrict__ x_col, int64_t N, int64_t D,
                              const float *__restrict__ centroids, int64_t K,
                              int32_t *__restrict__ assignments,
                              unsigned long long *__restrict__ changed, int count_changes)
{
    extern __shared__ unsigned long long s_changed[];
    const unsigned tid = threadIdx.x;
    const uint64_t n = static_cast<uint64_t>(N);
    unsigned long long local_changed = 0;

    for (uint64_t i = global_thread_index(); i < n; i += grid_stride()) {
        // k = 0 initialises the best candidate (DEC-005).
        float best = 0.0f;
        for (int64_t j = 0; j < D; ++j) {
            const float diff = x_col[static_cast<uint64_t>(j) * n + i] - centroids[j];
            best += diff * diff;
        }
        int32_t best_k = 0;

        for (int64_t k = 1; k < K; ++k) {
            const float *centroid = centroids + static_cast<uint64_t>(k) * static_cast<uint64_t>(D);
            float distance = 0.0f;
            for (int64_t j = 0; j < D; ++j) {
                const float diff = x_col[static_cast<uint64_t>(j) * n + i] - centroid[j];
                distance += diff * diff;
            }
            if (distance < best) {  // strictly less: ties keep the lower k
                best = distance;
                best_k = static_cast<int32_t>(k);
            }
        }

        if (count_changes != 0 && assignments[i] != best_k) {
            ++local_changed;
        }
        assignments[i] = best_k;
    }

    // CUDA-002: one 64-bit atomicAdd per block, not per point. Integer atomics are exact, so the
    // counter is deterministic and does not compromise the atomic-free accumulation of Approach B.
    s_changed[tid] = local_changed;
    __syncthreads();
    for (unsigned stride = reduction_start_stride(blockDim.x); stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            s_changed[tid] += s_changed[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0 && count_changes != 0) {
        atomicAdd(changed, s_changed[0]);
    }
}

// --- Section 6: Approach A, atomic accumulation ------------------------------------------------
//
// Thread mapping   one thread per element e of X, 1-D grid-stride over N*D, with
//                  i = e mod N and j = e / N over the dimension-major copy (ATOM-002, CUDA-009).
//                  Since x_col holds element (i, j) at j*N + i, that index is exactly e, so the
//                  read is a straight sequential walk and coalesces.
// Ownership        sums and counts are shared by every thread; all writes are atomic
//                  read-modify-writes, so no update is lost (ATOM-007). The caller zeroes both
//                  buffers on the same stream before the launch (ATOM-001).
// Synchronisation  none inside the kernel. The ordering against `update` is the kernel boundary
//                  of ATOM-005 / DEC-010.
// Numerics         binary64 accumulation (DEC-007) in an order the hardware chooses, so S may
//                  differ in its last bits between runs; counts are integers and are exact.
__global__ void accum_atomic_kernel(const float *__restrict__ x_col, int64_t N, int64_t D,
                                    const int32_t *__restrict__ assignments, int64_t K,
                                    double *__restrict__ sums, unsigned int *__restrict__ counts)
{
    const uint64_t n = static_cast<uint64_t>(N);
    const uint64_t total = n * static_cast<uint64_t>(D);

    for (uint64_t e = global_thread_index(); e < total; e += grid_stride()) {
        const uint64_t i = e % n;
        const uint64_t j = e / n;
        const int32_t k = assignments[i];
        // CUDA-005: every global index is guarded. A is always in [0, K) because `assign`
        // produced it; the guard keeps a deliberately corrupted A (fault injection, poisoning)
        // from writing outside the buffers, so memcheck stays clean in those tests.
        if (k < 0 || static_cast<int64_t>(k) >= K) {
            continue;
        }
        atomicAdd(&sums[static_cast<uint64_t>(k) * static_cast<uint64_t>(D) + j],
                  static_cast<double>(x_col[e]));
        if (j == 0) {
            atomicAdd(&counts[k], 1u);  // ATOM-004
        }
    }
}

// --- Section 7 Stage 1: ownership-based privatised accumulation --------------------------------
//
// Thread mapping   one logical worker per (g, j) with flat index w = g*D + j, 1-D grid-stride
//                  over G*D; a thread may process several workers in sequence (RED-002).
// Ownership        worker (g, j) exclusively owns the column P[g][.][j] -- the K elements at
//                  stride D -- and, when j == 0, the count row Q[g][.]. Every element of P and Q
//                  has exactly one writer, so there are no write-write races, no atomics and no
//                  lost updates (RED-004). x_row and assignments are read-only here.
// Synchronisation  none at all inside the kernel. The only ordering needed is Stage 1 -> Stage 2,
//                  which is a kernel boundary (DEC-010).
// Initialisation   step 1 writes zero to every element the worker owns on every call, so no stale
//                  value from an earlier iteration can survive (RED-006, MEM-006, TEST-U06).
// Numerics         RED-009: ascending i within the chunk, so for fixed (N, D, K, G) the order is
//                  fixed and the result is expected to be bitwise reproducible.
// Coalescing       x_row is row-major, so for fixed i the D workers of one chunk read consecutive
//                  addresses (CUDA-009).
__global__ void reduce_stage1_kernel(const float *__restrict__ x_row, int64_t N, int64_t D,
                                     const int32_t *__restrict__ assignments, int64_t K, int64_t G,
                                     double *__restrict__ partials,
                                     unsigned int *__restrict__ partial_counts)
{
    const uint64_t d = static_cast<uint64_t>(D);
    const uint64_t workers = static_cast<uint64_t>(G) * d;

    for (uint64_t w = global_thread_index(); w < workers; w += grid_stride()) {
        const int64_t g = static_cast<int64_t>(w / d);
        const int64_t j = static_cast<int64_t>(w % d);

        // P[g][k][j] lives at (g*K + k)*D + j, so the owned column starts here and strides by D.
        double *owned_sums =
            partials + static_cast<uint64_t>(g) * static_cast<uint64_t>(K) * d + static_cast<uint64_t>(j);
        unsigned int *owned_counts =
            (j == 0) ? partial_counts + static_cast<uint64_t>(g) * static_cast<uint64_t>(K)
                     : nullptr;

        for (int64_t k = 0; k < K; ++k) {
            owned_sums[static_cast<uint64_t>(k) * d] = 0.0;
            if (owned_counts != nullptr) {
                owned_counts[k] = 0u;
            }
        }

        // DEC-001 chunk bounds in 64-bit: g <= G <= N <= 2^31 - 1, so g*N <= 2^62. The chunks
        // share endpoints and cover [0, N) exactly (RED-005).
        const int64_t begin = (g * N) / G;
        const int64_t end = ((g + 1) * N) / G;
        for (int64_t i = begin; i < end; ++i) {
            const int32_t k = assignments[i];
            if (k < 0 || static_cast<int64_t>(k) >= K) {
                continue;  // CUDA-005, as in accum_atomic_kernel
            }
            owned_sums[static_cast<uint64_t>(k) * d] +=
                static_cast<double>(x_row[static_cast<uint64_t>(i) * d + static_cast<uint64_t>(j)]);
            if (owned_counts != nullptr) {
                owned_counts[k] += 1u;
            }
        }
    }
}

// --- Section 7 Stage 2: fixed-order tree combine -----------------------------------------------
//
// Thread mapping   one 32x8 block per tile tau in [0, K*T), T = ceil(D/32), grid-stride over
//                  tiles. Lane x in [0,32) selects j = (tau mod T)*32 + x; lane y in [0,8)
//                  selects the strided subset g = y, y+8, y+16, ... (RED-003).
// Ownership        each element of S and n is written by exactly one thread -- (x, 0) of the tile
//                  that owns that j, and (0, 0) of the j-tile 0 for the count. P and Q are
//                  read-only. Each shared slot is written only by its own thread.
// Synchronisation  three barriers per tile: one before the shared slots are overwritten (so the
//                  previous tile's tree reads have finished), one after they are filled, and one
//                  after each of the three tree levels. tau is derived from blockIdx only, so the
//                  tile loop and the j >= D guards are block-uniform and no barrier is reached by
//                  only part of the block (CUDA-005, RED-003).
// Numerics         RED-009: ascending g within each y-lane, then the fixed pairwise tree with
//                  strides 4, 2, 1. The order depends only on (G, D, K), never on scheduling.
__global__ void reduce_stage2_kernel(const double *__restrict__ partials,
                                     const unsigned int *__restrict__ partial_counts, int64_t D,
                                     int64_t K, int64_t G, double *__restrict__ sums,
                                     unsigned int *__restrict__ counts)
{
    __shared__ double s_sums[kStage2ChunkLanes][kStage2DimTile];
    __shared__ unsigned int s_counts[kStage2ChunkLanes];

    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    const uint64_t d = static_cast<uint64_t>(D);
    const int64_t tiles_per_cluster = (D + kStage2DimTile - 1) / kStage2DimTile;
    const int64_t tile_count = K * tiles_per_cluster;

    for (int64_t tau = blockIdx.x; tau < tile_count; tau += gridDim.x) {
        const int64_t k = tau / tiles_per_cluster;
        const int64_t dim_tile = tau % tiles_per_cluster;
        const int64_t j = dim_tile * kStage2DimTile + x;

        double partial_sum = 0.0;
        if (j < D) {
            for (int64_t g = y; g < G; g += kStage2ChunkLanes) {
                partial_sum += partials[(static_cast<uint64_t>(g) * static_cast<uint64_t>(K) +
                                         static_cast<uint64_t>(k)) *
                                            d +
                                        static_cast<uint64_t>(j)];
            }
        }
        // Counts are reduced by the tiles with j-tile index 0 only, through lane x == 0 (RED-003).
        unsigned int partial_count = 0;
        if (dim_tile == 0 && x == 0) {
            for (int64_t g = y; g < G; g += kStage2ChunkLanes) {
                partial_count +=
                    partial_counts[static_cast<uint64_t>(g) * static_cast<uint64_t>(K) +
                                   static_cast<uint64_t>(k)];
            }
        }

        __syncthreads();  // the previous tile's tree reads are complete before these writes
        s_sums[y][x] = partial_sum;
        if (x == 0) {
            s_counts[y] = partial_count;
        }
        __syncthreads();

        for (int stride = kStage2ChunkLanes / 2; stride >= 1; stride >>= 1) {
            if (y < stride) {
                s_sums[y][x] += s_sums[y + stride][x];
                if (x == 0) {
                    s_counts[y] += s_counts[y + stride];
                }
            }
            __syncthreads();
        }

        if (y == 0 && j < D) {
            sums[static_cast<uint64_t>(k) * d + static_cast<uint64_t>(j)] = s_sums[0][x];
        }
        if (y == 0 && x == 0 && dim_tile == 0) {
            counts[k] = s_counts[0];
        }
    }
}

// --- Section 2.5: centroid update --------------------------------------------------------------
//
// Thread mapping   one thread per centroid element, 1-D grid-stride over K*D.
// Ownership        element (k, j) of centroids_next is written by exactly one thread; sums,
//                  counts and centroids_cur are read-only. Double buffering (FR-010) is what
//                  makes that true: writing into centroids_cur would race with the reads of the
//                  retention branch and with `movement`.
// Synchronisation  none inside the kernel; the boundary to `movement` is a kernel boundary.
// Numerics         section 2.5: the division is performed in binary64 and the quotient is rounded
//                  once to binary32. DEC-003: when the count is zero no division is performed at
//                  all and the previous centroid is copied bit-exactly. Every (k, j) is written
//                  on every call, so centroids_next needs no initialisation (MEM-005).
__global__ void update_kernel(const double *__restrict__ sums,
                              const unsigned int *__restrict__ counts, int64_t K, int64_t D,
                              const float *__restrict__ centroids_cur,
                              float *__restrict__ centroids_next)
{
    const uint64_t d = static_cast<uint64_t>(D);
    const uint64_t total = static_cast<uint64_t>(K) * d;

    for (uint64_t e = global_thread_index(); e < total; e += grid_stride()) {
        const uint64_t k = e / d;
        const unsigned int count = counts[k];
        if (count == 0u) {
            centroids_next[e] = centroids_cur[e];  // DEC-003
        } else {
            centroids_next[e] = static_cast<float>(sums[e] / static_cast<double>(count));
        }
    }
}

// --- Section 2.6: centroid movement ------------------------------------------------------------
//
// Thread mapping   one thread per cluster, 1-D grid-stride over K; each thread computes the whole
//                  squared displacement of its cluster in ascending j.
// Ownership        block_max[blockIdx.x] is written by thread 0 of that block and by no one else.
//                  The centroid buffers and counts are read-only.
// Synchronisation  the per-block maximum is reduced through shared memory, so all threads reach
//                  all barriers; clusters outside the range contribute 0 but still participate.
// Numerics         differences are formed in binary64 from the binary32 centroids and summed in
//                  ascending j (section 2.6). Empty clusters have delta_k = 0 by construction and
//                  are excluded from the maximum, so no 0/0 can appear. The kernel keeps
//                  delta_k^2; the single square root is taken once in max_reduce_kernel.
__global__ void movement_kernel(const float *__restrict__ centroids_cur,
                                const float *__restrict__ centroids_next,
                                const unsigned int *__restrict__ counts, int64_t K, int64_t D,
                                double *__restrict__ block_max)
{
    extern __shared__ double s_max[];
    const unsigned tid = threadIdx.x;
    const uint64_t d = static_cast<uint64_t>(D);
    double local_max = 0.0;

    for (uint64_t k = global_thread_index(); k < static_cast<uint64_t>(K); k += grid_stride()) {
        if (counts[k] == 0u) {
            continue;  // section 2.6: excluded from the maximum
        }
        const uint64_t base = k * d;
        double squared = 0.0;
        for (int64_t j = 0; j < D; ++j) {
            const double diff = static_cast<double>(centroids_next[base + static_cast<uint64_t>(j)]) -
                                static_cast<double>(centroids_cur[base + static_cast<uint64_t>(j)]);
            squared += diff * diff;
        }
        if (squared > local_max) {
            local_max = squared;
        }
    }

    s_max[tid] = local_max;
    __syncthreads();
    for (unsigned stride = reduction_start_stride(blockDim.x); stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_max[blockIdx.x] = s_max[0];
    }
}

// --- Final maximum -----------------------------------------------------------------------------
//
// Thread mapping   a single block (CUDA-002), grid-stride over the per-block maxima.
// Ownership        *delta is written by thread 0 only; block_max is read-only.
// Synchronisation  shared-memory tree with a barrier after each level; one block, so no grid-wide
//                  synchronisation is involved (CUDA-004).
// Numerics         sqrt is applied once, to the maximum of the squared displacements. Because
//                  sqrt is monotone this selects the same cluster as section 2.6 and rounds the
//                  same binary64 sum once, so the value is identical to taking the square root
//                  per cluster and then the maximum.
__global__ void max_reduce_kernel(const double *__restrict__ block_max, int block_count,
                                  double *__restrict__ delta)
{
    extern __shared__ double s_max[];
    const unsigned tid = threadIdx.x;
    double local_max = 0.0;

    for (int b = static_cast<int>(tid); b < block_count; b += static_cast<int>(blockDim.x)) {
        if (block_max[b] > local_max) {
            local_max = block_max[b];
        }
    }

    s_max[tid] = local_max;
    __syncthreads();
    for (unsigned stride = reduction_start_stride(blockDim.x); stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *delta = sqrt(s_max[0]);
    }
}

}  // namespace

// --- launchers ---------------------------------------------------------------------------------

void launch_assign(const float *x_col, int64_t N, int64_t D, const float *centroids, int64_t K,
                   int32_t *assignments, unsigned long long *changed, bool count_changes,
                   int block, int sm_count, cudaStream_t stream)
{
    const int grid = grid_size(N, block, sm_count);
    const size_t shared = static_cast<size_t>(block_reduction_bytes(block));
    assign_kernel<<<grid, block, shared, stream>>>(x_col, N, D, centroids, K, assignments, changed,
                                                   count_changes ? 1 : 0);
    KMEANS_CHECK_LAUNCH("assign");
}

void launch_accum_atomic(const float *x_col, int64_t N, int64_t D, const int32_t *assignments,
                         int64_t K, double *sums, unsigned int *counts, int block, int sm_count,
                         cudaStream_t stream)
{
    uint64_t elements = 0;
    if (!checked_mul(static_cast<uint64_t>(N), static_cast<uint64_t>(D), &elements)) {
        throw CudaError("[CUDA-006] N*D overflows 64-bit arithmetic", kExitInvalidInput);
    }
    const int grid = grid_size(static_cast<int64_t>(elements), block, sm_count);
    accum_atomic_kernel<<<grid, block, 0, stream>>>(x_col, N, D, assignments, K, sums, counts);
    KMEANS_CHECK_LAUNCH("accum_atomic");
}

void launch_reduce_stage1(const float *x_row, int64_t N, int64_t D, const int32_t *assignments,
                          int64_t K, int64_t G, double *partials, unsigned int *partial_counts,
                          int block, int sm_count, cudaStream_t stream)
{
    uint64_t workers = 0;
    if (!checked_mul(static_cast<uint64_t>(G), static_cast<uint64_t>(D), &workers)) {
        throw CudaError("[CUDA-006] G*D overflows 64-bit arithmetic", kExitInvalidInput);
    }
    const int grid = grid_size(static_cast<int64_t>(workers), block, sm_count);
    reduce_stage1_kernel<<<grid, block, 0, stream>>>(x_row, N, D, assignments, K, G, partials,
                                                     partial_counts);
    KMEANS_CHECK_LAUNCH("reduce_stage1");
}

void launch_reduce_stage2(const double *partials, const unsigned int *partial_counts, int64_t D,
                          int64_t K, int64_t G, double *sums, unsigned int *counts, int sm_count,
                          cudaStream_t stream)
{
    // One block per tile per grid-stride step; the block shape is fixed by RED-003.
    const int grid = grid_size(stage2_tiles(K, D), 1, sm_count);
    const dim3 block(kStage2DimTile, kStage2ChunkLanes);
    reduce_stage2_kernel<<<grid, block, 0, stream>>>(partials, partial_counts, D, K, G, sums,
                                                     counts);
    KMEANS_CHECK_LAUNCH("reduce_stage2");
}

void launch_update(const double *sums, const unsigned int *counts, int64_t K, int64_t D,
                   const float *centroids_cur, float *centroids_next, int block, int sm_count,
                   cudaStream_t stream)
{
    uint64_t elements = 0;
    if (!checked_mul(static_cast<uint64_t>(K), static_cast<uint64_t>(D), &elements)) {
        throw CudaError("[CUDA-006] K*D overflows 64-bit arithmetic", kExitInvalidInput);
    }
    const int grid = grid_size(static_cast<int64_t>(elements), block, sm_count);
    update_kernel<<<grid, block, 0, stream>>>(sums, counts, K, D, centroids_cur, centroids_next);
    KMEANS_CHECK_LAUNCH("update");
}

void launch_movement(const float *centroids_cur, const float *centroids_next,
                     const unsigned int *counts, int64_t K, int64_t D, double *block_max,
                     int block, int sm_count, cudaStream_t stream)
{
    const int grid = movement_grid_size(K, block, sm_count);
    const size_t shared = static_cast<size_t>(block_reduction_bytes(block));
    movement_kernel<<<grid, block, shared, stream>>>(centroids_cur, centroids_next, counts, K, D,
                                                     block_max);
    KMEANS_CHECK_LAUNCH("movement");
}

void launch_max_reduce(const double *block_max, int block_count, double *delta,
                       cudaStream_t stream)
{
    // A single block (CUDA-002). The movement grid is capped at kMaxMovementBlocks = 1024, which
    // is also the maximum block size, so one thread per per-block maximum always fits; the block
    // is rounded up to a warp so the shared-memory tree has no partial warp.
    int threads = block_count < kMaxMovementBlocks ? block_count : kMaxMovementBlocks;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) {
        threads = 32;
    }
    const size_t shared = static_cast<size_t>(block_reduction_bytes(threads));
    max_reduce_kernel<<<1, threads, shared, stream>>>(block_max, block_count, delta);
    KMEANS_CHECK_LAUNCH("max_reduce");
}

}  // namespace gpu
}  // namespace kmeans
