// launch_config.h — launch geometry and buffer sizing arithmetic (requirements.md CUDA-007,
// RED-006, MEM-003, MEM-004, SMEM-003, DEC-001).
//
// Deliberately free of CUDA constructs: every number a kernel launch needs is computed here from
// plain integers, so the G selection of RED-006 and the chunk partition of DEC-001 can be unit
// tested with g++ on a machine that has no GPU (CLAUDE.md). The .cu translation units include this
// header and only turn its results into launch arguments.

#ifndef KMEANS_LAUNCH_CONFIG_H
#define KMEANS_LAUNCH_CONFIG_H

#include <cstdint>
#include <string>

#include "common.h"

namespace kmeans {

// --- CUDA-007 grid sizing ----------------------------------------------------------------------

// Every kernel uses a 1-D grid with a grid-stride loop, so the grid may be smaller than the work:
// grid = min(ceil(work / block), 32 * SM count). A grid of 0 would launch nothing, so one block is
// the floor even for zero work; the bounds checks of CUDA-005 make that block a no-op.
inline int grid_size(int64_t work, int block, int sm_count)
{
    if (work <= 0 || block <= 0) {
        return 1;
    }
    const int64_t needed = (work + block - 1) / block;
    const int64_t cap = 32 * static_cast<int64_t>(sm_count > 0 ? sm_count : 1);
    const int64_t chosen = needed < cap ? needed : cap;
    return static_cast<int>(chosen < 1 ? 1 : chosen);
}

// `movement` writes one maximum per block and `max_reduce` reduces that array in a single block.
// Capping the movement grid at 1024 (the "unless overridden" clause of CUDA-007) serves two
// purposes: max_reduce can use one thread per block maximum in a single 1024-thread block, and the
// cap equals the `movement_blocks` default of device_memory_bytes (common.h), so the MEM-002
// capacity model and the real allocation agree.
constexpr int kMaxMovementBlocks = 1024;

inline int movement_grid_size(int64_t K, int block, int sm_count)
{
    const int grid = grid_size(K, block, sm_count);
    return grid < kMaxMovementBlocks ? grid : kMaxMovementBlocks;
}

// --- Section 7 Stage 2 tile geometry -----------------------------------------------------------
// RED-003 fixes the Stage 2 block at 32 x 8: lane x selects j within a 32-dimension tile, lane y
// selects a strided subset of g. Both are compile-time constants because the fixed pairwise tree
// (strides 4, 2, 1) is written for exactly eight y-lanes.
constexpr int kStage2DimTile = 32;
constexpr int kStage2ChunkLanes = 8;

// Number of 32-dimension tiles per cluster, T = ceil(D / 32).
inline int64_t stage2_dim_tiles(int64_t D)
{
    return (D + kStage2DimTile - 1) / kStage2DimTile;
}

// Total Stage 2 tiles, K * T, enumerated by a 1-D grid-stride index (DEC-001).
inline int64_t stage2_tiles(int64_t K, int64_t D)
{
    return K * stage2_dim_tiles(D);
}

// --- SMEM-001 / SMEM-003 assign_smem tile defaults ---------------------------------------------
// Declared here so the startup capacity check and VAL-010 can see them before M4 implements the
// kernel; shared memory per block is 4 * T_K * T_D bytes (1 KiB at the defaults).
constexpr int kAssignTileK = 8;
constexpr int kAssignTileD = 32;

inline int64_t assign_smem_bytes(int tile_k, int tile_d)
{
    return 4 * static_cast<int64_t>(tile_k) * static_cast<int64_t>(tile_d);
}

// --- MEM-004 shared-memory footprints ----------------------------------------------------------

// `assign` reduces its per-thread change flags through one 8-byte slot per thread and `movement`
// one 8-byte partial maximum per thread, so both are at most 8 * block size bytes.
inline int64_t block_reduction_bytes(int block)
{
    return 8 * static_cast<int64_t>(block);
}

// `reduce_stage2`: 8 * 32 binary64 sums plus 8 uint32 counts = 2080 B, independent of the launch.
inline int64_t stage2_shared_bytes()
{
    return 8 * static_cast<int64_t>(kStage2ChunkLanes) * kStage2DimTile +
           4 * static_cast<int64_t>(kStage2ChunkLanes);
}

// --- RED-006 choice of G -----------------------------------------------------------------------

// MEM-003: B_red = min(user cap if given, 0.25 * (free - other buffers)). `free_bytes` is what
// cudaMemGetInfo reports and `other_bytes` is everything in MEM-001 except P and Q. A
// configuration whose non-reduction buffers already exceed free memory yields a budget of 0,
// which then fails the G_mem >= 1 requirement of RED-007 with a byte count in the message.
inline uint64_t reduction_budget_bytes(uint64_t free_bytes, uint64_t other_bytes,
                                       uint64_t user_cap_bytes)
{
    const uint64_t headroom = free_bytes > other_bytes ? free_bytes - other_bytes : 0;
    uint64_t budget = static_cast<uint64_t>(static_cast<double>(headroom) * 0.25);
    if (user_cap_bytes > 0 && user_cap_bytes < budget) {
        budget = user_cap_bytes;
    }
    return budget;
}

struct ChunkSelection {
    int64_t G = 0;                 // the selected chunk count, 0 when ok == false
    int64_t g_memory = 0;          // G_mem = floor(B_red / (8KD + 4K))
    int64_t g_target = 0;          // max(1, ceil(T_target / D))
    uint64_t budget_bytes = 0;     // B_red
    uint64_t bytes_per_chunk = 0;  // 8KD + 4K
    uint64_t required_bytes = 0;   // bytes for the selected G, or for the G that did not fit
    bool user_override = false;
    bool ok = true;
    std::string message;  // RED-007 explanation when ok == false
};

// G = min(N, G_mem, max(1, ceil(T_target / D))) with T_target = 2048 * SM count (DEC-001).
// `user_G` > 0 is the --reduce-chunks override. It is still checked against N and the budget: an
// override that does not fit must fail loudly rather than silently allocate past the budget.
inline ChunkSelection choose_reduce_chunks(int64_t N, int64_t D, int64_t K, int sm_count,
                                           uint64_t free_bytes, uint64_t other_bytes,
                                           uint64_t user_cap_bytes, int64_t user_G)
{
    ChunkSelection sel;
    sel.budget_bytes = reduction_budget_bytes(free_bytes, other_bytes, user_cap_bytes);

    uint64_t kd = 0, per_chunk = 0, counts = 0;
    const bool size_ok = checked_mul(static_cast<uint64_t>(K), static_cast<uint64_t>(D), &kd) &&
                         checked_mul(kd, 8, &per_chunk) &&
                         checked_mul(static_cast<uint64_t>(K), 4, &counts) &&
                         checked_add(per_chunk, counts, &per_chunk);
    if (!size_ok || per_chunk == 0) {
        sel.ok = false;
        sel.message = "the Approach B partial-sum buffer size 8KD + 4K overflows 64-bit "
                      "arithmetic for K=" + std::to_string(K) + " D=" + std::to_string(D);
        return sel;
    }
    sel.bytes_per_chunk = per_chunk;
    sel.g_memory = static_cast<int64_t>(sel.budget_bytes / per_chunk);

    // T_target = 2048 * SM count: enough workers to fill the device several times over, divided
    // among the D workers that each chunk already provides.
    const int64_t t_target = 2048 * static_cast<int64_t>(sm_count > 0 ? sm_count : 1);
    const int64_t by_target = (t_target + D - 1) / D;
    sel.g_target = by_target > 1 ? by_target : 1;

    int64_t g = sel.g_target;
    if (g > N) {
        g = N;
    }
    if (user_G > 0) {
        sel.user_override = true;
        g = user_G;
        if (g > N) {
            sel.ok = false;
            sel.message = "--reduce-chunks " + std::to_string(user_G) + " exceeds N=" +
                          std::to_string(N) + "; DEC-001 requires 1 <= G <= N";
            return sel;
        }
    } else if (g > sel.g_memory) {
        g = sel.g_memory;
    }

    if (g < 1 || g > sel.g_memory) {
        // RED-007: no usable G. Report what the requested G costs, or what one chunk would cost
        // when even that does not fit, so the message states how much memory the run needs.
        const int64_t wanted = g < 1 ? 1 : g;
        sel.ok = false;
        sel.required_bytes = per_chunk * static_cast<uint64_t>(wanted);
        sel.message = "Approach B needs " + std::to_string(sel.required_bytes) +
                      " B of partial-sum buffers for G=" + std::to_string(wanted) +
                      " (8KD + 4K = " + std::to_string(per_chunk) + " B per chunk), but the "
                      "MEM-003 budget is only " + std::to_string(sel.budget_bytes) + " B";
        return sel;
    }

    sel.G = g;
    sel.required_bytes = per_chunk * static_cast<uint64_t>(g);
    return sel;
}

// --- DEC-001 chunk partition -------------------------------------------------------------------
// chunk_g = [floor(gN/G), floor((g+1)N/G)). The products are formed in int64: g <= G <= N and
// N <= 2^31 - 1, so gN <= 2^62 and cannot overflow. Consecutive bounds share endpoints, the first
// chunk starts at 0 and the last ends at N, so the chunks partition [0, N) exactly (RED-005). The
// device code uses the identical expression.
inline int64_t chunk_begin(int64_t g, int64_t N, int64_t G)
{
    return (g * N) / G;
}

inline int64_t chunk_end(int64_t g, int64_t N, int64_t G)
{
    return ((g + 1) * N) / G;
}

}  // namespace kmeans

#endif  // KMEANS_LAUNCH_CONFIG_H
