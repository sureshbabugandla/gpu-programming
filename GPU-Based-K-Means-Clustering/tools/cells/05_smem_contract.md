### Kernel contracts, part 3 — `assign_smem`, the shared-memory assignment kernel

`assign_smem` has **the same contract** as the baseline `assign`: the same assignments, the same
change count, the same in-place write into `A`. The only difference is where the centroid
coordinates are read from. The baseline re-reads `C[k][j]` from global memory for every
(point, centroid) pair and relies on L1 to absorb the reuse; `assign_smem` stages a
`T_K × T_D` block of `C` in shared memory once per block and every thread of the block reads it
from there (SMEM-001).

#### `assign_smem` — §8, SMEM-001…SMEM-005, CUDA-002

| | |
|---|---|
| Thread mapping | One thread per point `i`, 1-D grid-stride (DEC-008), as in the baseline. The trip count of the grid-stride loop is derived from the block's **first** point index, not from the thread's own, so it is uniform across the block. This matters: the loop body contains block-collective tile loads and barriers, and a per-thread trip count would leave part of the block waiting at a `__syncthreads()` the rest has already left. |
| Ownership | Thread `i` remains the only reader and writer of `assignments[i]` (FR-007). The shared tile is written cooperatively by the block and read by every thread of it; the two phases are separated by barriers, so there is no shared-memory hazard for `racecheck` to find (TEST-S02). `x_col` and `centroids` are read-only. |
| Synchronization | Per dimension tile: `__syncthreads()` after the cooperative load, and `__syncthreads()` again after the last thread has consumed the tile and before the next load overwrites it (SMEM-001 step 2). Threads with `i ≥ N` perform the loads and reach **every** barrier; they only skip the arithmetic (SMEM-002, CUDA-005). The change count is then reduced exactly as in the baseline: one shared-memory tree, one 64-bit `atomicAdd` per block. |
| Numerics | Identical to the baseline, not merely equivalent. `part[t]` accumulates terms in ascending `j` from 0 — within a tile by the inner loop, across tiles because `j0` ascends — in binary32, never reordered (NUM-002). The comparison walks `k` in ascending order, skips `k ≥ K`, and replaces the best candidate only on a **strictly** smaller distance, so exactly equal computed distances keep the lowest index (DEC-005, SMEM-001 step 3). |
| Shared memory | `4·T_K·T_D` B for the tile (SMEM-003) plus `8` B per thread for the change-count reduction (MEM-004) — 3 KiB at the defaults with a 256-thread block. Both parts, and their sum, are checked against the device's per-block limit at startup, so an impossible tile configuration is rejected with exit code 2 rather than discovered as a launch failure (VAL-010, EDGE-34). |
| Registers | The `T_K` partial distances live in registers. The loop over `t` is unrolled over the **compile-time** `T_K` so every `part[t]` index is a constant; a dynamic index would push the array into local memory. Any spill that `-Xptxas -v` reports for this kernel must be reported with the benchmark results (CUDA-010). |
| Coalescing | `x_col` is dimension-major, so consecutive threads read consecutive addresses for each `j`, as in the baseline. The win is that the `x` value for one `j` is loaded **once per thread per tile** instead of once per (point, centroid) pair, and the centroid traffic becomes one cooperative tile load per block. `s_tile[t·T_D + jt]` is uniform across a warp and broadcasts, so the strided `t` index costs no bank conflicts. |

#### Why identical assignments, not just close ones (SMEM-004)

Both kernels perform the same sequence of binary32 operations in the same order on the same values,
so the computed distances are **bit-identical** and the comparison that follows them selects the
same `k`. SMEM-004 is therefore tested as exact equality of the whole run — assignments, counts,
centroids, iteration count, status and `Δ` — and not as agreement within NUM-006. The one admitted
explanation for a difference is `ptxas` contracting a multiply-add in one kernel and not the other;
that is why both are built in the same translation unit with the same flags and no
`--use_fast_math` (BLD-002, NUM-002). If a difference ever appears, SMEM-004 requires it to be
explained and bounded by NUM-005, not tolerated.

#### Tile sizes are a build-time choice (SMEM-001)

`T_K` and `T_D` are compile-time constants with defaults 8 and 32. The PERF-003 tile sweep
(`T_K ∈ {4,8,16}`, `T_D ∈ {16,32}`) recompiles with `-DKMEANS_TILE_K` / `-DKMEANS_TILE_D` rather
than passing a runtime option, which is what keeps the partial distances in registers and leaves
the CLI of BLD-003a unchanged.

#### What shared memory is and is not expected to buy (SMEM-005)

The baseline's centroid reads are already uniform across a warp and already hit L1, so shared
memory is **not** assumed to be faster. The expected benefit is fewer global and L1 transactions
for centroid data when `K·D` is large relative to L1, paid for with barriers and `T_K` registers
per thread. Section 10 reports where it helps and where it does not, with measurements — not with
an assumption.
