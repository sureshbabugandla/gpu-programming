### Kernel contracts, part 1 — assignment, atomic accumulation, update, convergence

One note per kernel: **thread mapping**, **memory ownership**, **synchronization points**. These are
the arguments to defend in review; the reduction kernels of Approach B have their own note below.
Common to all of them: every global index is guarded against its range (CUDA-005), out-of-range
threads skip work but still reach every barrier, all element offsets are unsigned 64-bit
(CUDA-006), and no kernel reads what another block of the same launch wrote — phase boundaries are
kernel boundaries on one stream (CUDA-004, DEC-010).

#### `assign` — §2.3, FR-004…FR-007, DEC-008

| | |
|---|---|
| Thread mapping | One thread per point `i`, 1-D grid-stride. Each thread loops over all `k` and all `j` itself. |
| Ownership | Thread `i` is the **only** reader and writer of `assignments[i]`, which is what lets FR-007 update it in place: read the old value, compute, compare, write. `x_col` and `centroids` are read-only. |
| Synchronization | None for the assignment. The change count is reduced through shared memory, so every thread of the block reaches every `__syncthreads()`, including threads whose grid-stride loop body never ran. One 64-bit `atomicAdd` per **block** into the change counter, not one per point. |
| Numerics | NUM-002: terms in binary32, accumulated in binary32 in ascending `j` from 0, never reordered. DEC-005: `k` ascends from 0 and the candidate is replaced only on a **strictly** smaller distance, so exactly equal computed distances keep the lowest index. |
| Coalescing | `x_col` is dimension-major, so for a fixed `j` consecutive threads read consecutive addresses; the centroid read is uniform across the warp and broadcasts (CUDA-009). |

The change counter is an integer, and integer atomics are exact, so it is deterministic and does
not compromise Approach B's atomic-free accumulation of centroid data.

#### `accum_atomic` — §6, ATOM-001…ATOM-007

| | |
|---|---|
| Thread mapping | One thread per element `e` of `X`, 1-D grid-stride over `N·D`, with `i = e mod N` and `j = e / N` (ATOM-002). Because `x_col` holds element `(i,j)` at `j·N + i`, that index *is* `e`, so the read is a sequential walk and coalesces (CUDA-009). |
| Ownership | `sums` and `counts` are shared by every thread; every write is an atomic read-modify-write, so no update is lost. The caller zeroes both buffers on the same stream first (ATOM-001), which is safe on reuse because the memset is stream-ordered after the previous consumer, `update`. |
| Synchronization | None inside the kernel. Ordering against `update` is the kernel boundary of ATOM-005 / DEC-010. |
| Numerics | Binary64 sums (DEC-007) in a hardware-chosen order, so `S` may differ in its last bits run to run (ATOM-007). Counts are integers and are always exact. **No bitwise reproducibility is claimed for Approach A.** |
| Contention | All threads for points of one cluster target the same `D` addresses, so contention grows as `K` falls — which is why the benchmark matrix includes small `K` (ATOM-010, PERF-003). |

#### `update` — §2.5, FR-010, FR-011, DEC-003

| | |
|---|---|
| Thread mapping | One thread per centroid element, 1-D grid-stride over `K·D`. |
| Ownership | Element `(k,j)` of `centroids_next` is written by exactly one thread; `sums`, `counts` and `centroids_cur` are read-only. Double buffering (FR-010) is what makes that true — writing into `centroids_cur` would race with the reads of the retention branch and with `movement`. |
| Synchronization | None inside the kernel; the boundary to `movement` is a kernel boundary. |
| Numerics | The division is performed in binary64 and the quotient rounded **once** to binary32. When the count is zero no division happens at all and the previous centroid is copied bit-exactly (DEC-003). Every `(k,j)` is written on every call, so `centroids_next` needs no initialization (MEM-005). |

#### `movement` and `max_reduce` — §2.6, FR-012

| | |
|---|---|
| Thread mapping | `movement`: one thread per cluster, grid-stride over `K`, each thread computing the whole squared displacement of its cluster in ascending `j`. `max_reduce`: a single block, grid-stride over the per-block maxima. |
| Ownership | `block_max[blockIdx.x]` is written by thread 0 of that block and no one else; `*delta` by thread 0 of the single `max_reduce` block. The centroid buffers, counts and `block_max` are read-only to their readers. |
| Synchronization | Both reduce through shared memory with a barrier after each tree level, and all threads reach all barriers. One block in `max_reduce`, so no grid-wide synchronization is assumed (CUDA-004). |
| Numerics | Differences are formed in binary64 from the binary32 centroids and summed in ascending `j`. Empty clusters have `δ_k = 0` by construction and are excluded from the maximum, so no `0/0` can arise. `movement` keeps `δ_k²`; the single `sqrt` is applied once in `max_reduce`. Because `sqrt` is monotone this picks the same cluster as §2.6 and rounds the same binary64 sum once, so the value is identical to taking the root per cluster and then the maximum. |

The movement grid is capped at **1024** blocks — the "unless overridden" clause of CUDA-007. That
cap lets `max_reduce` use one thread per per-block maximum in a single block, and it equals the
`movement_blocks` default of `device_memory_bytes`, so the MEM-002 capacity model and the real
allocation agree.

#### Non-power-of-two block sizes

CUDA-007 validates the block size as a multiple of 32, which need not be a power of two (96, 160
and 192 are all legal). The shared-memory reductions therefore start from the next power of two at
or above `blockDim.x` and guard the partner index, which is correct for any block size.

#### Per-iteration order (CUDA-003)

accumulation → `update` → `movement` → `max_reduce` → memset of the change counter → `assign` →
one 16-byte async copy of `Δ` and `changed` into pinned host memory → stream synchronization → host
decision (DEC-002) → centroid pointer swap. The swap is bookkeeping, so the `assign` of iteration
`t` reads the buffer `update` just wrote; the swap happens on every path, including the one that
breaks out of the loop, so the reported pair is always `(A(t), C(t))` and P1 holds by construction.
