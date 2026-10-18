### Kernel contracts, part 2 — Approach B, the two-stage ownership reduction (DEC-001, §7)

Approach B accumulates centroid sums **without atomics** by giving every element of the
intermediate buffers exactly one writer. The buffers are `P ∈ ℝ^{G×K×D}` (binary64, element
`(g,k,j)` at `(g·K + k)·D + j`) and `Q ∈ ℕ^{G×K}` (uint32, element `(g,k)` at `g·K + k`), and the
outputs `S` (`K×D` binary64) and `n` (`K` uint32) are the same buffers the `update` kernel consumes
in Approach A (RED-001, FR-018).

#### `reduce_stage1` — RED-002

| | |
|---|---|
| Thread mapping | One logical worker per `(g, j)`, flat index `w = g·D + j`, 1-D grid-stride over `G·D`; one thread may process several workers in sequence. |
| Ownership | Worker `(g,j)` **exclusively owns** the column `P[g][·][j]` — the `K` elements at stride `D` — and, when `j = 0`, the count row `Q[g][·]`. Every element of `P` and `Q` has exactly one writer. `x_row` and `assignments` are read-only. |
| Synchronization | **None at all inside the kernel**: no atomics, no barriers. The only ordering needed is Stage 1 → Stage 2, which is a kernel boundary (DEC-010). |
| Initialization | Step 1 writes zero to every element the worker owns, on **every** call, so no value from an earlier iteration can survive (RED-006, MEM-006; verified by poisoning in TEST-U06 and by `initcheck` in TEST-S03). |
| Numerics | Ascending `i` within the chunk, so for fixed `(N, D, K, G)` the order is fixed (RED-009). |
| Coalescing | `x_row` is row-major, so for a fixed `i` the `D` workers of one chunk read consecutive addresses (CUDA-009). |

The chunk bounds are `chunk_g = [⌊gN/G⌋, ⌊(g+1)N/G⌋)`, formed in 64-bit: `g ≤ G ≤ N ≤ 2³¹−1`, so
`gN ≤ 2⁶²` and cannot overflow.

#### `reduce_stage2` — RED-003

| | |
|---|---|
| Thread mapping | One **32×8** block per tile `τ ∈ [0, K·T)` with `T = ⌈D/32⌉`, grid-stride over tiles. Lane `x ∈ [0,32)` selects `j = (τ mod T)·32 + x`; lane `y ∈ [0,8)` selects the strided subset `g = y, y+8, y+16, …`. |
| Ownership | Each element of `S` is written by exactly one thread — lane `(x, 0)` of the tile owning that `j` — and each `n[k]` by lane `(0,0)` of that cluster's `j`-tile 0. `P` and `Q` are read-only. Each shared slot is written only by its own thread and read by others only after a barrier. |
| Synchronization | Per tile: one barrier **before** the shared slots are overwritten (so the previous tile's tree reads have finished), one **after** they are filled, and one after each of the three tree levels. `τ` derives from `blockIdx` only, so the tile loop and the `j ≥ D` and `τ mod T = 0` guards are **block-uniform** — no barrier is reached by only part of the block (CUDA-005). |
| Numerics | Ascending `g` within each `y`-lane, then the fixed pairwise tree with strides 4, 2, 1. The order depends only on `(G, D, K)`, never on scheduling (RED-009). |

#### RED-004 — the race-freedom argument (reviewed by TEST-S05)

1. **Stage 1.** Each element of `P` and `Q` has exactly one owning worker, which performs all
   writes to it. `A` and `X^row` are read-only throughout. No two threads write the same address,
   so there are no data races and no lost updates — and therefore no need for atomics.
2. **Stage 2.** Each element of `S` and `n` is written by exactly one thread. `P` and `Q` are
   read-only. Shared-memory slots are written by their own thread before a barrier and read by
   others only after it.
3. **Boundaries.** Stage 1 → Stage 2 and Stage 2 → `update` are kernel boundaries on one stream
   (DEC-010); stream order guarantees a kernel observes all global writes of its predecessors. No
   grid-wide synchronization inside a kernel is assumed anywhere.

`racecheck` detects shared-memory hazards only, so the global-memory claim rests on this argument,
supported by TEST-R01 (a hidden race would normally surface as run-to-run variation) and TEST-U05.

#### RED-005 — exactly-once inclusion

The chunks partition `[0, N)`: consecutive bounds share endpoints, `chunk_0` starts at 0 and
`chunk_{G−1}` ends at `N`. Each worker covers every point of its chunk for its own `j`, and Stage 2
sums all `g ∈ [0, G)`. Hence every `(i, j)` contributes exactly once. Checked by `Σ_k n_k = N`
exactly and by `Σ_k S_{kj}` reproducing the column sums of `X` (TEST-U05, TEST-U07).

#### RED-006 — choice of `G`, and RED-008 trade-offs

`G = min(N, G_mem, max(1, ⌈T_target/D⌉))` with `T_target = 2048 × SM count` and
`G_mem = ⌊B_red / (8KD + 4K)⌋`, where `B_red = min(user cap, 0.25 × (free − other buffers))`
(MEM-003). `G` is fixed for the whole run, reported, and overridable by `--reduce-chunks`; an
override is still checked against `N` and the budget. If no usable `G` exists the run fails before
any allocation with exit code 3 and a message stating the required bytes (RED-007).

Larger `G` increases Stage 1 parallelism but also the memory and the Stage 2 combine work; smaller
`G` lengthens the per-worker sequential loop. Stage 1 parallelism is `G·D`, so small `D` depends on
a large `G` to fill the device. Stage 1's read-modify-writes to `P` go through L2 and DRAM, in
binary64 — on a device whose FP64 rate is 1/32 of FP32 (ENV-06) that cost is the central thing to
measure, not to predict (PERF-010).

Whether Approach B beats Approach A is an **empirical** question, answered in M6 with measurements.
