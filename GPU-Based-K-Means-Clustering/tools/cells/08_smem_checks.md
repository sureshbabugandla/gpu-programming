### SMEM-004 — `assign_smem` must reproduce the baseline exactly

The baseline `assign` is already judged against the CPU reference kernel by kernel (TEST-I10
above), so the cheapest and strongest statement about the tiled kernel is that it reproduces the
baseline **bitwise**: same assignments, same counts, same centroid bit patterns, same iteration
count, same status, same `Δ`. Both kernels accumulate the same binary32 terms in the same
ascending-`j` order and compare candidates in the same ascending-`k` order with the same
strict-less-than rule, so anything less than exact equality means the tiled kernel reordered
arithmetic (NUM-002) or that the two were compiled with different FMA settings — the one difference
SMEM-004 admits, and then only if explained and bounded by NUM-005.

The cell compares the two kernels on:

| Datasets | Why these |
|---|---|
| H1–H6 and H2-a…e (§16.0) | `K ≤ 3` and `D ≤ 2`, so both `T_K = 8` and `T_D = 32` overhang and the partial-tile paths of the loader run on data whose answer is known by hand |
| `N, D, K` = (1,1,1), (257,33,7), (1023,32,8), (4096,65,9) | `D` and `K` below, exactly at, and just over the tile sizes; `N` not a multiple of the block size, so the final block is partial (SMEM-002, EDGE-17) |
| both strategies | the assignment kernel is independent of the accumulation strategy (FR-018), and a difference that appeared under only one of them would point at a launch-ordering fault rather than at the kernel |

With `QUICK = False` the cell additionally runs the full TEST-E16 size matrix — every combination
of `N ∈ {1, 31, 33, 255, 257, 1023}`, `D ∈ {1, 3, 31, 33, 65}`, `K ∈ {1, 3, 7, 9, 17}` — through
both kernels, and repeats the TEST-I10 lockstep run against the CPU reference with `assign_smem`
selected, which is TEST-U01 for the tiled kernel. Three larger shapes are added there too
(`D = 128`/`K = 100`, `D = 512`, `K = 1024`) because a tile loader that mishandles an overhang is
most likely to show up where many tiles are visited in both directions.

Expected output: `0 failures`, with the tile geometry and the shared-memory request per block
printed once from `launch_config.h`, so the configuration the kernel was built with is on the
record next to the result.
