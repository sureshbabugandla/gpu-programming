## 9. Sanitizers (TEST-S01–S04)

Four tools, four different questions, all four required to report zero (§16.4):

| Test | Tool | What it would catch here |
|---|---|---|
| TEST-S01 | `memcheck --leak-check full` | an out-of-bounds global access — a missing CUDA-005 guard on a partial block — and any device or pinned allocation an error path failed to release (MEM-007) |
| TEST-S02 | `racecheck` | a shared-memory hazard: a tile read before the load barrier or after the overwrite barrier in `assign_smem`, or a missing barrier in the `assign`, `movement` and `reduce_stage2` reductions |
| TEST-S03 | `initcheck` | a read of device memory that nothing wrote in the current phase — the MEM-006 claim that `P`, `Q`, `S`, `n` and `C_next` are fully overwritten, which TEST-U06 also tests by poisoning |
| TEST-S04 | `synccheck` | a barrier that only part of a block reaches, which is the failure mode the block-uniform grid-stride trip count of `assign_smem` exists to prevent (SMEM-002) |

#### Why these configurations

Small ones, deliberately. Sanitizers slow a kernel by one to two orders of magnitude, so the
benchmark sizes are out of the question; what matters is coverage of the *code paths*, and those
are chosen by shape, not by size. Every configuration below has `N`, `D` and `K` that are not
multiples of the block, tile or warp size, so every kernel runs with a partial final block and
every tile load overhangs (EDGE-16, EDGE-17). Both strategies and both assignment kernels are
covered, because `racecheck` and `synccheck` have nothing to say about a kernel that did not run.

`memcheck` additionally runs the two **error paths** of EDGE-27 and EDGE-28 under
`--leak-check full`. That is the only way to check the MEM-007 claim that matters: not that a
successful run frees its memory, but that a run which threw halfway through the allocation
sequence still released what it had already taken.

`racecheck` detects shared-memory hazards only. Approach B's freedom from *global* races rests on
the RED-004 ownership argument — worker `(g, j)` is the only writer of `P[g][·][j]` — and is
supported from two other directions: TEST-R01, where a hidden race would normally show up as
run-to-run variation, and TEST-U05, where the result is checked across the whole range of `G`.
This is stated here rather than left implicit, because a clean `racecheck` run is not by itself
evidence of global race freedom (§16.4).

#### Reading the output

The tools do not end the same way. `memcheck`, `initcheck` and `synccheck` close with
`ERROR SUMMARY: N errors`, and `memcheck` adds `LEAK SUMMARY: N bytes leaked`. `racecheck` prints
neither: it closes with `RACECHECK SUMMARY: N hazards displayed (E errors, W warnings)` and nothing
else. The cell accepts both forms and requires every count it finds to be zero — a racecheck
*warning* is a possible hazard the tool could not prove either way, so it is not something to pass
over. If **neither** summary line appears the case fails regardless: no summary means the tool did
not reach the end, and there is then nothing to conclude.

Expected output: `0 errors` and `0 bytes leaked` from `memcheck`, `0 hazards, 0 errors,
0 warnings` from `racecheck`, `0 errors` from `initcheck` and `synccheck`, and a clean summary at
the end. The cell raises otherwise. It is skipped when `QUICK = True`.
