### The full test suite (TEST-B02)

One cell, one verdict. Everything §16.1–§16.5 asks for runs here and the cell raises on the first
failure, so "restart kernel and run all" stops at the first thing that is wrong instead of
producing a notebook full of results nobody should trust (NFR-015, TEST-B02).

Four executables and one table:

| What runs | Covers |
|---|---|
| `./test_cpu` | §16.0 H1–H6, §16.3 the two distance paths, TEST-U08/U09/U11/U12/U13, TEST-I08 (the FR-015 summary and the §13.1 row), and the BLD-003a command line — all of it host-only and therefore also run by `g++` before every commit |
| `./test_gpu` (full) | TEST-I01…I07, I10, U01, U03…U07, E16 and SMEM-004: the kernels against the CPU reference, kernel by kernel |
| `./test_edge` (full) | TEST-E04…E12, E15, E18, E19, E26, E29, E30, E34, TEST-U02, TEST-I09, TEST-I11, TEST-R01 and TEST-R02 |
| the CLI table below | the §9.2 cases whose claim *is* the process exit code: EDGE-01/02/03/20/21, the BLD-005 code for each class of failure, and the fault-injection paths EDGE-27/28 |

#### Why the exit-code cases are driven from here

`N = 0`, `D = 0`, a `NaN` coordinate and an unparsable tolerance are rejected *before any device
allocation* (FR-001). What they assert is therefore not an in-process value but the behaviour of
the program as a program: the exit code of BLD-005 and the `error[RULE-ID]:` line of BLD-006. That
is only observable from outside, so the cell writes deliberately malformed `.kmd` files — a header
claiming zero rows, a payload containing a NaN — and checks what `./kmeans` does with them.

`EDGE-27` and `EDGE-28` cannot be reached by any legal input at all: the VAL-009 pre-check exists
precisely to make an allocation failure impossible, and a launch configuration this code computes
is always valid. They are reached through the MEM-008 hooks `KMEANS_FAIL_ALLOC=n` and
`KMEANS_FAIL_LAUNCH=<kernel>`, which raise at the n-th device allocation or at the named launch
and leave the rest of the program — the RAII unwind, the message, the exit code — exactly as it
would be for a real fault. The sanitizer section then runs the same two cases under
`--leak-check full` to confirm that the unwind actually released everything (MEM-007).

#### TEST-I08: the outputs are read back, not assumed

The last block runs the program with `--out-prefix` and `--verify final`, then reads
`_A.kma`, `_C.kmd` and `_summary.json` with `numpy` and `json` and checks they agree with each
other and with the summary: every assignment in `[0, K)`, the counts equal to a `bincount` of the
assignments and summing to `N`, the centroid file of shape `(K, D)`, `t_alg ≤ t_e2e`, and the
`FR-015` fields all present. A `--reps` run with `--csv` is parsed the same way, to confirm the
§13.1 raw file has one row per measured repetition and one field per header column. The timings
printed there are plumbing checks, not benchmark results — the benchmark section is §10.
