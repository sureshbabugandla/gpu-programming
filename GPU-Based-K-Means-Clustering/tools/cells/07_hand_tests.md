## 7. Hand datasets H1–H6 and the CPU-level tests

Runs the test executable, which checks the CPU reference against every hand-verifiable case of
§16.0 — H1–H6 and the five H2 variants — and then TEST-U08, U09, U11, U12 and U13 at host level.
The cell raises if any check fails, so a broken oracle stops "restart kernel and run all" here,
before any GPU result could be compared against it (NFR-015).

What the hand cases pin down, case by case:

| Case | What it fixes |
|---|---|
| H1 | the ordinary path: one update, then assignment stability, `t = 1` |
| H2 | a three-update trajectory with the deltas 2, 1.5, 3.5, ending in `CONVERGED_ASSIGNMENTS` |
| H2-a, H2-b | $\Delta \le \varepsilon$ **inclusive** (DEC-002, ENV-10): the tolerance equals the delta exactly, and the run must stop |
| H2-c | one notch below that delta, so the run continues — the other side of the same boundary |
| H2-d, H2-e | the iteration limit, including $I_{\max} = 0$: `ITERATION_LIMIT`, never reported as convergence (§2.7) |
| H3 | an exact tie resolved to the lowest index (DEC-005) and an empty cluster retained (DEC-003) |
| H4 | duplicate initial centroids (`W-DUPC`): every point ties, so index 0 takes them all |
| H5 | $K > N$ (`W-KGTN`): at least $K - N$ clusters stay empty for the whole run |
| H6 | the only 2-D case, which catches a transposed or stride-confused layout that every 1-D case above would pass |

The P2 flag is checked in every case, including the four where it is correctly **false**: after
`CONVERGED_MOVEMENT` or `ITERATION_LIMIT` the reported centroids are generally not the means of
the reported assignments, and §2.8 requires that to be reported honestly rather than repaired by
extra hidden iterations.

The cell finishes by generating one small Gaussian-blob dataset through the `kmeans-gen` CLI
(FR-003, BLD-003a), which confirms the binary format round-trips in this environment and prints
the measured minimum centre separation in units of the spread — the quantity §10.3 uses to call a
dataset well-separated.
