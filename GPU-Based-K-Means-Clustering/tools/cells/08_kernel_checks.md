### 8. Kernel checks: lockstep against the CPU reference, and buffer initialization

This section grows into the full TEST-B02 suite in M5. At M2–M3 it carries the two checks that
judge the kernels on realistic data rather than on four hand-computed points:

**TEST-U06 — buffer initialization (MEM-006).** `P`, `Q`, `S` and `n` are filled with `0xFF` bytes
before the run, which is a quiet NaN as binary64 and `0xFFFFFFFF` as uint32. A correct Approach B
overwrites every element of `P` and `Q` in Stage 1 and every element of `S` and `n` in Stage 2, so
nothing of the poison may reach a result. Expected: identical assignments and counts, centroids
**bitwise** identical for Approach B (RED-009) and within NUM-006 for Approach A (ATOM-007, which
fixes no summation order), and every centroid finite.

**TEST-I10 / §10.3 — lockstep.** A 100 000 × 16, K = 8 Gaussian-blob dataset, both strategies, with
every iteration compared against the CPU reference **kernel by kernel**, so a failure names the
kernel instead of only the end result:

| Checked | Against | Criterion |
|---|---|---|
| `n` | `cpu::accumulate` on the device's own `A(t)` | **exact**, and `Σ_k n_k = N` (RED-005, FR-009) |
| `S` | the same | `1e-12 · n_k · S_max` — proportional to the largest value the sum could take |
| `C(t+1)` | `cpu::update` on the same `(S, n)` | NUM-006; empty clusters retained **bit-exactly** (DEC-003) |
| `Δ` | `cpu::movement` on the device's own centroids and counts | isolates `movement`/`max_reduce` from any centroid difference |
| `A(t+1)` | `cpu::assign` on the device's own `C(t+1)` | disagreements allowed **only** on near-ties (NUM-005) |
| `changed` | the reference count of differences `A(t)` vs `A(t+1)` | **exact** (TEST-U03) |

The first snapshot additionally pins down the initial assignment pass of §2.7, which no later
snapshot covers. This validates every kernel on realistic data without requiring device and
reference to follow identical trajectories, which is exactly what §10.3 item 1 is for.

Expected output: `0 failures`, and for each strategy a line reporting the iterations checked, the
near-tie assignment differences (a count, not a failure) and the worst `|ΔS|`, `|ΔC|` and `|ΔΔ|`.
The reduction run also prints its selected `G` with `G_target`, `G_mem` and the bytes of partials
(RED-006, RED-007). Lockstep runs are marked **not timing-valid** — the per-iteration downloads sit
inside the measured region, so FR-019 bars them from being benchmark samples.

With `QUICK = False` the cell also runs the sweeps that take longer: TEST-U05 (`G ∈ {1, 2, 7, N}`
and auto), TEST-U07 (`K = 1024`, `D ∈ {1, 33, 65}`, `G = 7` — partial Stage 2 tiles and chunks with
no points of most clusters, EDGE-35) and TEST-E16 (every combination of `N ∈ {1, 31, 33, 255, 257,
1023}`, `D ∈ {1, 3, 31, 33, 65}`, `K ∈ {1, 3, 7, 9, 17}` against the CPU reference on
well-separated data, §10.3 method 2).
