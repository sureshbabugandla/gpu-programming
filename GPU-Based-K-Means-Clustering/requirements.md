# requirements.md — GPU-Based K-Means Clustering Using CUDA (Problem 5)

| Field | Value |
|---|---|
| Document type | Software Requirements Specification (Specification-Driven Development) |
| Project | Problem 5: GPU-Based K-Means Clustering |
| Stage | Stage 1 — Specification (no implementation code) |
| Version | 1.7 |
| Status | Specification complete; Stage 2 may begin once machine-dependent items in §1.6.2 are recorded |
| Normative keywords | **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, **MAY** as in RFC 2119 |

### 0.1 Conventions

- **Source tags.** Every requirement is tagged with its origin:
  - `[ASG]` — mandated by the assignment statement.
  - `[DEC]` — engineering decision made in this document (justified in §3).
  - `[ASM]` — assumption that must be confirmed (listed in §1.6).
- **Identifier prefixes.** `FR` functional, `CUDA` execution model, `ATOM` atomic approach, `RED` reduction approach, `SMEM` shared memory, `VAL` input validation, `NUM` numerical, `NFR` non-functional, `MEM` memory/resources, `PERF` benchmarking, `BLD` build/execution, `DEC` design decision, `OPT` optional optimization, `TEST-*` tests, `EDGE-*` corner cases, `ARC` architecture components.
- Identifiers are stable. A removed requirement is marked *Withdrawn*, never renumbered.
- Indices are zero-based throughout: points $i \in [0,N)$, dimensions $j \in [0,D)$, clusters $k \in [0,K)$.
- $u_{32} = 2^{-24} \approx 5.96\times10^{-8}$ denotes the unit roundoff of IEEE-754 binary32.

---

## 1. Project Overview and Scope

### 1.1 Objective

Implement Lloyd's K-Means algorithm on an NVIDIA GPU using CUDA. Given $N$ points in $\mathbb{R}^D$ and $K$ initial centroids, the program alternates between (a) assigning every point to its nearest centroid by squared Euclidean distance and (b) recomputing every centroid as the mean of its assigned points, until a documented convergence criterion holds or an iteration limit is reached. The program reports final assignments, final centroids, the iteration count, and execution time.

### 1.2 Why the problem suits a GPU

- **Assignment** costs $\Theta(NKD)$ arithmetic with no dependence between points. Each point's nearest-centroid search is independent, so the step maps directly onto one thread (or thread group) per point. All threads read the same $K\times D$ centroid matrix, which makes centroid reuse through shared memory a measurable optimization.
- **Centroid recomputation** is a *keyed reduction*: $\Theta(ND)$ additions into $K\times D$ accumulators, where the key (cluster) of each contribution is data-dependent. Many threads contribute to the same accumulator, so the update requires either hardware atomics or a race-free reduction design. This contention structure is the central engineering question of the assignment.

### 1.3 Scope

**In scope**

1. A CUDA implementation of the full iteration (assignment, accumulation, update, convergence evaluation, finalization).
2. Two centroid accumulation strategies with separate implementation contracts: **Approach A (atomic)** and **Approach B (reduction)**.
3. A shared-memory variant of the assignment kernel, evaluated against a non-shared-memory baseline.
4. A single-threaded CPU reference implementation used as the correctness oracle.
5. Input validation, error handling, deterministic synthetic dataset generation, unit/integration/edge-case tests, CUDA sanitizer checks, and a benchmark harness with CSV output.

**Out of scope**

- Centroid initialization algorithms (k-means++, random restarts). Initial centroids are an assignment input. The dataset generator produces them deterministically for benchmarks only (FR-003).
- Multi-GPU execution, distributed execution, out-of-core datasets larger than device memory.
- Alternative K-Means variants (mini-batch, Elkan/Hamerly triangle-inequality pruning, spherical K-Means).
- Python bindings, GUIs, or frameworks unrelated to the assignment.

### 1.4 Mandatory work versus optional optimizations

| Category | Items |
|---|---|
| Mandatory `[ASG]` | CUDA K-Means; parallel distance computation; atomic-based accumulation; reduction-based accumulation; safe concurrent updates; shared-memory use where beneficial; comparison of time and numerical results across N, D, K; required inputs and outputs. |
| Mandatory `[DEC]` | CPU reference; input validation; documented convergence, empty-cluster, tie, iteration-count and finalization policies; tests; sanitizer runs; benchmark methodology. |
| Optional (`OPT-*`) | Shared-memory privatized atomics (OPT-001); FP32 accumulation mode (OPT-002); whole-centroid-set shared-memory assignment (OPT-003); tree-reduction alternatives for Stage 1 of Approach B (OPT-004); warp-cooperative assignment for large D (OPT-005); cross-GPU comparison on a second Colab GPU (OPT-006, PERF-012). Optional items MUST NOT be implemented before the mandatory baseline passes all correctness tests. |

### 1.5 Deliverables

1. **One Jupyter notebook, `kmeans_cuda.ipynb`, as the final code file** (DEC-011). It contains all CUDA and host source (written to files by notebook cells), the compile commands, the tests, the benchmarks, the plots and the written evaluation. The executed notebook with saved outputs is the submission evidence.
2. CPU reference implementation and test suite.
3. Test evidence: logs of unit, integration, edge-case and sanitizer runs.
4. Benchmark raw and summary CSV files, environment record, and plots.
5. A short evaluation report comparing Approach A and Approach B (time and numerical results).
6. This `requirements.md`, updated with any approved changes (§18.4).

### 1.6 Assumptions and unresolved environment items

The supervisor has confirmed that the problem statement is the complete brief: no datasets, sizes, language, build system or output format are prescribed. Items that depended on further guidance are therefore closed as engineering decisions (§1.6.1). Items that depend on the physical machine remain open and are resolved by recording the student's own environment (§1.6.2). No hardware, compiler version or benchmark number in this document is presumed.

#### 1.6.1 Closed items (decided in this specification)

| ID | Item | Decision `[DEC]` | Rationale |
|---|---|---|---|
| ENV-04 | Build system and deliverable form | Jupyter notebook `kmeans_cuda.ipynb`; source written by `%%writefile` cells and compiled with `nvcc` from notebook cells (DEC-011). No CMake. | Requested by the student: the final code file must be `.ipynb` |
| ENV-11 | Notebook runtime | **Google Colab with a GPU runtime** (Kaggle Notebooks as fallback if Colab GPU quota is unavailable) | The student's laptop (Intel Core i7-1355U, Intel Iris Xe integrated graphics only) has no NVIDIA GPU, so CUDA cannot run locally; the laptop is used only as a browser client |
| ENV-05 | Language | CUDA C++ with C++17 host code | Native CUDA language; no other language requested |
| ENV-07 | Datasets and sizes | Synthetic Gaussian blobs (FR-003), hand datasets H1–H6 (§16.0), benchmark matrix PERF-003 scaled to device memory | Problem statement only requires "different values of N, D and K"; synthetic data gives control over all three and a known ground truth |
| ENV-09 | Output format | Binary `.kmd`/`.kma` files plus JSON run summary (BLD-004) and CSV benchmark files (§13.1) | Problem statement lists output contents, not format |
| ENV-10 | Interpretation of "movement becomes smaller than a specified tolerance" | $\Delta \le \varepsilon$ (inclusive), DEC-002 | With strict $<$, $\varepsilon = 0$ could never trigger the movement criterion; inclusive comparison makes $\varepsilon=0$ mean "no movement". Strict and inclusive differ only when $\Delta$ equals $\varepsilon$ exactly. |

#### 1.6.2 Open items (record from the student's machine before Stage 2)

These cannot be decided by specification; they are facts about the hardware used. Each is resolved by running the listed command and recording the output in the notebook environment cell and in every benchmark CSV row (PERF-008).

| ID | Item | How to record | Default assumed | Impact if different |
|---|---|---|---|---|
| ENV-01 | GPU model and compute capability | `!nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv` | CC ≥ 6.0 `[ASM]` | Below 6.0, native `atomicAdd(double)` is unavailable; ATOM-003 would need a CAS-loop fallback (spec change per §18.4). |
| ENV-02 | CUDA toolkit and driver version | `!nvcc --version`; `!nvidia-smi` | Toolkit supports the GPU and ships `compute-sanitizer` `[ASM]` | `-arch` flags; sanitizer availability. |
| ENV-03 | Host OS and host compiler of the Colab VM | `!cat /etc/os-release`, `!g++ --version` | Linux + g++ (Colab) `[ASM]` | Compiler flags in BLD-002. |
| ENV-06 | FP64 throughput of the GPU | Look up the GPU's FP64:FP32 ratio from the vendor specification for the recorded model | Unknown | Low FP64 rate (typical on consumer GPUs) affects interpretation of both strategies and motivates OPT-002. |
| ENV-08 | Profiler availability | `!nsys --version`, `!ncu --version` | Available `[ASM]` | If absent, PERF-009 and TEST-P03 fall back to CUDA-event phase timing (PERF-007); recorded as a limitation. |

---

## 2. Problem Definition and Mathematical Specification

### 2.1 Notation, types and layouts

| Symbol | Meaning | Type / representation | Valid range |
|---|---|---|---|
| $N$ | number of points | 64-bit integer on input, stored as `int64` | $1 \le N \le 2^{31}-1$ |
| $D$ | number of dimensions | as above | $1 \le D \le 2^{31}-1$, subject to MEM-002 |
| $K$ | number of clusters | as above | $1 \le K \le 2^{31}-1$, subject to MEM-002 |
| $X \in \mathbb{R}^{N\times D}$ | data matrix, $x_{ij}$ | IEEE-754 binary32 | finite, range-limited (VAL-006) |
| $C^{(t)} \in \mathbb{R}^{K\times D}$ | centroids after $t$ updates, $c^{(t)}_{kj}$ | binary32 | finite |
| $A^{(t)} \in \{0,\dots,K-1\}^N$ | assignments computed from $C^{(t)}$ | `int32` | $[0,K)$ |
| $n^{(t)}_k$ | number of points with $A^{(t)}_i = k$ | `uint32` | $[0,N]$ |
| $S^{(t)}_{kj}$ | coordinate sum of cluster $k$, dimension $j$ | binary64 (default, DEC-007) | finite |
| $I_{\max}$ | maximum number of centroid updates | 64-bit integer | $0 \le I_{\max} \le 2^{31}-1$ |
| $\varepsilon$ | movement tolerance | binary64 | finite, $\varepsilon \ge 0$ |

**Layouts.**

- Host and file layout of $X$ and $C$: row-major, element $(i,j)$ at offset $i\cdot D + j$.
- Device copies of $X$ (DEC-009): a dimension-major copy $X^{\mathsf{col}}$ with element $(i,j)$ at $j\cdot N + i$ (coalesced reads when consecutive threads handle consecutive points), and a row-major copy $X^{\mathsf{row}}$ (coalesced reads when consecutive threads handle consecutive dimensions of one point).
- Device centroids: row-major $K\times D$.
- All element offsets are computed in unsigned 64-bit arithmetic (CUDA-006).

### 2.2 Squared Euclidean distance

$$
d^2(x_i, c_k) = \sum_{j=0}^{D-1} \left(x_{ij} - c_{kj}\right)^2
$$

**Evaluation semantics (NUM-002).** On the device and in the FP32 path of the CPU reference, each term is computed in binary32 and accumulated in binary32 in ascending order of $j$, starting from $0$. The implementation MUST NOT reorder the summation (e.g. by splitting over $j$ across threads) in the baseline assignment kernel. The device compiler MAY contract a multiply-add into an FMA; this is the only permitted source of difference from the CPU FP32 path, and it is handled by the near-tie rule (NUM-005).

**Why the square root is omitted.** For non-negative reals, $\sqrt{\cdot}$ is strictly increasing, so $\arg\min_k d(x_i,c_k) = \arg\min_k d^2(x_i,c_k)$. Omitting it saves one transcendental per pair and avoids an extra rounding step. The mean update (§2.5) minimizes the sum of squared distances, so $d^2$ is also the natural quantity for the objective (§2.9).

### 2.3 Assignment rule and tie-breaking

$$
A^{(t)}_i = \min\left\{ k \in [0,K) \;:\; d^2(x_i, c^{(t)}_k) = \min_{m} d^2(x_i, c^{(t)}_m) \right\}
$$

That is, the nearest centroid, and among exactly equal computed distances, the **lowest centroid index** (DEC-005). Operationally: iterate $k$ in ascending order, initialize the best candidate with $k=0$, and replace it only when the new distance is **strictly less** than the current best. "Equal" refers to equality of the computed binary32 values under §2.2 semantics.

### 2.4 Cluster counts and sums

$$
n^{(t)}_k = \sum_{i=0}^{N-1} \mathbf{1}\!\left[A^{(t)}_i = k\right],\qquad
S^{(t)}_{kj} = \sum_{i : A^{(t)}_i = k} x_{ij}
$$

$S$ is accumulated in binary64 by default (DEC-007). The summation order is implementation-defined in Approach A and fixed in Approach B (§6, §7).

### 2.5 Centroid update and empty clusters

$$
c^{(t+1)}_{kj} =
\begin{cases}
\mathrm{fl}_{32}\!\left( S^{(t)}_{kj} / n^{(t)}_k \right) & n^{(t)}_k > 0 \\[4pt]
c^{(t)}_{kj} & n^{(t)}_k = 0 \quad \text{(retain previous centroid, DEC-003)}
\end{cases}
$$

The division is performed in binary64 and the quotient is rounded once to binary32 ($\mathrm{fl}_{32}$).

### 2.6 Centroid movement

$$
\delta_k^{(t+1)} = \left\| c^{(t+1)}_k - c^{(t)}_k \right\|_2 ,\qquad
\Delta^{(t+1)} = \max_{k \in \mathcal{N}^{(t)}} \delta_k^{(t+1)},\qquad
\mathcal{N}^{(t)} = \{k : n^{(t)}_k > 0\}
$$

$\delta_k$ is computed in binary64 from the binary32 centroid values: differences are formed in binary64, squared, summed in ascending $j$, and square-rooted. Empty clusters have $\delta_k = 0$ by construction and are excluded from $\mathcal{N}$; because $N \ge 1$, $\mathcal{N}$ is never empty.

### 2.7 Algorithm, convergence, iteration counting and termination

Let $\text{ASSIGN}$ apply §2.3 to all points, $\text{ACCUM}$ compute §2.4, and $\text{UPDATE}$ apply §2.5.

```
Input : X, C(0), I_max, ε
A(0) ← ASSIGN(X, C(0))                      // initial assignment pass
t ← 0
status ← ITERATION_LIMIT
while t < I_max:
    (S, n) ← ACCUM(X, A(t))
    C(t+1) ← UPDATE(S, n, C(t))
    Δ ← max over k with n_k > 0 of ‖C(t+1)_k − C(t)_k‖₂
    t ← t + 1
    A(t) ← ASSIGN(X, C(t))
    changed ← |{ i : A(t)_i ≠ A(t−1)_i }|
    if changed = 0:  status ← CONVERGED_ASSIGNMENTS; break
    if Δ ≤ ε:        status ← CONVERGED_MOVEMENT;    break
FINALIZE(A(t), C(t), t, status)              // §2.8
```

This is pseudocode for the mathematical algorithm, not an implementation.

**Convergence (DEC-002).** The two criteria in the assignment are treated as **alternative stopping criteria combined by logical OR**:

1. *Assignment stability*: $A^{(t)} = A^{(t-1)}$ exactly (integer comparison, no tolerance).
2. *Movement*: $\Delta^{(t)} \le \varepsilon$ (inclusive comparison in binary64).

Assignment stability is checked first; when both hold, the reported status is `CONVERGED_ASSIGNMENTS`.

**Iteration count (DEC-006).** The reported iteration count is the number of completed centroid updates, $t$. The number of assignment passes in the loop is $t+1$ (one initial pass plus one per update). The finalization accumulation pass (§2.8) is not counted.

**Termination statuses.**

| Status | Condition | Is it convergence? |
|---|---|---|
| `CONVERGED_ASSIGNMENTS` | criterion 1 held after update $t$ | yes |
| `CONVERGED_MOVEMENT` | criterion 2 held, criterion 1 did not | yes |
| `ITERATION_LIMIT` | $t = I_{\max}$ and neither criterion held (includes $I_{\max}=0$) | no |
| `ERROR` | validation, resource, CUDA or I/O failure; no clustering output is produced | no |

**Boundary behavior.**

- $I_{\max} = 0$: the initial assignment pass runs, no updates occur, $t = 0$, status `ITERATION_LIMIT`, $C^{\text{final}} = C^{(0)}$, $A^{\text{final}} = A^{(0)}$.
- Convergence on the first iteration: if $A^{(1)} = A^{(0)}$ (e.g. $C^{(0)}$ already equals the cluster means of $A^{(0)}$), status `CONVERGED_ASSIGNMENTS` with $t = 1$.
- $\varepsilon = 0$: criterion 2 holds only if every non-empty centroid's update is bit-identical to its previous value.
- Convergence is checked before the limit: if update $t = I_{\max}$ also satisfies a criterion, the status is the convergence status.
- Rounding: $\Delta$ is computed from rounded binary32 centroids, so it reflects representable movement; a true movement below the binary32 spacing of the coordinates may appear as $0$.

### 2.8 Finalization and output consistency (DEC-004)

Four properties are kept distinct:

| Property | Definition | Guarantee |
|---|---|---|
| **P1 Assignment consistency** | every $A^{\text{final}}_i$ satisfies §2.3 against $C^{\text{final}}$ | **Always guaranteed by construction**: the loop exits only immediately after $A^{(t)} = \text{ASSIGN}(X, C^{(t)})$, and the reported pair is $(A^{(t)}, C^{(t)})$. |
| **P2 Centroid consistency** | for every $k$ with $n_k^{\text{final}} > 0$, $C^{\text{final}}_k$ equals the mean of points with $A^{\text{final}}_i = k$ within tolerance NUM-006 | **Verified and reported**, not assumed. Holds by construction (up to rounding) when status is `CONVERGED_ASSIGNMENTS`; may be false for `CONVERGED_MOVEMENT` or `ITERATION_LIMIT`. |
| **P3 Convergence** | a criterion in §2.7 held | Reported by status. |
| **P4 Termination** | the algorithm stopped | Always, for non-error runs. |

`FINALIZE` performs exactly these steps:

1. Run one accumulation pass $\text{ACCUM}(X, A^{\text{final}})$ with the selected strategy, giving $n^{\text{final}}$ and $M_{kj} = \mathrm{fl}_{32}(S_{kj}/n_k)$ for non-empty $k$.
2. Set `centroid_consistent = true` iff every non-empty $k$ and every $j$ satisfy the NUM-006 tolerance between $C^{\text{final}}_{kj}$ and $M_{kj}$.
3. Report $A^{\text{final}}$, $C^{\text{final}}$ (unchanged by finalization), $n^{\text{final}}$, $t$, status, `centroid_consistent`, and `assignment_consistent = true` (by construction; independently checked in verification mode, TEST-I07).

**Explicit non-guarantees.**

- The program does **not** run extra hidden iterations to enforce P2; that would change the meaning of $I_{\max}$.
- The program does **not** replace $C^{\text{final}}$ with $M$; doing so would break P1.
- Termination is never reported as convergence.
- When $I_{\max}$ is exhausted, both P1 and P2 cannot always be achieved; P1 has priority and P2 is reported honestly.

Behavior in specific situations:

| Situation | Behavior |
|---|---|
| Convergence detected after an update | Exit with `CONVERGED_*`; P1 holds; P2 verified. |
| Limit reached without convergence | `ITERATION_LIMIT`; P1 holds; P2 verified and typically false. |
| Final assignment pass changes assignments | This is the normal case for `CONVERGED_MOVEMENT`/`ITERATION_LIMIT`; reported centroids are those the final assignments were computed from. |
| Recomputing centroids would change assignments again | Not performed; captured by `centroid_consistent = false`. |
| Empty cluster with retained centroid | Retained centroid is part of $C^{\text{final}}$; P1 is evaluated against all $K$ centroids; P2 is not evaluated for that $k$; $n_k^{\text{final}}$ reported (may be 0). |
| FP differences near decision boundaries | Handled by NUM-005 in verification; does not change runtime semantics. |

### 2.9 Objective (reported, not used for control)

$$
J = \sum_{i=0}^{N-1} d^2\!\left(x_i, c^{\text{final}}_{A^{\text{final}}_i}\right)
$$

$J$ is computed on the host in binary64 after the run, outside timed regions (FR-016). It is used for comparing results across strategies, not for convergence.

### 2.10 Phase separation

| Phase | Responsibility | Executed by |
|---|---|---|
| Input validation | Reject invalid parameters and data (§9) | host |
| Initial centroid loading | Load $C^{(0)}$ from input; no selection algorithm in the core program | host |
| Point assignment | §2.3, plus change count | device kernel |
| Centroid accumulation | §2.4 via Approach A or B | device kernel(s) |
| Centroid update | §2.5 | device kernel |
| Convergence evaluation | $\Delta$ and `changed`; decision on host | device kernels + host |
| Termination | §2.7 statuses | host |
| Final output validation | §2.8 | device accumulation + comparison |

---

## 3. Critical Design Decisions (finalized before coding)

### DEC-001 — Reduction-based accumulation strategy

**Selected algorithm: two-stage, ownership-based, deterministic reduction ("chunk-privatized partial sums + fixed-order tree combine").**

*Stage 1 (privatized partial accumulation).* The point index range is partitioned into $G$ contiguous, non-overlapping chunks:

$$
\text{chunk}_g = \left[\left\lfloor \tfrac{gN}{G}\right\rfloor,\ \left\lfloor \tfrac{(g+1)N}{G}\right\rfloor\right),\quad g \in [0,G),\quad 1 \le G \le N .
$$

A logical worker is the pair $(g, j)$, with flat index $w = g\cdot D + j$. Worker $(g,j)$ **exclusively owns** the column $P[g][\cdot][j]$ of a partial-sum buffer $P \in \mathbb{R}^{G\times K\times D}$ (binary64). Worker $(g,0)$ additionally owns the count row $Q[g][\cdot]$ of $Q \in \mathbb{N}^{G\times K}$ (`uint32`). Each worker:

1. writes $0$ to every element it owns ($K$ elements of $P$; plus $K$ of $Q$ for $j=0$);
2. iterates the points of chunk $g$ in ascending $i$, reads $k = A_i$ and $x_{ij}$ from $X^{\mathsf{row}}$, and performs $P[g][k][j] \mathrel{+}= x_{ij}$ (and, for $j=0$, $Q[g][k] \mathrel{+}= 1$).

Because every element of $P$ and $Q$ has exactly one writer, Stage 1 has no write-write races, needs no atomics and no barriers. Total work is $\Theta(ND + GKD)$.

*Stage 2 (fixed-order tree combine).* For each output tile of 32 consecutive dimensions of one cluster, a block of $32\times8$ threads (lane $x \in [0,32)$ selects $j$, lane $y \in [0,8)$ selects a strided subset of $g$) computes

$$
S_{kj} = \sum_{g=0}^{G-1} P[g][k][j],\qquad n_k = \sum_{g=0}^{G-1} Q[g][k]
$$

as follows: each thread sums $g = y, y+8, y+16, \dots$ sequentially in ascending order; the 8 partial values per $j$ are stored in shared memory and combined by a fixed pairwise tree ($y \leftarrow y + y{+}4$, then $+2$, then $+1$) with `__syncthreads()` between levels. Counts are reduced the same way by the tiles with $j$-tile index 0. Tiles are enumerated by a 1-D grid-stride index $\tau \in [0, K\cdot\lceil D/32\rceil)$.

*Choice of $G$ (RED-006).* $G = \min\left(N,\ G_{\text{mem}},\ \max\left(1, \lceil T_{\text{target}}/D \rceil\right)\right)$, where $T_{\text{target}} = 2048 \times (\text{number of SMs})$ and $G_{\text{mem}} = \lfloor B_{\text{red}} / (8KD + 4K) \rfloor$ with $B_{\text{red}}$ the partial-buffer memory budget (MEM-003). $G$ MAY be overridden by `--reduce-chunks`. $G$ is fixed for the whole run and reported.

| Aspect | Record |
|---|---|
| Intermediate structures | $P$: $G\times K\times D$ binary64; $Q$: $G\times K$ uint32 |
| Thread/block responsibilities | Stage 1: one logical worker per $(g,j)$; Stage 2: one $32\times8$ block per (cluster, 32-dimension tile) |
| Synchronization boundaries | Assignment kernel → Stage 1 (kernel boundary); Stage 1 → Stage 2 (kernel boundary); intra-block `__syncthreads()` in Stage 2 only |
| Memory complexity | $8GKD + 4GK$ bytes extra |
| Empty clusters | Produce $n_k = 0$ and $S_{k\cdot} = 0$; handled by UPDATE (DEC-003) |
| Numerical character | Fixed summation order for fixed $(N,D,K,G)$, binary and device → run-to-run bitwise reproducible (to be verified, TEST-R01). Different $G$ changes the order. |
| Expected trade-offs | No contention and determinism; costs extra memory and $GKD$ zeroing/combining work; Stage 1 parallelism is $G\cdot D$, so small $D$ relies on large $G$. Whether it beats atomics is an empirical question (PERF-010). |
| Alternatives considered | (a) Sort points by cluster then segmented reduction: deterministic and memory-light, but adds a full sort per iteration and a dependency on a sort library. (b) One block per cluster scanning all $N$ assignments with masking: $\Theta(NKD)$ work, wasteful for large $K$. (c) Per-block shared-memory histograms with shared atomics: fast but still atomic-based and nondeterministic; kept as OPT-001 for Approach A. (d) Per-thread private accumulators: infeasible because $k$ is data-dependent and registers cannot be indexed dynamically. |
| Verifying tests | TEST-U05, TEST-U06, TEST-U07, TEST-I02, TEST-R01, TEST-S01–S04, EDGE-35 |

### DEC-002 — Convergence semantics

Assignment stability **OR** movement $\Delta \le \varepsilon$, with Euclidean (not squared) displacement compared inclusively to $\varepsilon$, assignment stability taking reporting precedence (§2.7). Rationale: the assignment names both criteria joined by "or"; assignment stability is exact and immune to tolerance tuning, while the movement criterion lets the user stop early on slowly drifting centroids. Comparing $\Delta$ rather than $\Delta^2$ keeps $\varepsilon$ in coordinate units and avoids overflow of $\varepsilon^2$. The inclusive comparison is the documented reading of "smaller than a specified tolerance" (ENV-10); no further guidance was given by the supervisor. Verified by TEST-U08, TEST-I03–I05, EDGE-22–25.

### DEC-003 — Empty-cluster policy

**Retain the previous centroid.** No division by zero; no data-dependent reinitialization; deterministic. Consequences: the cluster contributes $\delta_k = 0$ and is excluded from $\Delta$; the retained centroid can capture points in later iterations; with $K > N$ at least $K-N$ clusters are permanently empty; reproducibility is unaffected. Alternative (reinitialize to the farthest point) was rejected: it requires an extra global argmax pass, introduces ties of its own, and changes the objective trajectory in ways the assignment does not request. Verified by TEST-U09, EDGE-04, EDGE-14.

### DEC-004 — Final assignment–centroid consistency

As in §2.8: report $(A^{(t)}, C^{(t)})$ where $A^{(t)} = \text{ASSIGN}(C^{(t)})$; P1 guaranteed; P2 verified by one finalization accumulation pass and reported as a flag. Verified by TEST-I06, TEST-I07, EDGE-25.

### DEC-005 — Tie-breaking

Lowest centroid index among exactly equal computed binary32 distances, implemented by strict-less-than replacement in ascending $k$. The shared-memory assignment variant MUST preserve the ascending-$k$ comparison order. Verified by TEST-U02, EDGE-13.

### DEC-006 — Iteration counting

Iteration count = completed centroid updates. Verified by TEST-I03, TEST-I05, EDGE-22, EDGE-23.

### DEC-007 — Numeric types

Points and centroids binary32 (memory bandwidth, matches typical K-Means workloads); distances binary32; centroid sums binary64 in both strategies by default (reduces cancellation and order sensitivity so the two strategies are comparable); counts `uint32`; movement in binary64; tolerance binary64. Native `atomicAdd` on `double` requires compute capability ≥ 6.0 (ENV-01). OPT-002 MAY add binary32 accumulation as a measured variant.

### DEC-008 — Thread mapping for assignment

**Baseline: one thread per point**, grid-stride loop, loops over all $k$ and $j$. Rationale: preserves the sequential $j$ summation order (NUM-002) and the ascending-$k$ tie rule with no inter-thread communication, and gives $N$-way parallelism, which is ample for the target sizes. Alternatives: (a) one warp per point with shuffle reduction over $j$ — better for very large $D$, but changes summation order (OPT-005); (b) one thread per (point, centroid) pair with a segmented argmin — better when $N$ is small and $K$ large, but needs a second reduction and complicates ties; (c) shared-memory centroid tiling — adopted as SMEM-001.

### DEC-009 — Device data layout

Keep two read-only device copies of $X$: $X^{\mathsf{col}}$ (dimension-major) for the assignment kernel and Approach A, and $X^{\mathsf{row}}$ (row-major) for Approach B Stage 1. Cost: $8ND$ bytes total instead of $4ND$. Rationale: both access patterns are coalesced without a per-iteration transpose. Both copies are produced once from the host row-major array before the timed loop (transpose on host or by a one-time kernel; included in $T_{\text{e2e}}$, excluded from $T_{\text{alg}}$).

### DEC-010 — Global synchronization

Global phase boundaries are realized **only by kernel boundaries on a single CUDA stream** (stream order guarantees that a kernel observes all global-memory writes of preceding kernels). Cooperative-groups grid synchronization is not used. `__syncthreads()` is used only within a block.

### DEC-011 — Deliverable form: single Jupyter notebook

The final code file is `kmeans_cuda.ipynb`. The CUDA implementation stays in CUDA C++ (not CuPy or Numba), because the assignment's kernels, shared-memory tiling, atomics and launch configuration must be written and explained explicitly.

- Source files (`kmeans.cu`, `cpu_reference.cpp`, `common.h`, test drivers) are maintained in the repository under `src/` and `tests/`. The generator `tools/build_notebook.py` embeds each file verbatim in a `%%writefile` cell, so the delivered notebook contains the complete source and recreates every file when run. The notebook is a build output: it is never edited by hand, and its source cells MUST be byte-identical to the repository files (checked by the generator).
- Compilation, test runs, sanitizer runs and benchmarks are notebook cells calling `nvcc`, the compiled executables and `compute-sanitizer` through the shell (`!` prefix).
- Benchmark CSVs are loaded with pandas and plotted with matplotlib inside the notebook.
- The notebook MUST run top to bottom without manual steps ("Restart kernel and run all") on the recorded environment, and is submitted with its outputs saved.
- The notebook targets the Colab Linux runtime (ENV-11); paths and the target architecture are set in one configuration cell so the notebook also runs on Kaggle without other changes.

Alternatives considered: (a) CuPy `RawKernel` / Numba — keeps everything in Python, but hides launch, error-checking and build details the assignment asks to demonstrate, and Numba's CUDA dialect differs from CUDA C++; (b) `nvcc4jupyter` cell magic — convenient, but adds a third-party dependency and compiles each cell in isolation, which makes a multi-file program with shared headers awkward. Verified by TEST-B01, TEST-B02.

### DEC-012 — Development workflow: GitHub repository + Colab execution

The student's laptop has no NVIDIA GPU (ENV-11) and Colab has no remote-execution interface, so development follows this loop:

1. Code is written and compiled/tested where possible locally (CPU reference and host code with g++; CUDA compile-only checks if `nvcc` is installed), then committed and pushed to `https://github.com/sureshbabugandla/gpu-programming`, folder `GPU-Based-K-Means-Clustering/`.
2. The generated notebook `GPU-Based-K-Means-Clustering/kmeans_cuda.ipynb` is opened in Colab from GitHub (File → Open notebook → GitHub) on a T4 runtime and run top to bottom.
3. The executed notebook is written back with File → Save a copy in GitHub to `GPU-Based-K-Means-Clustering/results/kmeans_cuda_<YYYYMMDD-HHMM>.ipynb`, with a commit message naming the milestone.
4. Results are pulled and inspected; failures are fixed in `src/`, never in the notebook.

GPU test results are accepted only from executed notebooks committed under `results/` (Stage 3 evidence). Credentials (tokens, `kaggle.json`, Drive keys) MUST NOT be committed.

### 3.1 Decision summary table

| Decision | Selected policy | Alternatives considered | Rationale | Verification |
|---|---|---|---|---|
| Centroid accumulation (B) | Two-stage ownership-based reduction (DEC-001) | Sort + segmented reduce; masked per-cluster scan; shared-memory atomic histograms | Race-free without atomics, deterministic, $\Theta(ND)$ main work | TEST-U05–U07, I02, R01, S01–S04 |
| Centroid accumulation (A) | Element-wise global `atomicAdd` on binary64 sums and uint32 counts (§6) | Thread-per-point atomics; shared-memory privatization (OPT-001) | Simplest correct atomic design, coalesced reads | TEST-U04, I01 |
| Convergence | Assignment stability OR $\Delta \le \varepsilon$ (Euclidean, inclusive) | Single criterion; squared displacement; relative tolerance | Matches assignment wording; exact primary test | TEST-U08, I03–I05, EDGE-22–25 |
| Empty clusters | Retain previous centroid | Reinitialize to farthest point; drop cluster | Stable, deterministic, no extra pass | TEST-U09, EDGE-04, EDGE-14 |
| Final consistency | Report last (A, C) pair; P1 by construction; P2 verified as flag | Extra iterations; replacing C by means | Honest reporting within $I_{\max}$ | TEST-I06, I07, EDGE-25 |
| Tie-breaking | Lowest index on exact binary32 equality | Random; highest index | Deterministic, matches CPU reference | TEST-U02, EDGE-13 |
| Iteration counting | Number of completed centroid updates | Number of assignment passes | Matches "iterations" of Lloyd's update | TEST-I03, I05 |
| Numeric types | FP32 data/distances, FP64 sums | All FP32; all FP64 | Bandwidth vs. accuracy balance | TEST-U04–U07, NUM tests |
| Assignment mapping | Thread per point | Warp per point; thread per pair | Order-preserving, simple | TEST-U01–U03 |
| Global sync | Kernel boundaries on one stream | Grid sync via cooperative groups | Portable, provably correct | TEST-S04, code review checklist |
| Deliverable form | Single `.ipynb` generated from `src/`; CUDA C++ written by `%%writefile`, compiled by `nvcc` | Hand-edited notebook; CuPy/Numba; `nvcc4jupyter` | Requested format; real source files are easier to review and diff | TEST-B01, TEST-B02 |
| Workflow | GitHub repo + Colab run + executed notebook saved back to `results/` (DEC-012) | Kaggle API automation; browser automation | No local GPU; minimal manual steps; full history as evidence | Executed notebooks in `results/` |

---

## 4. Functional Requirements

Each block lists priority, pre/postconditions, behavior, constraints, corner cases, verification and dependencies.

**FR-001 — Parameter and data validation** `[ASG][DEC]` — MUST
- Pre: raw inputs (N, D, K, $I_{\max}$, $\varepsilon$, data file, centroid file) available.
- Behavior: apply every rule in §9.1 before any device allocation; on the first violation, emit an error naming the parameter and rule ID and exit with the code in §15.5.
- Post: either all VAL rules hold, or the program has exited with status `ERROR` and allocated no device memory.
- Corner cases: EDGE-01–03, 20, 21, 30.
- Verify: TEST-E01–E05, TEST-E20, TEST-E21. Depends: VAL-001–VAL-010.

**FR-002 — Input loading** `[ASG]` — MUST
- Behavior: read $X$ ($N\times D$) and $C^{(0)}$ ($K\times D$) from files in the format of §15.4; verify header dimensions match the declared N, D, K and that the file size equals header + payload exactly.
- Post: host arrays populated in row-major binary32.
- Corner cases: truncated file, trailing bytes, dimension mismatch → error code 5 (I/O) or 2 (validation).
- Verify: TEST-U12. Depends: BLD-004.

**FR-003 — Deterministic dataset generation (benchmark/test harness)** `[DEC]` — MUST
- Behavior: generate Gaussian-blob datasets from (N, D, K_true, separation, spread, seed) using `std::mt19937_64` raw output and a self-implemented Box–Muller transform (standard-library distributions are not portable across implementations). Initial centroids are generated as the points at indices $\lfloor qN/K \rfloor$, $q \in [0,K)$, of a seeded permutation. Datasets are written to files and identified by a dataset ID derived from all parameters.
- Post: identical parameters produce bit-identical files on the same platform.
- Verify: TEST-U13 (regenerate and compare hashes). Depends: FR-002.

**FR-004 — Initial assignment pass** `[ASG]` — MUST
- Behavior: compute $A^{(0)} = \text{ASSIGN}(X, C^{(0)})$ on the device.
- Verify: TEST-U01, TEST-I01–I02. Depends: CUDA-002.

**FR-005 — Distance computation** `[ASG]` — MUST
- Behavior: compute $d^2(x_i,c_k)$ per §2.2 for all pairs as part of assignment; no square root.
- Constraints: binary32 terms, ascending-$j$ accumulation.
- Verify: TEST-U01. Depends: NUM-002.

**FR-006 — Nearest-centroid assignment with tie rule** `[ASG][DEC]` — MUST
- Behavior: §2.3; output in $[0,K)$ for every point.
- Corner cases: identical centroids, equidistant points (EDGE-10, 12, 13).
- Verify: TEST-U02, TEST-U03. Depends: DEC-005.

**FR-007 — Assignment storage and change counting** `[ASG][DEC]` — MUST
- Behavior: store assignments in a device `int32` array of length N. During each loop assignment pass, count points whose assignment differs from the previous pass into a 64-bit device counter that is zeroed before the pass.
- Constraints: thread $i$ is the only reader and writer of element $i$ during the pass, so in-place update is permitted (read old value, compute, compare, write).
- Verify: TEST-U03 (change count exact vs. CPU). Depends: CUDA-002.

**FR-008 — Centroid sum accumulation** `[ASG]` — MUST
- Behavior: compute $S$ per §2.4 using the strategy selected by `--strategy {atomic,reduction}`.
- Post: every point contributes exactly once to exactly one cluster's sums in every dimension.
- Verify: TEST-U04–U07. Depends: §6, §7.

**FR-009 — Cluster membership counting** `[ASG]` — MUST
- Behavior: compute $n$ per §2.4 exactly (integer); $\sum_k n_k = N$.
- Verify: TEST-U04–U07 (exact equality). Depends: FR-008.

**FR-010 — Centroid mean update** `[ASG]` — MUST
- Behavior: §2.5 into a buffer distinct from the current centroids (double-buffering); pointers swapped after the iteration.
- Verify: TEST-U10. Depends: FR-008, FR-009, FR-011.

**FR-011 — Empty-cluster handling** `[DEC]` — MUST
- Behavior: DEC-003. No division is performed when $n_k = 0$.
- Verify: TEST-U09, EDGE-14. Depends: DEC-003.

**FR-012 — Convergence evaluation** `[ASG][DEC]` — MUST
- Behavior: after each update, compute $\Delta$ on the device per §2.6 (per-cluster $\delta_k^2$ in binary64, then a max reduction to one scalar); copy $\Delta$ and `changed` (16 bytes) to pinned host memory; the host applies DEC-002.
- Verify: TEST-U08, TEST-I03–I05. Depends: DEC-002.

**FR-013 — Iteration control and limit** `[ASG]` — MUST
- Behavior: §2.7 loop; never perform more than $I_{\max}$ updates; support $I_{\max}=0$.
- Verify: TEST-I05, EDGE-22, EDGE-24. Depends: DEC-006.

**FR-014 — Finalization** `[DEC]` — MUST
- Behavior: §2.8 steps 1–3.
- Verify: TEST-I06, TEST-I07, EDGE-25. Depends: DEC-004.

**FR-015 — Output reporting** `[ASG]` — MUST
- Behavior: write final assignments, final centroids, final counts and a run summary (§15.4) containing: status, iteration count, `centroid_consistent`, `assignment_consistent`, $\Delta$ of the last update (or null when $t=0$), strategy, configuration, $T_{\text{alg}}$, $T_{\text{e2e}}$, and environment fields.
- Verify: TEST-I08. Depends: FR-014, FR-017.

**FR-016 — Objective reporting** `[DEC]` — SHOULD
- Behavior: compute $J$ (§2.9) on the host in binary64 after output download, outside timed regions.
- Verify: TEST-I08.

**FR-017 — Execution-time measurement** `[ASG]` — MUST
- Behavior: report $T_{\text{alg}}$ and $T_{\text{e2e}}$ as defined in PERF-005.
- Verify: TEST-P01. Depends: PERF-005.

**FR-018 — Strategy selection** `[ASG]` — MUST
- Behavior: the accumulation strategy is a runtime option; both strategies share identical assignment, update, convergence and finalization code paths so that only accumulation differs.
- Verify: TEST-I01, TEST-I02, TEST-I09. Depends: ARC-05, ARC-06.

**FR-019 — Verification mode** `[DEC]` — MUST
- Behavior: `--verify {none,final,lockstep}`. `final`: after the run, check P1 and P2 against the CPU reference rules (NUM-005, NUM-006). `lockstep`: at every iteration, download device state and check the device's assignment pass and accumulation against the CPU reference applied to the same inputs (§10.3). Verification work is excluded from timing; runs with `lockstep` MUST NOT be used as benchmark samples.
- Verify: TEST-I07, TEST-I10. Depends: ARC-03.

**FR-020 — Fixed-iteration benchmark mode** `[DEC]` — MUST
- Behavior: `--fixed-iters` disables both convergence criteria so exactly $I_{\max}$ updates run (status `ITERATION_LIMIT`). Used only to compare strategies at equal work (PERF-004).
- Verify: TEST-P02.

---

## 5. CUDA Execution and Parallelization Requirements

**CUDA-001 — Target architecture** `[ASM]` — MUST. Build for the confirmed compute capability (ENV-01), minimum 6.0. At startup, query and record device name, compute capability, SM count, global memory, shared memory per block (default and opt-in maximum), and FP64 support; reject devices below 6.0 with exit code 3.

**CUDA-002 — Kernel responsibilities** `[DEC]` — MUST. The device code consists of the following kernels with single, documented responsibilities:

| Kernel | Responsibility | Launch mapping |
|---|---|---|
| `assign` (baseline) | §2.3 per point; in-place assignment write; per-block change count then one 64-bit `atomicAdd` per block into the change counter | 1-D grid-stride, one thread per point |
| `assign_smem` | Same contract, centroid tiles in shared memory (SMEM-001) | 1-D, one thread per point, block-cooperative tile loads |
| `accum_atomic` | Approach A (§6) | 1-D grid-stride over $N\cdot D$ elements |
| `reduce_stage1` | Approach B Stage 1 (§7) | 1-D grid-stride over $G\cdot D$ workers |
| `reduce_stage2` | Approach B Stage 2 (§7) | 1-D grid-stride over tiles, block $32\times8$ |
| `update` | §2.5 into the next-centroid buffer | 1-D grid-stride over $K\cdot D$ |
| `movement` | $\delta_k^2$ per cluster then block max; writes per-block maxima | 1-D grid-stride over $K$ |
| `max_reduce` | Final max over per-block maxima → one binary64 scalar | single block |

The change counter is an integer; integer atomics are exact, so the counter is deterministic and does not compromise Approach B's atomic-free accumulation.

**CUDA-003 — Per-iteration launch sequence** `[DEC]` — MUST. On one stream, in order: accumulation (A: `cudaMemsetAsync` of sums and counts, then `accum_atomic`; B: `reduce_stage1`, `reduce_stage2`) → `update` → `movement` → `max_reduce` → `cudaMemsetAsync` of change counter → `assign` → async copy of $\Delta$ and `changed` to pinned host memory → stream synchronization → host decision → pointer swap of centroid buffers. No kernel of iteration $t+1$ is launched before the host decision for iteration $t$.

**CUDA-004 — No intra-kernel global phase boundaries** `[DEC]` — MUST. `__syncthreads()` synchronizes only the threads of one block. No kernel may read data written by another block of the same launch unless the write-read relation is through an atomic whose result is not consumed within that launch. In particular, centroids MUST NOT be updated in the same kernel launch that computes assignments or accumulates sums. Verify: design review checklist (TEST-S05) and TEST-S04.

**CUDA-005 — Bounds checks** `[DEC]` — MUST. Every kernel guards every global index against its range ($i<N$, $j<D$, $k<K$, $g<G$, $w<GD$, $e<ND$). Threads that fall outside the valid range in kernels containing `__syncthreads()` MUST NOT return early; they skip work but participate in every barrier. Verify: TEST-E16, TEST-E17, TEST-S01, TEST-S04.

**CUDA-006 — 64-bit indexing** `[DEC]` — MUST. All element offsets and grid-stride loop variables are unsigned 64-bit. Products $N\cdot D$, $K\cdot D$, $G\cdot K\cdot D$ are computed with overflow checks on the host before allocation (MEM-002). Verify: TEST-E30.

**CUDA-007 — Grid sizing** `[DEC]` — MUST. All kernels use 1-D grids with grid-stride loops; grid size $= \min(\lceil \text{work}/\text{block}\rceil,\ 32\times\text{SM count})$ unless overridden. No kernel relies on grid dimensions $y$ or $z$ (limited to 65 535). Default block size 256 (configurable via `--block-size`, validated as a multiple of 32 in $[32, 1024]$ and against the device limit).

**CUDA-008 — Error checking** `[DEC]` — MUST. Every CUDA runtime API call is checked; failure produces a message with file, line, call and `cudaGetErrorString`, cleanup (MEM-007) and exit code 4. After each kernel launch, `cudaGetLastError()` is checked (launch errors). Asynchronous execution errors are detected at the per-iteration stream synchronization; a debug build option (`KMEANS_SYNC_CHECK`) synchronizes after every launch to localize faults. Verify: TEST-E26–E28.

**CUDA-009 — Memory access patterns** `[DEC]` — SHOULD. `assign` reads $X^{\mathsf{col}}$ so that consecutive threads read consecutive addresses for each $j$; centroid reads are uniform across a warp (broadcast). `accum_atomic` maps flat element $e$ to $(i = e \bmod N,\ j = \lfloor e/N \rfloor)$ over $X^{\mathsf{col}}$. `reduce_stage1` maps worker $w$ to $(g = \lfloor w/D \rfloor,\ j = w \bmod D)$ over $X^{\mathsf{row}}$. Verify: Nsight Compute load efficiency (PERF-009, optional evidence).

**CUDA-010 — Registers and occupancy** `[DEC]` — SHOULD. Record per-kernel register and shared-memory usage from `-Xptxas -v` in the build log. `__launch_bounds__` MAY be used; any spill to local memory in `assign_smem` MUST be reported with benchmark results.

**CUDA-011 — Host–device transfers** `[ASG][DEC]` — MUST. Inputs are copied host→device once before the loop; outputs device→host once after finalization; within the loop only the 16-byte convergence scalars are transferred (pinned host buffer). Verify: Nsight Systems trace (TEST-P03).

**CUDA-012 — Non-aligned sizes** `[DEC]` — MUST. Correctness for $N$, $D$, $K$ not divisible by block size, tile sizes or 32. Verify: TEST-E16.

---

## 6. Approach A — Atomic-Based Accumulation

**ATOM-001 — Buffer initialization** — MUST. Before each accumulation (loop and finalization), zero the device sum buffer ($K\times D$ binary64) and count buffer ($K$ uint32) with `cudaMemsetAsync` on the same stream. Buffers are reused across iterations; reuse is safe because initialization is stream-ordered after the previous consumer (`update`).

**ATOM-002 — Parallel processing** — MUST. One logical thread per element $e \in [0, N\cdot D)$, grid-stride, mapping per CUDA-009.

**ATOM-003 — Atomic sum** — MUST. Thread for $(i,j)$ performs `atomicAdd(&sum[A_i*D + j], (double)x_ij)`. Requires compute capability ≥ 6.0 (CUDA-001).

**ATOM-004 — Atomic count** — MUST. The thread with $j = 0$ performs `atomicAdd(&count[A_i], 1u)`.

**ATOM-005 — Synchronization before update** — MUST. `update` is launched after `accum_atomic` on the same stream (DEC-010).

**ATOM-006 — Division and empty clusters** — MUST. Shared `update` kernel (FR-010, FR-011).

**ATOM-007 — Race-safety versus reproducibility** — MUST (documentation and test). Atomic read-modify-write guarantees no lost updates: each contribution is added exactly once. It does **not** guarantee a fixed order of floating-point additions, so $S_{kj}$ may differ between runs in the last bits of binary64. After division and rounding to binary32 this rarely changes $c_{kj}$, but it can, and a changed centroid can flip a near-tie assignment and change the trajectory. Counts are exact integers and are always reproducible. The program MUST NOT claim bitwise reproducibility for Approach A.

**ATOM-008 — Correctness verification despite nondeterminism** — MUST. Counts compared exactly; sums/centroids compared with NUM-006 tolerances; run-to-run differences measured and reported (TEST-R02), not treated as failures.

**ATOM-009 — Error handling** — MUST. As CUDA-008.

**ATOM-010 — Contention note** — documentation. All threads for points of one cluster target the same $D$ addresses; contention grows as $K$ decreases. Benchmarks MUST include small $K$ (PERF-003) so this effect is observable.

---

## 7. Approach B — Reduction-Based Accumulation

The selected design is DEC-001. This section states its implementation contract.

### 7.A Reduction data model

**RED-001** — MUST. Buffers: $P$ (binary64, $G\times K\times D$, offset $(g\cdot K + k)\cdot D + j$), $Q$ (uint32, $G\times K$, offset $g\cdot K + k$), and outputs $S$ ($K\times D$ binary64) and $n$ ($K$ uint32) shared with the `update` kernel. The cluster receiving each contribution is $A_i$ read from the assignment array produced by the preceding `assign` launch.

### 7.B Parallel execution

**RED-002 — Stage 1 worker contract** — MUST. Worker $w \in [0, G\cdot D)$, $g = \lfloor w/D\rfloor$, $j = w \bmod D$:
1. For $k \in [0,K)$: $P[g][k][j] \leftarrow 0$; if $j=0$: $Q[g][k] \leftarrow 0$.
2. For $i$ in chunk $g$ in ascending order: $k \leftarrow A_i$; $P[g][k][j] \mathrel{+}= x_{ij}$; if $j=0$: $Q[g][k] \mathrel{+}= 1$.

Chunk bounds use 64-bit arithmetic exactly as in DEC-001. A grid-stride thread MAY process several workers sequentially.

**RED-003 — Stage 2 contract** — MUST. For tile $\tau \in [0, K\cdot T)$, $T = \lceil D/32\rceil$, $k = \lfloor \tau / T \rfloor$, $j = (\tau \bmod T)\cdot 32 + x$:
1. Thread $(x,y)$: $s \leftarrow \sum_{g \equiv y \ (\mathrm{mod}\ 8),\ g<G} P[g][k][j]$ in ascending $g$ (0 if $j \ge D$ or no such $g$); same for counts when $\tau \bmod T = 0$ and $x=0$ uses lanes $y$ over $Q[g][k]$.
2. Store $s$ in shared memory `[8][32]`; `__syncthreads()`; fixed tree over $y$ with strides 4, 2, 1 and `__syncthreads()` after each level.
3. Thread $(x, 0)$ writes $S_{kj}$ if $j < D$; the count tile writes $n_k$.

All threads of the block execute every barrier (CUDA-005).

### 7.C Synchronization and correctness

**RED-004 — Race freedom argument** — MUST (documented in code review and in the notebook).
- Stage 1: each element of $P$ and $Q$ has exactly one owning worker, which performs all writes to it; $A$ and $X^{\mathsf{row}}$ are read-only during Stage 1. No two threads write the same address → no data races, no lost updates.
- Stage 2: each element of $S$ and $n$ is written by exactly one thread; $P$ and $Q$ are read-only; shared-memory slots are written by their own thread before a barrier and read by others only after it.
- Stage 1 → Stage 2 and Stage 2 → `update` boundaries are kernel boundaries (DEC-010); no grid-wide synchronization inside a kernel is assumed.

**RED-005 — Exactly-once inclusion** — MUST. The chunks partition $[0,N)$ (consecutive bounds share endpoints, $\text{chunk}_0$ starts at 0, $\text{chunk}_{G-1}$ ends at $N$), each worker covers all points of its chunk for its $j$, and Stage 2 sums all $g \in [0,G)$. Hence every $(i,j)$ contributes once. Verified by TEST-U05 ($\sum_k n_k = N$ exactly and per-cluster sums vs. reference) and TEST-U07 (sum of all $S_{kj}$ over $k$ equals the column sum of $X$ within tolerance).

**RED-006 — Initialization and empty contributions** — MUST. Every element of $P$ and $Q$ is overwritten in Stage 1 step 1 of every accumulation, so no stale data from earlier iterations survives. A chunk with no points of cluster $k$ leaves zeros, which are valid contributions. $G \le N$ guarantees every chunk is non-empty, but correctness does not depend on it. Verified by poisoning $P$, $Q$ with NaN / 0xFFFFFFFF before the first accumulation (TEST-U06) and `compute-sanitizer --tool initcheck` (TEST-S03).

### 7.D Memory and resource management

**RED-007 — Memory model** — MUST. Extra memory $= 8GKD + 4GK$ bytes. $G$ is chosen per DEC-001 such that this fits within $B_{\text{red}}$ (MEM-003). If $G_{\text{mem}} < 1$ the run fails before allocation with exit code 3 and a message stating the required bytes. The selected $G$ is printed and recorded in benchmark CSV.

**RED-008 — Trade-offs documented** — MUST. The notebook states: larger $G$ increases Stage 1 parallelism and memory/Stage 2 work; smaller $G$ lengthens per-worker sequential loops; small $D$ yields $G\cdot D$ workers that may under-fill the GPU; Stage 1 RMW traffic to $P$ goes through L2/DRAM.

### 7.E Selected baseline justification

Of the considered designs (DEC-001), the ownership-based two-stage reduction is the simplest design that (i) uses no atomics for centroid data, (ii) does main work proportional to $ND$ rather than $NKD$, (iii) has a short correctness proof, and (iv) gives a deterministic order for fixed configuration. Its weaknesses (memory $\propto GKD$, under-utilization at small $D$) are measurable and are part of the comparison.

### 7.F Numerical behavior

**RED-009** — MUST. For fixed $(N, D, K, G)$, binary and device, the summation order of every $S_{kj}$ is fixed: ascending $i$ within a chunk, ascending $g$ within each $y$-lane, then the fixed tree. The design is therefore expected to be bitwise reproducible run to run; this is a claim to be verified by TEST-R01, not assumed. Results may differ from Approach A and from the CPU reference in the last bits of binary64 sums, with effects described in NUM-005.

---

## 8. Shared-Memory Optimization Requirements

**SMEM-001 — Tiled assignment kernel** `[ASG]` — MUST. `assign_smem` processes centroids in tiles of $T_K$ centroids × $T_D$ dimensions (defaults $T_K = 8$, $T_D = 32$; compile-time constants, alternatives configurable at build). Each thread keeps $T_K$ partial distances in registers for its point:
1. For each centroid tile $[k_0, k_0+T_K)$: zero the $T_K$ partials.
2. For each dimension tile $[j_0, j_0+T_D)$: the block cooperatively loads $C[k_0{:}k_0{+}T_K][j_0{:}j_0{+}T_D]$ into shared memory (out-of-range elements are not read and are not used); `__syncthreads()`; each thread with $i<N$ adds terms for $j$ in ascending order within the tile; `__syncthreads()` before the tile is overwritten.
3. After all dimension tiles, compare the $T_K$ partials to the running best in ascending $k$ with strict-less-than (DEC-005), skipping $k \ge K$.

Summation over $j$ remains ascending from $j=0$ per (point, centroid), matching NUM-002.

**SMEM-002 — Barrier participation** — MUST. Threads with $i \ge N$ participate in all loads and barriers (CUDA-005).

**SMEM-003 — Capacity** — MUST. Shared memory $= 4\,T_K T_D$ bytes per block (1 KiB at defaults), checked against the device limit at startup.

**SMEM-004 — Correctness baseline first** — MUST. `assign` (baseline) is implemented and passes all tests before `assign_smem`. `assign_smem` MUST produce identical assignments to `assign` on all test datasets when both are compiled with the same FMA settings; any difference MUST be explained (expected only from FMA contraction differences) and is bounded by NUM-005.

**SMEM-005 — Evaluation** — MUST. Benchmarked against `assign` with identical inputs and strategy (PERF-003). Expected benefit: fewer global/L1 transactions for centroid reads when $K\cdot D$ is large relative to L1, at the cost of barriers and $T_K$ registers per thread. Shared memory is not assumed to be faster; the notebook reports where it helps and where it does not, with measurements.

**SMEM-006 — Optional whole-set variant (OPT-003)** — MAY. When $4KD$ bytes fit in shared memory, load all centroids once per block. Same correctness contract.

---

## 9. Input Validation and Corner Cases

### 9.1 Validation rules

| ID | Rule | On violation |
|---|---|---|
| VAL-001 | $N \ge 1$, $D \ge 1$, $K \ge 1$ (integers) | reject, exit 2 |
| VAL-002 | $N \le 2^{31}-1$, $D \le 2^{31}-1$, $K \le 2^{31}-1$ (assignment values fit `int32`, counts fit `uint32`) | reject, exit 2 |
| VAL-003 | $K > N$ is **accepted** with warning `W-KGTN`; at least $K-N$ clusters remain empty and retain their initial centroids (DEC-003) | warn, continue |
| VAL-004 | $I_{\max}$ integer, $0 \le I_{\max} \le 2^{31}-1$; $I_{\max}=0$ accepted (§2.7) | reject if outside, exit 2 |
| VAL-005 | $\varepsilon$ parses as a finite binary64 with $\varepsilon \ge 0$; NaN, ±Inf, negative rejected | reject, exit 2 |
| VAL-006 | Every coordinate of $X$ and $C^{(0)}$ is finite (no NaN/Inf) | reject, exit 2, report first offending (row, column) |
| VAL-007 | Range bound: with $M = \max(\max_{ij}|x_{ij}|, \max_{kj}|c^{(0)}_{kj}|)$, require $4 D M^2 \le 2^{126}$ (evaluated in binary64). Since all later centroids are means of points or retained initial centroids, they lie within $[-M, M]^D$ up to rounding, so no binary32 distance can overflow during the run | reject, exit 2, message states $M$ and the bound |
| VAL-008 | Initial centroid file has exactly $K$ rows of $D$ values; duplicates accepted with warning `W-DUPC` | reject on shape mismatch (exit 2); warn on duplicates |
| VAL-009 | All size products and allocation sizes computed with overflow-checked 64-bit arithmetic; total required device memory (MEM-002) ≤ free device memory reported by `cudaMemGetInfo` minus a safety margin of 5 % | reject, exit 3, message states required and available bytes |
| VAL-010 | Option values (`--block-size`, `--reduce-chunks`, tile sizes) within documented ranges and device limits | reject, exit 2 |

The program never continues with undefined behavior after a failed rule.

### 9.2 Corner-case matrix

Disposition: **A** accepted (normal processing), **R** rejected with error, **S** accepted with special handling/warning, **U** reported as unsupported.

| ID | Scenario | Disp. | Expected behavior | Acceptance criterion | Test |
|---|---|---|---|---|---|
| EDGE-01 | $N=0$ | R | VAL-001 error, exit 2, no device allocation | exit code 2; message names N | TEST-E01 |
| EDGE-02 | $D=0$ | R | as above for D | exit 2 | TEST-E02 |
| EDGE-03 | $K=0$ | R | as above for K | exit 2 | TEST-E03 |
| EDGE-04 | $K>N$ | S | warning `W-KGTN`; ≥ $K-N$ empty clusters retain $C^{(0)}$ | counts: ≥ $K-N$ zeros; retained centroids bit-equal to input; matches CPU | TEST-E04 |
| EDGE-05 | $N=K$ | A | normal; with distinct points and $C^{(0)} = X$ converges at $t=1$ with each point its own cluster | status `CONVERGED_ASSIGNMENTS`, $t=1$, all $n_k=1$ | TEST-E05 |
| EDGE-06 | $N=1$ | A | single point assigned to nearest centroid; converges at $t=1$ | $t=1$, $n$ has one 1; P2 true | TEST-E06 |
| EDGE-07 | $D=1$ | A | normal | matches hand results H1–H3 | TEST-I03 |
| EDGE-08 | $K=1$ | A | all points in cluster 0; centroid = global mean; converges at $t=1$ | $A \equiv 0$; centroid within NUM-006 of mean | TEST-E08 |
| EDGE-09 | Duplicate input points | A | duplicates always share a cluster | duplicates have equal assignments | TEST-E09 |
| EDGE-10 | Duplicate initial centroids | S | warning `W-DUPC`; higher-index duplicate receives no points while identical (tie rule) and retains its value | lower index gets the points; matches CPU | TEST-E10 |
| EDGE-11 | All input points identical | A | all assigned to the lowest-index nearest centroid; converges at $t=1$ | exact match with CPU | TEST-E11 |
| EDGE-12 | All centroids identical | S | all points to cluster 0; others empty and retained | $A\equiv 0$; $n_0 = N$ | TEST-E12 |
| EDGE-13 | Points exactly equidistant from several centroids | A | lowest index chosen (DEC-005) | H3 and constructed 2-D cases match exactly | TEST-U02 |
| EDGE-14 | Empty cluster arising during iteration | S | retain centroid; excluded from $\Delta$ | H2 variant and constructed case match CPU | TEST-U09 |
| EDGE-15 | Cluster with exactly one point | A | centroid equals that point | bit-equal to the point (division by 1 exact) | TEST-E15 |
| EDGE-16 | $N, D, K$ not multiples of block/tile/32 sizes | A | correct via bounds checks | results equal to CPU for $N\in\{1,31,33,255,257,1023\}$, $D\in\{1,3,31,33,65\}$, $K\in\{1,3,7,9,17\}$ | TEST-E16 |
| EDGE-17 | Partial CUDA blocks | A | out-of-range threads do no work, join barriers | sanitizer clean; results match | TEST-E16, TEST-S04 |
| EDGE-18 | Very large magnitudes within VAL-007 | A | no overflow; correct | data scaled to $M$ near the bound runs without Inf; matches CPU within NUM-006 | TEST-E18 |
| EDGE-19 | Very small coordinate differences | A | distances may round; ties resolved by DEC-005; differences below binary32 resolution are indistinguishable by design | assignment mismatches vs. FP64 oracle only on near-ties (NUM-005) | TEST-E19 |
| EDGE-20 | NaN or ±Inf in $X$ or $C^{(0)}$ | R | VAL-006, exit 2 with location | exit 2; row/column reported | TEST-E20 |
| EDGE-21 | $\varepsilon$ negative, NaN, Inf, unparsable | R | VAL-005, exit 2 | exit 2 | TEST-E21 |
| EDGE-22 | $I_{\max}=0$ | A | §2.7: $t=0$, `ITERATION_LIMIT`, $C^{\text{final}} = C^{(0)}$, $A = A^{(0)}$, P2 verified | H2 with $I_{\max}=0$: $A=\{0,1,1,1\}$, $C=\{0,3\}$, P2 false | TEST-I05 |
| EDGE-23 | Convergence on first iteration | A | `CONVERGED_ASSIGNMENTS`, $t=1$ | H1: $t=1$ | TEST-I03 |
| EDGE-24 | No convergence within limit | A | `ITERATION_LIMIT`, $t=I_{\max}$, P2 reported | H2 with $I_{\max}=2$: $t=2$, $C=\{1,6.5\}$, P2 false | TEST-I05 |
| EDGE-25 | Assignments change after the final update | A | report $(A^{(t)}, C^{(t)})$, P1 true, P2 false where applicable | H2 with $\varepsilon=2$: `CONVERGED_MOVEMENT`, $t=1$, $A=\{0,0,1,1\}$, $C=\{0,5\}$, P2 false | TEST-I04 |
| EDGE-26 | Insufficient GPU memory | R | VAL-009 pre-check fails, exit 3 | exit 3; required/available bytes printed | TEST-E26 |
| EDGE-27 | CUDA allocation failure despite pre-check | R | CUDA-008, cleanup, exit 4 | fault-injection hook forces failure; exit 4; no leak under sanitizer leak check | TEST-E27 |
| EDGE-28 | Kernel launch failure | R | CUDA-008, exit 4 | test hook forces invalid launch; exit 4 | TEST-E28 |
| EDGE-29 | Large $N$, $D$ or $K$ | A/R | accepted if VAL-009 passes, otherwise rejected | largest benchmark size runs; oversize request exits 3 | TEST-E29 |
| EDGE-30 | Index products near or above $2^{63}$ / allocation overflow | R | overflow-checked arithmetic rejects, exit 2 or 3 | unit test of size calculator with extreme values | TEST-E30 |
| EDGE-31 | FP differences between atomic and reduction | A | expected; within NUM-006 for lockstep accumulation | TEST-I09 statistics reported | TEST-I09 |
| EDGE-32 | Near-equal distances flip under different accumulation results | A | allowed only on near-ties; trajectories may diverge; diagnosed per NUM-005 | mismatch report lists points, margins | TEST-I10 |
| EDGE-33 | Repeated execution, identical inputs | A | reduction: bitwise identical outputs expected; atomic: may differ, differences reported | TEST-R01 (equal hashes over 10 runs), TEST-R02 (report) | TEST-R01, R02 |
| EDGE-34 | Shared-memory tile size near hardware limit | S/U | tile configurations exceeding the per-block limit are rejected at startup (exit 2); configurations at the limit run correctly | max legal config runs and matches; oversize rejected | TEST-E34 |
| EDGE-35 | Reduction chunks with no points of some cluster / partial tiles in Stage 2 | A | zeros contribute nothing; partial $j$-tiles guarded | constructed case with $K \gg$ points-per-chunk; matches CPU | TEST-U07 |

---

## 10. Numerical Accuracy and Correctness Requirements

**NUM-001 — Types** — MUST. Per DEC-007.

**NUM-002 — Distance semantics** — MUST. Per §2.2. Device code MUST NOT be compiled with `--use_fast_math`. The default `-fmad=true` is permitted; a verification build with `-fmad=false` MAY be used to reduce near-tie mismatches.

**NUM-003 — CPU reference (oracle)** — MUST. A single-threaded C++ implementation of §2 with: binary32 distance path (ascending $j$, compiled without fast-math and with `-ffp-contract=off`), a binary64 distance path used only for near-tie diagnosis, sequential binary64 sums in ascending $i$, identical tie rule, empty-cluster policy, convergence rule, iteration counting and finalization.

**NUM-004 — Exact-equality quantities** — MUST. Compared exactly: counts $n_k$; change counts; iteration count, status and flags on datasets designated *well-separated* (§10.3); assignments except near-ties (NUM-005); centroids of single-point clusters; retained centroids of empty clusters.

**NUM-005 — Near-tie rule for assignments** — MUST. For point $i$, let $d_{(1)} \le d_{(2)}$ be the two smallest binary64 distances to the centroids under test. Define the relative margin $m_i = (d_{(2)} - d_{(1)}) / \max(d_{(2)}, \text{FLT\_MIN})$. A point is a *near-tie* if $m_i \le \delta_{\text{tie}}$ with

$$
\delta_{\text{tie}} = 4\,(D+2)\,u_{32}.
$$

Rationale: each binary32 term $(x-c)^2$ carries relative error $\le 3u_{32}$ (approximately) and the sum of $D$ non-negative terms adds at most $(D-1)u_{32}$ relative error, so computed binary32 distances can reorder two candidates only when their true relative gap is of order $(D+2)u_{32}$; the factor 4 adds margin. An assignment mismatch between device and reference is a **failure** if the point is not a near-tie and is **reported** (point index, both candidate clusters, $m_i$) if it is.

**NUM-006 — Centroid tolerance** — MUST. With $S_{\max} = \max(M, 1\text{e-}30)$ ($M$ from VAL-007), two centroid values $a$ (test) and $b$ (reference) agree if

$$
|a - b| \le 4u_{32}\,|b| + 4u_{32}\,S_{\max}\cdot 2^{-10}
$$

for FP64 accumulation (both sides round a binary64 quotient once to binary32, so they can differ by at most about one binary32 ulp; the absolute term covers values near zero). For OPT-002 (FP32 accumulation), the tolerance is $|a-b| \le 10^{-4}(|b| + S_{\max}\cdot 2^{-10})$ and the observed maximum error MUST be reported.

**NUM-007 — Movement and convergence** — MUST. $\Delta$ per §2.6. In end-to-end comparisons the tolerance $\varepsilon$ is chosen so that the reference's $\Delta$ is not within $10^{-6}\varepsilon$ of $\varepsilon$ at any iteration, except in dedicated boundary tests (TEST-I04), where values are exactly representable.

**NUM-008 — No bitwise default** — MUST. Floating-point outputs are never compared bitwise by default; bitwise comparison is used only in TEST-R01 (reproducibility claim for Approach B), EDGE-15, EDGE-04/10/12 (retained or copied values) and exact hand cases.

### 10.3 Comparison methodology

1. **Lockstep step tests (primary for general data).** At each iteration, the device's current $C^{(t)}$ is given to the CPU assignment step and compared with the device's $A^{(t+1)}$ (NUM-005); the device's $A^{(t)}$ is given to the CPU accumulation and compared with device $n$ (exact) and $C^{(t+1)}$ (NUM-006). This validates every kernel on realistic data without requiring identical trajectories.
2. **End-to-end on well-separated data.** Blobs with separation ≥ 10 × spread and initial centroids drawn from distinct blobs; no near-ties are expected (verified by computing all $m_i$). Device and CPU must agree exactly on assignments, counts, iteration count and status; centroids within NUM-006.
3. **End-to-end on general data.** Trajectories may diverge after a near-tie flip. Pass criteria: P1 verified with NUM-005; P2 flag correct; status consistent with the device's own $\Delta$/change values. Objective $J$ and assignment agreement with CPU are reported, not thresholded.
4. **Cross-strategy.** Approach A vs. Approach B on identical inputs under the same three methods; differences reported (TEST-I09).

---

## 11. Non-Functional Requirements

| ID | Category | Requirement | Priority | Acceptance criterion |
|---|---|---|---|---|
| NFR-001 | Correctness | All MUST requirements of §§2–10 hold | MUST | All acceptance tests in §16.7 pass |
| NFR-002 | Numerical stability | No Inf/NaN produced for inputs passing VAL rules | MUST | TEST-E18 and all runs: outputs finite |
| NFR-003 | Memory safety | No out-of-bounds or misaligned device accesses | MUST | `compute-sanitizer --tool memcheck` reports 0 errors on the sanitizer suite (TEST-S01) |
| NFR-004 | Race safety | No shared-memory hazards; global-memory race freedom by RED-004 argument | MUST | `racecheck` 0 hazards (TEST-S02); RED-004 reviewed (TEST-S05) |
| NFR-005 | Synchronization correctness | No barrier divergence; kernel-boundary phase ordering | MUST | `synccheck` 0 errors (TEST-S04) |
| NFR-006 | Initialization | No uninitialized device reads | MUST | `initcheck` 0 errors (TEST-S03) |
| NFR-007 | Reproducibility | Approach B bitwise reproducible for fixed config; Approach A differences quantified | MUST (claim verified) | TEST-R01, TEST-R02 |
| NFR-008 | Performance | Both strategies benchmarked per §13; no target speed fixed before hardware is known | MUST | PERF results delivered |
| NFR-009 | Scalability | Runs for every configuration in the benchmark matrix that passes VAL-009 | MUST | PERF-003 matrix complete or exclusions justified |
| NFR-010 | Resource utilization | Register/shared usage and achieved occupancy reported for main kernels | SHOULD | build log + profiler output |
| NFR-011 | Portability | Builds and runs on any device with CC ≥ 6.0 with the confirmed toolkit; no vendor library dependencies beyond the CUDA runtime | MUST | Build on the confirmed environment; arch configurable |
| NFR-012 | Maintainability | Components per §14 with the stated interfaces; no duplicated algorithm logic between strategies outside accumulation | SHOULD | Code review |
| NFR-013 | Error handling | Every failure path yields a message and a documented exit code; resources released | MUST | TEST-E26–E28 |
| NFR-014 | Build reliability | "Restart kernel and run all" on a fresh copy of the notebook completes without errors; compile cells produce no host-code warnings | MUST | TEST-B01 |
| NFR-015 | Testability | One notebook cell runs the full test suite and raises an exception (stopping run-all) on any failure | MUST | TEST-B02 |

---

## 12. Memory and Resource Management Requirements

**MEM-001 — Device buffers** — MUST.

| Buffer | Type | Elements | Bytes | Lifetime |
|---|---|---|---|---|
| $X^{\mathsf{col}}$ | f32 | $ND$ | $4ND$ | run |
| $X^{\mathsf{row}}$ | f32 | $ND$ | $4ND$ | run |
| $C_{\text{cur}}, C_{\text{next}}$ | f32 | $2KD$ | $8KD$ | run (pointer swap) |
| $A$ | i32 | $N$ | $4N$ | run |
| $S$ | f64 | $KD$ | $8KD$ | run, re-initialized each accumulation |
| $n$ | u32 | $K$ | $4K$ | run, re-initialized each accumulation |
| $P, Q$ (Approach B only) | f64 / u32 | $GKD$, $GK$ | $8GKD + 4GK$ | run, overwritten each accumulation |
| movement block maxima | f64 | ≤ grid size | small | run |
| $\Delta$ scalar, change counter | f64, u64 | 2 | 16 | run |
| pinned host scalars | f64, u64 | 2 | 16 | run |

**MEM-002 — Capacity model** — MUST. Required bytes $= 8ND + 4N + 16KD + 4K + [8GKD + 4GK]_{B} + \text{small}$, computed with overflow checks and compared with free memory (VAL-009) before any allocation.

**MEM-003 — Reduction budget** — MUST. $B_{\text{red}} = \min(\text{user cap if given},\ 0.25 \times (\text{free} - \text{other buffers}))$.

**MEM-004 — Shared memory** — MUST. `assign_smem`: $4T_KT_D$ B; `reduce_stage2`: $8\cdot 8\cdot 32$ B for sums $+ 4\cdot 8$ B for counts; `assign` and `movement` reductions: $\le 8 \times$ block size B. All checked against the device per-block limit.

**MEM-005 — Initialization and reuse** — MUST. $S$, $n$, change counter: `cudaMemsetAsync` before use each iteration (Approach A and counter) or full overwrite (Approach B $S,n$ by Stage 2; $P,Q$ by Stage 1). $C_{\text{next}}$ is fully overwritten by `update` (every $(k,j)$ written: mean or retained value).

**MEM-006 — Stale-data prevention** — MUST. No kernel reads a buffer element that has not been written in the current phase or loaded from input; verified by poisoning tests (TEST-U06) and `initcheck`.

**MEM-007 — Cleanup** — MUST. Device and pinned allocations are owned by RAII wrappers on the host so every exit path, including errors, releases them; verified with `compute-sanitizer --leak-check full` (TEST-S01).

**MEM-008 — Fault injection** — SHOULD. A test-only hook (environment variable or compile-time flag) makes the N-th allocation or a chosen launch fail, to exercise EDGE-27/28 without exhausting real memory.

---

## 13. Performance and Benchmarking Requirements

**PERF-001 — Correctness gate** — MUST. Every benchmark configuration is first run once with `--verify final`; configurations failing verification are recorded with `correctness_status=FAIL` and excluded from performance conclusions.

**PERF-002 — Identical workloads** — MUST. Strategy and kernel variants are compared on the same dataset files, the same $C^{(0)}$, the same $I_{\max}$ and $\varepsilon$, the same block size and the same build.

**PERF-003 — Benchmark matrix** — MUST. One-factor-at-a-time sweeps around a base point, plus a small full grid. Sizes are adjusted to the confirmed GPU memory (ENV-01); any excluded configuration is listed with the reason.

| Sweep | N | D | K | Variants |
|---|---|---|---|---|
| Base point | $10^6$ | 32 | 64 | A, B × `assign`, `assign_smem` |
| N sweep | $10^4, 10^5, 10^6, 10^7$ | 32 | 64 | same |
| D sweep | $10^6$ | 2, 8, 32, 128, 512 | 64 | same |
| K sweep | $10^6$ | 32 | 2, 8, 64, 256, 1024 | same |
| Block size | base point | | | 128, 256, 512 |
| Reduction $G$ | base point | | | auto, ¼×, 4× auto |
| Tile sizes | base point and D = 512, K = 1024 | | | $T_K \in \{4,8,16\}$, $T_D \in \{16,32\}$ |
| Small grid | $\{10^4, 10^6\}$ | $\{4, 64\}$ | $\{4, 256\}$ | A, B |

Datasets: Gaussian blobs (FR-003) with $K_{\text{true}} = K$, fixed seed per configuration.

**PERF-004 — Fixed-work comparison** — MUST. Timing comparisons of strategies use `--fixed-iters` with $I_{\max}=20$ (default; configurable) so both perform the same number of updates. Converging runs are reported separately with their iteration counts, and per-iteration time $T_{\text{alg}}/(t+1)$ is reported.

**PERF-005 — Timing definitions** — MUST.
- $T_{\text{alg}}$ (primary "execution time"): CUDA events recorded on the stream immediately before the initial `assign` and immediately after finalization accumulation; includes all loop kernels, per-iteration scalar transfers and host decision gaps; excludes input H2D, layout preparation and output D2H.
- $T_{\text{e2e}}$: host `std::chrono::steady_clock` from the start of input H2D to completion of output D2H (after allocation); excludes CUDA context creation, file I/O and allocation.
- $T_{\text{alloc}}$, $T_{\text{H2D}}$, $T_{\text{D2H}}$ reported separately.
- Elapsed times are read only after `cudaEventSynchronize` on the end event.

**PERF-006 — Repetitions** — MUST. 3 warm-up runs (not recorded) then ≥ 10 measured repetitions in the same process on preloaded data; report mean, median (primary), standard deviation, min and max.

**PERF-007 — Phase timing** — MUST. A separate `--profile-phases` mode records events around each phase (assign, accumulate, update, convergence) and sums them across iterations. Phase-timing runs are reported separately because extra events can perturb totals; they are not mixed with headline numbers.

**PERF-008 — Environment record** — MUST. Each CSV row records GPU name, compute capability, SM count, driver version, CUDA runtime version, `nvcc` version, host compiler and version, OS, build type and flags, git commit.

**PERF-009 — Profiling evidence** — SHOULD. For the base point, Nsight Systems timeline and Nsight Compute metrics (achieved occupancy, memory throughput, atomic throughput, load efficiency) for each main kernel.

**PERF-010 — Analysis** — MUST. The report explains observed differences using measurements and the models below; no claim that either strategy is faster is made without data.
- Assignment: $\approx 3NKD$ flops, $4ND$ bytes of point reads per pass; compute-bound for large $K$.
- Approach A: $ND$ binary64 atomics onto $KD$ addresses; contention $\approx N/K$ per address.
- Approach B: $ND$ reads, $ND$ binary64 RMWs to private partials, $GKD$ zeroing and $GKD$ combine reads; FP64 throughput of the device (ENV-06) matters.

**PERF-011 — No fabrication** — MUST. Only measured values are reported; missing configurations are marked as not run.

**PERF-012 — Cross-GPU comparison (OPT-006)** — MAY. Optional extension, started only after the primary evaluation on the T4 is complete (all §16 tests passed and the full PERF-003 matrix measured).

| Aspect | Requirement |
|---|---|
| Purpose | Test whether the relative ranking of Approach A vs. B, and of `assign` vs. `assign_smem`, changes with GPU architecture, and explain any change from FP64 throughput, memory bandwidth, L2 size and atomic throughput. Absolute speed-up between GPUs is reported but is not the goal. |
| Primary GPU | Tesla T4 (Colab free tier), full matrix. |
| Secondary GPU | One of A100 (preferred: high FP64 rate, high bandwidth), L4 or G4, whichever is available. More than one secondary GPU MAY be added under the same rules. |
| Reduced matrix | Base point; N sweep; D sweep; K sweep (PERF-003 rows 1–4); both strategies; both assignment kernels; block size 256; automatic $G$. Block-size, $G$ and tile sweeps are not required. |
| Identical conditions | Same notebook version (git commit or notebook hash recorded), same dataset files (regenerated from the same seeds and verified by hash), same $C^{(0)}$, $I_{\max}$, $\varepsilon$, `--fixed-iters` setting, repetitions and warm-ups. Only `SM_ARCH` changes, set from the recorded compute capability (CUDA-001). |
| Correctness gate | The test suite (§16.1–16.5) and the sanitizer suite are re-run on each secondary GPU before benchmarking; PERF-001 applies per GPU. |
| Separation | Each GPU's rows are written to their own CSV files (`bench_raw_<gpu>.csv`, `bench_summary_<gpu>.csv`). No figure or table mixes measurements from different GPUs in one series without labelling the GPU. All rows of one GPU come from one Colab session where possible; a session change is recorded. |
| Reported metrics | Per GPU: median $T_{\text{alg}}$ for each configuration; ratio $T_{\text{alg}}(B)/T_{\text{alg}}(A)$ and $T_{\text{alg}}(\texttt{smem})/T_{\text{alg}}(\texttt{baseline})$; per-iteration time; numerical agreement between strategies (TEST-I09 statistics). Cross-GPU: the same ratios side by side and a statement of where the ranking changes. |
| Hardware context | For each GPU, record from the vendor specification: architecture, compute capability, SM count, memory bandwidth, FP32 and FP64 peak rates, L2 size. Values are cited, not estimated. |
| Numerical results | Assignments, counts and centroids on the well-separated datasets MUST match across GPUs within NUM-004/NUM-006; Approach B is expected to be bitwise identical across GPUs only if $G$, block sizes and FMA contraction are the same, which is checked and reported, not assumed. |
| Cost control | Paid compute units are optional; if unavailable, OPT-006 is recorded as "not performed" with no effect on completion (§19). |

Verified by TEST-P05.

### 13.1 CSV schema

Raw file `bench_raw.csv` — one row per measured repetition:

| Column | Type | Notes |
|---|---|---|
| run_id | string | unique per row |
| timestamp_utc | ISO-8601 | |
| git_commit | string | |
| dataset_id | string | encodes generator parameters |
| generator_seed | u64 | |
| N, D, K | int | |
| strategy | enum | `atomic` / `reduction` |
| assign_kernel | enum | `baseline` / `smem` |
| accum_precision | enum | `fp64` / `fp32` |
| block_size | int | |
| tile_k, tile_d | int / NA | |
| reduce_chunks_G | int / NA | |
| max_iter, tolerance | int, f64 | |
| fixed_iters | 0/1 | |
| rep_index | int | |
| iterations | int | |
| status | enum | §2.7 |
| centroid_consistent, assignment_consistent | 0/1 | |
| t_alg_ms, t_e2e_ms, t_alloc_ms, t_h2d_ms, t_d2h_ms | f64 | |
| t_assign_ms, t_accum_ms, t_update_ms, t_conv_ms | f64 / NA | phase mode only |
| objective_J | f64 | |
| correctness_status | enum | `PASS` / `FAIL` / `NOT_CHECKED` |
| gpu_name, compute_capability, sm_count, driver_version, cuda_runtime, nvcc_version, host_compiler, os, build_flags | string | |

Summary file `bench_summary.csv`: one row per configuration with mean, median, std, min, max of `t_alg_ms` and `t_e2e_ms`, repetition count and the configuration columns above.

---

## 14. Proposed Implementation Architecture

| ID | Component | Purpose | Inputs | Outputs | Data ownership | Error behavior | Depends on | Verification |
|---|---|---|---|---|---|---|---|---|
| ARC-01 | Config & CLI | Parse options, apply VAL rules | argv | validated `Config` | host | exit 2 on invalid | — | TEST-E01–E05, E21 |
| ARC-02 | Dataset I/O & generator | Load/write binary files; generate blobs | paths, generator params | host arrays | host, owned by caller | exit 5 on I/O, 2 on shape | ARC-01 | TEST-U12, U13 |
| ARC-03 | CPU reference | Oracle per NUM-003; step functions `assign`, `accumulate`, `update`, `run` | host arrays, config | A, C, n, t, status, flags, margins | host | none (pure) | — | hand cases H1–H6 |
| ARC-04 | Device context & buffers | Device query, capacity check, RAII allocations, layout preparation | config, host arrays | device buffers | device, RAII owner | exit 3/4 | ARC-01 | TEST-E26, E27, S01 |
| ARC-05 | Assignment kernels | `assign`, `assign_smem` | $X^{\mathsf{col}}$, $C$, $A$ | $A$, change count | device | launch check | ARC-04 | TEST-U01–U03 |
| ARC-06 | Atomic accumulation | Approach A | $X^{\mathsf{col}}$, $A$ | $S$, $n$ | device | launch check | ARC-04 | TEST-U04 |
| ARC-07 | Reduction accumulation | Approach B Stages 1–2, $G$ selection | $X^{\mathsf{row}}$, $A$ | $S$, $n$ | device | exit 3 if $G_{\text{mem}}<1$ | ARC-04 | TEST-U05–U07 |
| ARC-08 | Update & convergence | `update`, `movement`, `max_reduce`; host decision | $S$, $n$, $C_{\text{cur}}$ | $C_{\text{next}}$, $\Delta$ | device + host | launch check | ARC-06/07 | TEST-U08–U10 |
| ARC-09 | Driver loop & finalization | §2.7, §2.8 orchestration; timing | all above | `Result` | host | propagates errors | ARC-05–08 | TEST-I01–I08 |
| ARC-10 | Verification | P1/P2 checks, lockstep mode, mismatch reports | `Result`, ARC-03 | verification report | host | exit 6 on failure in verify mode | ARC-03, ARC-09 | TEST-I07, I10 |
| ARC-11 | Benchmark harness | Warm-up, repetitions, CSV | config list | CSV files | host | records failures, continues | ARC-09 | TEST-P01–P03 |
| ARC-12 | Test utilities | Comparison with tolerances, poisoning, fault injection hooks | — | — | host | — | — | used by all tests |

Interface contract (illustrative, not code to be copied): the driver exposes one entry point taking validated inputs and a strategy enum and returning a `Result` record with `assignments`, `centroids`, `counts`, `iterations`, `status`, `last_delta`, `centroid_consistent`, `assignment_consistent` and timing fields. The CPU reference exposes the same entry point plus the individual step functions used by lockstep verification.

---

## 15. Build, Execution and Deliverables

### 15.1 Environment and prerequisites

**BLD-001 — Language and toolchain** — CUDA C++ (C++17 host code) in a Jupyter notebook (DEC-011). Toolkit and compiler per ENV-02/03.

**BLD-001a — Toolkit/GPU compatibility** — MUST. CUDA Toolkit 13.x removed offline compilation for Maxwell, Pascal and Volta (compute capability < 7.5). If ENV-01 reports CC 6.x or 7.0, a CUDA 12.x toolkit MUST be used; for CC ≥ 7.5 either 12.x or 13.x is acceptable. The toolkit must also be supported by the installed driver (`nvidia-smi` shows the highest CUDA version the driver supports).

**BLD-001b — Prerequisites, local Windows runtime** — *Not applicable* (ENV-11 = Colab; no NVIDIA GPU on the student's laptop). Retained for reference only if a Windows machine with an NVIDIA GPU is used later.

| # | Component | Purpose | Check |
|---|---|---|---|
| 1 | NVIDIA GPU driver | Runs CUDA programs; provides `nvidia-smi` | `nvidia-smi` |
| 2 | Visual Studio 2022 (Community is sufficient) with the **"Desktop development with C++"** workload, installed **before** the CUDA Toolkit | `nvcc` on Windows requires MSVC `cl.exe` as host compiler; MinGW/g++ is not supported | `cl` in "x64 Native Tools Command Prompt for VS 2022" |
| 3 | CUDA Toolkit (version per BLD-001a) | `nvcc`, CUDA runtime, `compute-sanitizer`; installer also offers Nsight Systems and Nsight Compute | `nvcc --version`, `compute-sanitizer --version` |
| 4 | Python 3.10+ with `notebook` (or `jupyterlab`), `numpy`, `pandas`, `matplotlib` | Notebook runtime, analysis, plots | `python --version`, `jupyter --version` |

Jupyter MUST be started from the "x64 Native Tools Command Prompt for VS 2022" so `cl.exe` is on `PATH` for `nvcc`; alternatively the notebook passes `-ccbin` with the full path to `cl.exe`.

**BLD-001c — Prerequisites, Google Colab runtime** — MUST (selected runtime). Runtime type set to a GPU; `nvcc`, the driver and Python packages are preinstalled. Profilers may be absent or restricted; ENV-08 fallback applies. The notebook records the assigned GPU, because Colab may assign different GPUs across sessions; all benchmark rows of one evaluation MUST come from one GPU model and SHOULD come from one session. Benchmark CSVs, outputs and the notebook MUST be saved to Google Drive (mounted in the notebook) when `USE_DRIVE` is enabled and the mount succeeds, so a runtime disconnect does not lose results; if `USE_DRIVE` is disabled or the mount fails, the notebook MUST emit a warning, fall back to `/content/kmeans`, and record the fallback in the environment record so results that were not persisted are identifiable. The fallback keeps "restart kernel and run all" free of manual steps (NFR-014, DEC-011). `SM_ARCH` is set from the recorded compute capability (e.g. `sm_75` for a Tesla T4), never hard-coded before ENV-01 is recorded.

### 15.2 Compilation

**BLD-002 — Compiler options** — MUST. A release build (`-O3 -lineinfo -arch=sm_XX`, `XX` from ENV-01, `-std=c++17`) and a sanitizer/debug build (`-lineinfo`, `KMEANS_SYNC_CHECK` defined). No `--use_fast_math`. CPU reference compiled with FP contraction disabled (`-Xcompiler -ffp-contract=off` with g++ on Colab). `-Xptxas -v` output is shown in the compile cell output (CUDA-010).

### 15.2a Repository layout

```
gpu-programming/
└── GPU-Based-K-Means-Clustering/
    ├── requirements.md        # this specification
    ├── CLAUDE.md              # rules for the coding assistant
    ├── src/                   # common.h, kmeans.cu, cpu_reference.cpp, gen.cpp, ...
    ├── tests/                 # test drivers, toolchain smoke test, hand datasets H1–H6
    ├── tools/
    │   ├── build_notebook.py  # notebook generator (DEC-011); holds the §15.3 cell manifest
    │   └── cells/             # markdown and Python cell content embedded verbatim by the generator
    ├── kmeans_cuda.ipynb      # generated; final deliverable
    └── results/               # executed notebooks and CSVs saved from Colab
```

### 15.3 Notebook structure

**BLD-003 — Cell structure** — MUST. The notebook is organized in this order, each part a Markdown heading:

1. Title, problem statement, summary of design decisions (§3.1).
2. Environment cell: OS detection, `nvidia-smi`, `nvcc --version`, host compiler version, GPU properties; result stored and printed (ENV record, PERF-008).
3. Configuration cell: Google Drive mount, `SM_ARCH` derived from ENV-01, paths.
4. Toolchain smoke test: a minimal `hello.cu` written by `%%writefile`, compiled with `nvcc -arch=$SM_ARCH`, and run; it prints the device name, compute capability, SM count and the reported CUDA runtime version from host code (CUDA-001) together with a value written by a kernel, proving the recorded `SM_ARCH`, the toolkit and kernel launch all work before any program source is compiled. The cell MUST raise on a non-zero compile or run exit code, or on an unexpected kernel result, so "restart kernel and run all" stops at the first toolchain fault (NFR-014, CUDA-008). No CUDA or driver version is assumed; whatever `nvcc` and the runtime report is printed.
5. Source cells (`%%writefile`): common header, CPU reference, kernels, host driver, dataset generator, test driver.
6. Compile cell(s).
7. Hand-dataset demonstration (H1–H6) with outputs shown.
8. Test suite cell (TEST-B02).
9. Sanitizer cells (TEST-S01–S04).
10. Benchmark cell(s) writing CSVs (§13.1).
11. Analysis cells: tables and plots of time vs. N, D, K for both strategies and both assignment kernels; numerical comparison tables.
12. Written evaluation and limitations.

The executables keep the CLI of BLD-003a so that the same commands work from a notebook cell and a terminal.

**BLD-003a — Executable CLI** — MUST.

```
kmeans --data X.kmd --centroids C0.kmd --k K --max-iter I --tol EPS
       --strategy {atomic|reduction} [--assign {baseline|smem}]
       [--block-size B] [--reduce-chunks G] [--fixed-iters]
       [--verify {none|final|lockstep}] [--profile-phases]
       [--out-prefix PATH] [--reps R --warmup W --csv FILE]
kmeans-gen --n N --d D --k K --sep S --spread SIGMA --seed SEED --out-prefix PATH
```



### 15.4 Formats, codes and messages

**BLD-004 — File format** — MUST. Binary `.kmd`: 4-byte magic `KMD1`, `uint32` version (=1), `uint64` rows, `uint64` cols, then rows×cols binary32 little-endian row-major. Assignment output `.kma`: magic `KMA1`, version, `uint64` N, then N `int32`. Centroid output uses `.kmd`. Run summary: JSON with the fields of FR-015. The notebook reads these files with `numpy.fromfile` for display and plots.

**BLD-005 — Exit codes** — MUST. 0 success (any non-error status); 2 invalid input; 3 unsupported device or insufficient resources; 4 CUDA runtime error; 5 I/O error; 6 verification failure (only with `--verify`). Notebook cells that run executables check the exit code and raise on unexpected values.

**BLD-006 — Error messages** — MUST. Single-line prefix `error[<RULE-ID>]:` followed by the cause and offending value; warnings prefixed `warning[<ID>]:`.

**BLD-007 — Test execution** — MUST. Tests, sanitizer runs and benchmarks are separate notebook sections so the benchmark section can be skipped during development.

**BLD-008 — Documentation** — MUST. The notebook's Markdown cells replace a separate README: environment record, how to run, formats, design decisions, results, limitations.

**BLD-009 — Deliverables** — per §1.5.

---

## 16. Testing Strategy and Acceptance Criteria

### 16.0 Hand-verifiable datasets

| ID | Data | Parameters | Expected outcome |
|---|---|---|---|
| H1 | $D=1$: $X=\{0,1,9,10\}$, $C^{(0)}=\{0,10\}$ | $K=2$, $\varepsilon=0$, $I_{\max}=10$ | $A^{(0)}=\{0,0,1,1\}$; $C^{(1)}=\{0.5,9.5\}$; $A^{(1)}=A^{(0)}$ → `CONVERGED_ASSIGNMENTS`, $t=1$, $n=\{2,2\}$, P2 true |
| H2 | $D=1$: $X=\{0,2,3,10\}$, $C^{(0)}=\{0,3\}$ | $K=2$, $\varepsilon=0$, $I_{\max}=10$ | $A^{(0)}=\{0,1,1,1\}$; $C^{(1)}=\{0,5\}$, $\Delta=2$, $A^{(1)}=\{0,0,1,1\}$; $C^{(2)}=\{1,6.5\}$, $\Delta=1.5$, $A^{(2)}=\{0,0,0,1\}$; $C^{(3)}=\{5/3,10\}$, $\Delta=3.5$, $A^{(3)}=A^{(2)}$ → `CONVERGED_ASSIGNMENTS`, $t=3$, $n=\{3,1\}$, P2 true |
| H2-a | H2 with $\varepsilon=2$ | | `CONVERGED_MOVEMENT`, $t=1$, $A=\{0,0,1,1\}$, $C=\{0,5\}$, P2 false (mean of cluster 0 is 1) |
| H2-b | H2 with $\varepsilon=1.5$ | | `CONVERGED_MOVEMENT`, $t=2$, $A=\{0,0,0,1\}$, $C=\{1,6.5\}$, P2 false |
| H2-c | H2 with $\varepsilon=1.4999$ | | as H2 (`CONVERGED_ASSIGNMENTS`, $t=3$) |
| H2-d | H2 with $I_{\max}=2$ | | `ITERATION_LIMIT`, $t=2$, $A=\{0,0,0,1\}$, $C=\{1,6.5\}$, P2 false |
| H2-e | H2 with $I_{\max}=0$ | | `ITERATION_LIMIT`, $t=0$, $A=\{0,1,1,1\}$, $C=\{0,3\}$, P2 false |
| H3 | $D=1$: $X=\{1\}$, $C^{(0)}=\{0,2\}$ | $K=2$ | tie → $A^{(0)}=\{0\}$; $C^{(1)}=\{1,2\}$ (cluster 1 empty, retained); `CONVERGED_ASSIGNMENTS`, $t=1$, $n=\{1,0\}$ |
| H4 | $D=1$: $X=\{0,1\}$, $C^{(0)}=\{5,5\}$ | $K=2$ | all to cluster 0; $C^{(1)}=\{0.5,5\}$; $t=1$, $n=\{2,0\}$ |
| H5 | $D=1$: $X=\{4\}$, $C^{(0)}=\{0,1,2\}$ | $K=3$ ($K>N$) | warning `W-KGTN`; $A=\{2\}$; $C=\{0,1,4\}$; $t=1$, $n=\{0,0,1\}$ |
| H6 | $D=2$: $X=\{(0,0),(0,1),(10,0),(10,1)\}$, $C^{(0)}=\{(0,0),(10,0)\}$ | $K=2$ | $C^{(1)}=\{(0,0.5),(10,0.5)\}$; $t=1$; `CONVERGED_ASSIGNMENTS` |

All H values except $5/3$ are exactly representable in binary32; $5/3$ is compared with NUM-006.

### 16.1 Unit tests

| ID | Target | Method | Pass criterion |
|---|---|---|---|
| TEST-U01 | Distance + assignment kernel (`assign`, `assign_smem`) | Random data, compare with CPU FP32 path given identical $C$ | Mismatches only on near-ties (NUM-005) |
| TEST-U02 | Tie-breaking | H3; 2-D points on perpendicular bisectors of integer centroids; duplicate centroids | Lowest index chosen exactly |
| TEST-U03 | Change count | Perturb a known assignment, run pass | Count equals CPU count exactly |
| TEST-U04 | Atomic accumulation | Fixed $A$, random $X$ | $n$ exact; $S$ within $10^{-12}$ relative (FP64), centroids within NUM-006 |
| TEST-U05 | Reduction accumulation | As U04 for $G \in \{1, 2, 7, N\}$ and auto | Same criteria; $\sum n_k = N$ |
| TEST-U06 | Reduction initialization | Poison $P$, $Q$, $S$, $n$ before first call; call twice | Results identical to unpoisoned run |
| TEST-U07 | Reduction partial chunks/tiles | $K=1024$, small chunks, $D \in \{1,33,65\}$ | Matches CPU; column sums of $X$ preserved within tolerance |
| TEST-U08 | Movement / convergence | Constructed $C^{(t)}, C^{(t+1)}$ with known $\Delta$ incl. exact boundary | $\Delta$ exact for representable cases; inclusive comparison |
| TEST-U09 | Empty clusters | H3, H4, and a run where a cluster empties mid-iteration | Retained centroid bit-equal; excluded from $\Delta$ |
| TEST-U10 | Update kernel | Given $S$, $n$ | Matches CPU per NUM-006; single-point clusters bit-exact |
| TEST-U11 | Size/overflow calculator | Extreme N, D, K, G | Overflow detected; correct byte counts |
| TEST-U12 | File I/O | Valid, truncated, trailing-bytes, mismatched-header files | Valid loads; others exit 5/2 |
| TEST-U13 | Generator determinism | Generate twice | Identical file hashes |

### 16.2 Integration tests

| ID | Purpose | Pass criterion |
|---|---|---|
| TEST-I01 | Full run, Approach A, H1–H6 | Exact outcomes in §16.0 (5/3 via NUM-006) |
| TEST-I02 | Full run, Approach B, H1–H6 | Same |
| TEST-I03 | Iteration counting and first-iteration convergence | H1 $t=1$; H2 $t=3$ |
| TEST-I04 | Movement criterion and boundary | H2-a, H2-b, H2-c outcomes |
| TEST-I05 | Iteration limit incl. zero | H2-d, H2-e outcomes |
| TEST-I06 | Finalization flags | P2 flag equals CPU-computed value in all H cases |
| TEST-I07 | `--verify final` on random data, both strategies | P1 holds (NUM-005); P2 flag correct |
| TEST-I08 | Output files and summary | Files parse; fields complete; values match in-memory result |
| TEST-I09 | Cross-strategy comparison | Lockstep accumulation A vs. B within NUM-006; report max abs/rel difference |
| TEST-I10 | Lockstep verification on general data | Zero non-near-tie mismatches over all iterations |
| TEST-I11 | Well-separated end-to-end | Exact match with CPU on A, n, t, status; C within NUM-006 |

### 16.3 CPU reference tests

The CPU reference is itself validated on H1–H6 (exact) and by checking that its FP32 and FP64 distance paths agree on all non-near-tie points (TEST-U01 prerequisite). GPU-vs-CPU comparisons are TEST-U01–U10, I07, I09–I11.

### 16.4 Concurrency and memory tests

| ID | Tool | Scope | Pass criterion |
|---|---|---|---|
| TEST-S01 | `compute-sanitizer --tool memcheck --leak-check full` | Small and partial-block configurations, both strategies, both assign kernels, error-path tests | 0 errors, 0 leaks |
| TEST-S02 | `--tool racecheck` | Kernels using shared memory (`assign`, `assign_smem`, `reduce_stage2`, `movement`) | 0 hazards |
| TEST-S03 | `--tool initcheck` | Full runs, both strategies | 0 uninitialized reads |
| TEST-S04 | `--tool synccheck` | All kernels with barriers, partial blocks | 0 errors |
| TEST-S05 | Design review checklist | CUDA-004, CUDA-005, RED-004 arguments checked against code | All items signed off |

`racecheck` detects shared-memory hazards only; global-memory race freedom of Approach B rests on RED-004 and is supported by TEST-R01 (a hidden race would normally show up as run-to-run variation) and TEST-U05.

### 16.5 Reproducibility tests

| ID | Purpose | Pass criterion |
|---|---|---|
| TEST-R01 | Approach B, 10 runs, fixed config | Bitwise identical centroids, assignments, $t$, status |
| TEST-R02 | Approach A, 10 runs | Report: number of runs differing, max centroid difference, any assignment/trajectory differences (no pass threshold beyond NUM-006 per run) |

### 16.6 Performance tests

| ID | Purpose | Pass criterion |
|---|---|---|
| TEST-P01 | Timing fields present and consistent ($T_{\text{alg}} \le T_{\text{e2e}}$) | Holds for all rows |
| TEST-P02 | Fixed-iteration mode | Exactly $I_{\max}$ updates; status `ITERATION_LIMIT` |
| TEST-P03 | Transfer discipline (CUDA-011) | Nsight Systems trace shows only 16-byte copies inside the loop |
| TEST-P04 | Repeatability | Coefficient of variation of `t_alg_ms` reported; repetitions ≥ 10 |
| TEST-P05 | Cross-GPU comparison (optional, PERF-012) | Per-GPU CSVs exist; dataset hashes and notebook version identical across GPUs; test and sanitizer suites passed on each GPU; well-separated results agree across GPUs within NUM-006; no unlabelled mixed-GPU series |

### 16.7 Acceptance test matrix

| Test ID | Requirement IDs | Purpose | Input | Expected result | Method | Pass/fail |
|---|---|---|---|---|---|---|
| TEST-I01 | FR-004–015, ATOM-* | Atomic end-to-end | H1–H6 | §16.0 | automated | exact / NUM-006 |
| TEST-I02 | FR-004–015, RED-* | Reduction end-to-end | H1–H6 | §16.0 | automated | exact / NUM-006 |
| TEST-I03 | DEC-006, FR-013 | Iteration count | H1, H2 | $t=1$, $t=3$ | automated | exact |
| TEST-I04 | DEC-002, FR-012 | Movement boundary | H2-a/b/c | as listed | automated | exact |
| TEST-I05 | FR-013, EDGE-22/24 | Limit | H2-d/e | as listed | automated | exact |
| TEST-I06 | DEC-004, FR-014 | P2 flag | all H | flag correct | automated | exact |
| TEST-I07 | FR-019, NUM-005 | P1/P2 on random data | blobs, both strategies | P1 holds; flag correct | automated | 0 non-near-tie violations |
| TEST-I09 | ATOM-007, RED-009 | A vs. B | base point (small) | within NUM-006 | automated + report | tolerance |
| TEST-I10 | NUM-005, §10.3 | Lockstep | general data | no hard mismatches | automated | 0 hard mismatches |
| TEST-I11 | NUM-004 | Well-separated | blobs sep ≥ 10σ | exact | automated | exact |
| TEST-U01–U13 | FR-005–011, RED-005/006, MEM-006 | Kernel units | §16.1 | §16.1 | automated | §16.1 |
| TEST-E01–E35 | VAL-*, EDGE-* | Corner cases | §9.2 | §9.2 | automated | §9.2 |
| TEST-S01–S05 | NFR-003–006, CUDA-004/005 | Sanitizers, review | §16.4 | 0 errors | tool + review | 0 errors |
| TEST-R01–R02 | NFR-007 | Reproducibility | fixed config | §16.5 | automated | §16.5 |
| TEST-P01–P04 | PERF-* | Benchmark validity | matrix | §16.6 | automated | §16.6 |
| TEST-P05 | PERF-012 (optional) | Cross-GPU comparison | reduced matrix on secondary GPU | §16.6 | automated + report | §16.6 |
| TEST-B01 | NFR-014, DEC-011 | Clean run of notebook | fresh copy, restart and run all | all cells execute, no host warnings | saved notebook outputs | success |
| TEST-B02 | NFR-015 | Test cell | fresh run | exception raised on failure | inject failing test | behaves |

TEST-Exx corresponds one-to-one with EDGE-xx (EDGE-07, 13, 14, 22–25, 31–33, 35 are covered by the tests named in §9.2). No test is reported as passed until its execution log exists.

---

## 17. Requirement Traceability Matrix

| AS-ID | Assignment statement | Requirement IDs | Description | Component | Tests | Verification | Status |
|---|---|---|---|---|---|---|---|
| AS-01 | Partition N points of D dimensions into K clusters using K-means | FR-004–015, §2.7 | Lloyd iteration | ARC-09 | TEST-I01, I02, I11 | automated | Specified |
| AS-02 | Distance $d(x_i,c_k)=\sum_j (x_{ij}-c_{kj})^2$ every iteration | FR-005, NUM-002 | squared Euclidean, FP32, ascending $j$ | ARC-05 | TEST-U01 | automated | Specified |
| AS-03 | Assign each point to the closest centroid | FR-006, FR-007, DEC-005 | argmin with lowest-index ties | ARC-05 | TEST-U01–U03 | automated | Specified |
| AS-04 | Recompute every centroid as the mean of assigned points | FR-008–011, DEC-003 | sums, counts, mean, empty retain | ARC-06/07/08 | TEST-U04–U10 | automated | Specified |
| AS-05 | Continue until assignments stop changing or movement < tolerance | FR-012, FR-013, DEC-002, DEC-006 | OR of criteria, inclusive $\le\varepsilon$ | ARC-08/09 | TEST-I03–I05, U08 | automated | Specified ("smaller than" read as $\le$; ENV-10, DEC-002) |
| AS-06 | Implement using CUDA | CUDA-001–012 | kernels, launches, errors | ARC-04–08 | TEST-S01–S05 | tools + review | Specified |
| AS-07 | Parallelize point-to-centroid distance | DEC-008, CUDA-002, SMEM-001 | thread per point | ARC-05 | TEST-U01, P-matrix | automated + bench | Specified |
| AS-08 | Efficient parallel reduction for centroid coordinates | DEC-001, RED-001–009 | two-stage ownership reduction | ARC-07 | TEST-U05–U07, R01 | automated | Specified |
| AS-09 | Correctly handle concurrent updates to the same centroid | ATOM-003/004/007, RED-004/005 | atomics (A); single-writer ownership (B) | ARC-06/07 | TEST-U04–U06, S02–S05, R01 | tools + review | Specified |
| AS-10 | At least two approaches: atomic and reduction | FR-018, §6, §7 | runtime-selectable strategies | ARC-06/07 | TEST-I01, I02, I09 | automated | Specified |
| AS-11 | Use shared memory to reduce repeated centroid accesses where beneficial | SMEM-001–006 | tiled assignment, measured | ARC-05 | TEST-U01, PERF-003 tiles sweep | automated + bench | Specified |
| AS-12 | Compare execution time for different N, D, K | PERF-001–011 | benchmark matrix, CSV | ARC-11 | TEST-P01–P04 | bench | Specified |
| AS-13 | Compare numerical results for different N, D, K | NUM-004–008, §10.3, TEST-I09 | A vs. B vs. CPU across matrix | ARC-10 | TEST-I07, I09–I11, R01, R02 | automated + report | Specified |
| AS-14 | Inputs: N, D, K, data, initial centroids, max iterations, tolerance | FR-001, FR-002, VAL-001–010, BLD-003/004 | validated CLI and files | ARC-01/02 | TEST-E01–E05, E20, E21, U12 | automated | Specified |
| AS-15 | Outputs: final assignments, final centroids, iteration count, execution time | FR-014, FR-015, FR-017, PERF-005 | files + summary | ARC-09 | TEST-I06, I08, P01 | automated | Specified |

"Status" moves to *Implemented* in Stage 2 and *Verified* in Stage 3 only with linked evidence.

---

## 18. Stage-Gate Workflow

### 18.1 Stage 1 — Specification (this document)

Exit criteria (all MUST hold before Stage 2):
- [x] Every assignment statement mapped (§17).
- [x] Mathematical specification internally consistent (§2, checked in §20).
- [x] Reduction strategy selected and specified (DEC-001, §7).
- [x] Convergence semantics defined (DEC-002).
- [x] Empty-cluster policy selected (DEC-003).
- [x] Final consistency addressed (DEC-004, §2.8).
- [x] Corner cases have dispositions (§9.2).
- [x] Acceptance criteria and tests defined (§16).
- [x] Environment items listed (§1.6).
- [x] Supervisor brief confirmed as the problem statement only; guidance-dependent items closed (ENV-04, 05, 07, 09, 10).
- [x] Runtime selected: Google Colab GPU (ENV-11).
- [ ] Machine-dependent items ENV-01, 02, 03, 06, 08 recorded from a Colab GPU session — **pending** (first task of Stage 2, before any kernel is written).

### 18.2 Stage 2 — Implementation

Order: ARC-01/02 → ARC-03 (CPU reference, validated on H1–H6) → ARC-04 → `assign` baseline → Approach A → update/convergence/finalization → Approach B → `assign_smem` → optional items. Each step passes its unit tests before the next begins. Convergence semantics, empty-cluster behavior, tolerances and finalization MUST NOT change without §18.4.

### 18.3 Stage 3 — Verification and evaluation

Execute §16 and §13; record test logs, sanitizer output, numerical difference reports, benchmark CSVs, environment record, deviations, approved specification changes and known limitations. A stage is complete only when evidence exists, not when files exist.

### 18.4 Change control

A change is recorded in §21 with: requirement IDs affected, reason (evidence), old and new text, impact on tests, and approval date. Tests are updated in the same change.

---

## 19. Definition of Done

The assignment is complete when all of the following are evidenced:

1. All MUST requirements are implemented (traceability status *Implemented*).
2. Approach A and Approach B are both selectable and pass TEST-I01/I02.
3. Concurrent updates are safe: TEST-U04–U06, S02, S05 pass.
4. Synchronization boundaries are correct: TEST-S04, S05 pass.
5. `assign_smem` preserves correctness (SMEM-004) and is benchmarked against `assign`.
6. CPU-reference, edge-case and integration tests executed with logs.
7. Numerical differences within NUM-005/006 or explicitly reported.
8. Final outputs validated per §2.8 (P1 verified, P2 flag correct).
9. Status and iteration counts correct (TEST-I03–I05).
10. Sanitizer suite executed with 0 errors.
11. Benchmarks cover the PERF-003 matrix (or justified exclusions).
12. Both strategies evaluated on identical workloads (PERF-002, PERF-004).
13. Timing methodology and environment documented (PERF-005, PERF-008).
14. Required outputs produced (FR-015).
15. Build and run instructions reproduce on the documented environment (TEST-B01).
16. Deviations, assumptions and limitations documented.
17. Traceability matrix complete with evidence links (status *Verified*).

Optional items, including the cross-GPU comparison (OPT-006, PERF-012), are not part of the Definition of Done. If performed, they must satisfy their own requirements and TEST-P05.

---

## 20. Specification Self-Review (Stage 1)

| Check | Result |
|---|---|
| All 15 assignment statements captured | Yes (§17) |
| Distance, assignment, update, movement definitions agree across §2, §4, §10 | Yes; the same ascending-$j$ FP32 distance and FP64 sums are used everywhere |
| Iteration count convention consistent in §2.7, DEC-006, H-cases | Yes ($t$ = completed updates; H2 $t=3$) |
| Convergence rule identical in algorithm, FR-012, CPU reference (NUM-003), tests (TEST-I03–I05) | Yes |
| Final consistency not assumed | Yes; P2 verified and reported as a flag |
| Approach A and B have distinct contracts | Yes (§6 vs. §7); shared code limited to assignment, update, convergence |
| Race safety: atomics (A), single-writer ownership (B), kernel boundaries | Yes (ATOM-003/007, RED-004, DEC-010) |
| No intra-kernel grid-wide synchronization assumed | Yes (CUDA-004) |
| Shared-memory correctness conditions (barriers, bounds, capacity) | Yes (SMEM-001–003) |
| Every EDGE case has a disposition and test | Yes (§9.2) |
| Floating-point comparison accounts for rounding and near-ties | Yes (NUM-005/006) |
| Benchmarks fair and repeatable; no results fabricated | Yes (§13); no numbers reported |
| No hardware or toolchain invented | Yes; only minimum CC 6.0 required, marked `[ASM]` |
| Stages separate; no CUDA source in this document | Yes |
| Interpretation of "smaller than tolerance" | Closed: $\Delta \le \varepsilon$ (ENV-10) |
| Remaining open points | Machine-dependent facts only (§1.6.2) |

### 20.1 Academic and code-style expectations for Stage 2

- Meaningful names, straightforward control flow, comments only where they explain synchronization, ownership, numerical choices or non-obvious indexing.
- No unnecessary frameworks or abstraction layers; the component list in §14 is the upper bound.
- The student must be able to explain every kernel's ownership and synchronization argument.
- Use of AI assistance is disclosed as required by the course; no claim is made that the work is free of AI involvement, and no attempt is made to evade detection tools.

---

## 21. Change Log

| Version | Date | Change | Approved by |
|---|---|---|---|
| 1.0 | 2026-10-17 | Initial specification (Stage 1 draft) | — |
| 1.1 | 2026-10-17 | Supervisor confirmed problem statement is the full brief. §1.6 split into closed decisions (ENV-04, 05, 07, 09, new ENV-10 on $\le\varepsilon$) and machine-dependent items with recording commands (ENV-01, 02, 03, 06, 08). DEC-002, §17 AS-05, §18.1 and §20 updated accordingly. No change to algorithmic semantics. | Student |
| 1.2 | 2026-10-17 | Final code deliverable changed to a single Jupyter notebook (DEC-011, ENV-04, new ENV-11 runtime choice). §15 rewritten: Windows and Colab prerequisites, CUDA 12.x/13.x compatibility rule, notebook cell structure. NFR-014/015 and TEST-B01/B02 adapted. No change to algorithmic semantics. | Student |
| 1.3 | 2026-10-17 | ENV-11 closed: student laptop has only Intel Iris Xe graphics (no CUDA), so the runtime is Google Colab GPU (Kaggle fallback). Windows prerequisites marked not applicable; recording commands converted to notebook `!` form; Drive persistence and SM_ARCH rule added. No change to algorithmic semantics. | Student |
| 1.4 | 2026-10-17 | Added optional cross-GPU comparison (OPT-006, PERF-012, TEST-P05): T4 primary with full matrix, one secondary Colab GPU (A100 preferred) with reduced matrix, identical conditions, per-GPU CSVs. Not part of Definition of Done. | Student |
| 1.5 | 2026-10-17 | DEC-011 amended: source kept in `src/`, notebook generated by `tools/build_notebook.py`. New DEC-012: GitHub (`sureshbabugandla/gpu-programming`, folder `GPU-Based-K-Means-Clustering/`) + Colab workflow with executed notebooks saved to `results/` as evidence. Repository layout §15.2a added. No change to algorithmic semantics. | Student |
| 1.6 | 2026-10-17 | Project folder renamed to `GPU-Based-K-Means-Clustering` (no spaces, so paths need no quoting in shell cells, GitHub or Colab). | Student |
| 1.7 | 2026-10-17 | M0 decisions recorded. **BLD-001c** (old text: "Benchmark CSVs, outputs and the notebook MUST be saved to Google Drive (mounted in the notebook) so a runtime disconnect does not lose results."): Drive persistence is now conditional — results MUST be saved to Drive when `USE_DRIVE` is enabled and the mount succeeds; otherwise the notebook warns, falls back to `/content/kmeans` and records the fallback in the environment record. Reason: `drive.mount()` requires an interactive authorisation click, which contradicts the manual-step-free "restart and run all" of NFR-014/DEC-011; the fallback resolves the conflict and keeps unpersisted results identifiable. **§15.2a**: `tools/cells/` added to the layout (markdown and Python cell content embedded verbatim by the generator, keeping `build_notebook.py` to manifest and assembly logic only); `tests/` comment extended to name the toolchain smoke test. **§15.3**: new cell part 4, the toolchain smoke test (`hello.cu` compiled with the recorded `SM_ARCH` and run, printing device name, compute capability, SM count, reported CUDA runtime version and a kernel-written value, raising on any failure); former parts 4–11 renumbered 5–12. Reason: validate `SM_ARCH`, toolkit and kernel launch before any program source is compiled, so a toolchain fault is diagnosed in one small cell instead of a multi-file build. Test impact: no test IDs added, changed or withdrawn; part 4 is covered by TEST-B01 (clean run-all) and exercises CUDA-001/CUDA-008 reporting paths. No change to algorithmic semantics. | Student |
