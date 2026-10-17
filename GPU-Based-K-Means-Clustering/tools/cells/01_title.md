# GPU-Based K-Means Clustering Using CUDA — Problem 5

Lloyd's K-Means on an NVIDIA GPU in CUDA C++: given $N$ points in $\mathbb{R}^D$ and $K$ initial
centroids, alternate between assigning every point to its nearest centroid by squared Euclidean
distance and recomputing every centroid as the mean of its assigned points, until assignments stop
changing or the largest centroid movement is $\le \varepsilon$, or until the iteration limit is
reached. The program reports final assignments, final centroids, the iteration count and the
execution time.

The two centroid-accumulation strategies required by the assignment — **Approach A (global
atomics)** and **Approach B (deterministic two-stage reduction)** — are both implemented and
selectable at run time, and the assignment kernel exists in a baseline and a shared-memory variant.

> **This notebook is generated.** `tools/build_notebook.py` assembles it from the files under
> `src/`, `tests/` and `tools/cells/`, and its `%%writefile` cells are checked to be byte-identical
> to those files (DEC-011). Never edit the notebook by hand: change the repository files and
> regenerate, or the next regeneration silently discards the edit.

## Design decisions (requirements.md §3.1)

| Decision | Selected policy |
|---|---|
| Centroid accumulation (A) | Element-wise global `atomicAdd` on binary64 sums and `uint32` counts (§6) |
| Centroid accumulation (B) | Two-stage ownership-based reduction: chunk-privatised partial sums, then a fixed-order tree combine (DEC-001) — race-free without atomics, and deterministic |
| Convergence | Assignment stability **or** $\Delta \le \varepsilon$ (Euclidean, inclusive); stability takes reporting precedence (DEC-002) |
| Empty clusters | Retain the previous centroid; excluded from $\Delta$ (DEC-003) |
| Final consistency | Report the last $(A^{(t)}, C^{(t)})$ pair; P1 holds by construction, P2 verified and reported as a flag (DEC-004) |
| Tie-breaking | Lowest centroid index on exact binary32 equality, via strict-less-than replacement in ascending $k$ (DEC-005) |
| Iteration counting | Number of completed centroid updates (DEC-006) |
| Numeric types | binary32 points, centroids and distances; binary64 centroid sums, movement and tolerance; `uint32` counts (DEC-007) |
| Assignment mapping | One thread per point, grid-stride, ascending-$j$ distance accumulation (DEC-008) |
| Device layout | Two read-only copies of $X$: dimension-major for assignment and Approach A, row-major for Approach B Stage 1 (DEC-009) |
| Global synchronisation | Kernel boundaries on a single stream only; no cooperative-groups grid sync (DEC-010) |
| Deliverable | This single notebook, generated from `src/` (DEC-011) |
| Workflow | Edit and test locally, push to GitHub, run on a Colab GPU, save the executed notebook under `results/` (DEC-012) |

## How to run

1. **Runtime → Change runtime type → GPU** (ENV-11). The configuration cell fails fast with no GPU.
2. Run the cells top to bottom. The environment cell records the machine (§1.6.2), the
   configuration cell derives `SM_ARCH` from the recorded compute capability, and the smoke test
   proves the toolchain before any program source is compiled.
3. Save the executed notebook with **File → Save a copy in GitHub** to
   `GPU-Based-K-Means-Clustering/results/kmeans_cuda_<YYYYMMDD-HHMM>.ipynb` (DEC-012). GPU results
   count as evidence only from an executed notebook committed there.
