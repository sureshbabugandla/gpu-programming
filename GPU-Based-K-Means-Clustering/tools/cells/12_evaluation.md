## 12. Written evaluation and limitations

Skeleton only. Every heading below is to be written by hand from the measured tables that the
next cell prints; no number is written into this markdown and no conclusion is drawn here.
PERF-010 is explicit that no claim about either strategy being faster may be made without data,
and PERF-011 that only measured values are reported — so a heading whose measurement is missing
is answered with "not measured", not with an expectation.

### 12.1 What was measured

_To be written: the GPU and toolchain of the session, which sections of the PERF-003 matrix were
run, how many configurations and repetitions, and which were excluded and why. The next cell
prints the environment record, the configuration count and the exclusion table._

### 12.2 Approach A against Approach B

_To be written, from the $T_{\text{alg}}(B)/T_{\text{alg}}(A)$ table. PERF-010 gives the model to
explain the observation against: Approach A performs $ND$ binary64 atomics onto $KD$ addresses
with contention $\approx N/K$ per address; Approach B performs $ND$ reads, $ND$ binary64
read-modify-writes to private partials, $GKD$ zeroing and $GKD$ combine reads, and depends on the
device's FP64 throughput (ENV-06)._

### 12.3 The baseline assignment kernel against `assign_smem`

_To be written, from the $T_{\text{alg}}(\texttt{smem})/T_{\text{alg}}(\texttt{baseline})$ table
and the tile sweep. SMEM-005 asks whether the shared-memory tiling pays, and the register and
shared-memory usage printed by the compile cell (CUDA-010) is part of the answer._

### 12.4 Scaling in $N$, $D$ and $K$

_To be written, from the three sweep figures and their tables. The assignment phase is
$\approx 3NKD$ flops over $4ND$ bytes of point reads per pass and is compute-bound for large $K$
(PERF-010); say whether the measured slopes agree with that._

### 12.5 Block size, reduction chunks $G$ and the tile sizes

_To be written, from the block-size, $G$ and tile tables. RED-006 picks $G$ automatically; the
sweep says whether that choice was a good one._

### 12.6 Where the time goes (PERF-007)

_To be written, from the phase breakdown. State explicitly that the phase-timing runs carry extra
events and are not the headline totals, and what the unattributed remainder consists of._

### 12.7 Numerical agreement between the strategies

_To be written, from the TEST-I09 table and the §16.5 reproducibility output: how far apart the
two strategies ended, whether the difference is consistent with EDGE-32 trajectory divergence,
and what TEST-R01 and TEST-R02 showed about run-to-run reproducibility (bitwise for Approach B;
reported, with both $\max|dC|$ and $\max|dS|$, for Approach A)._

### 12.8 Threats to validity

_To be written. Candidates: a shared Colab GPU and the resulting run-to-run variation (the
reported coefficient of variation is the evidence either way); a single GPU model, so nothing
here separates an architecture effect from an implementation effect (PERF-012 is the optional
extension that would); the dataset family is one generator (FR-003 Gaussian blobs), so $K_{true}
= K$ in every point of the matrix; and the gate, the warm-ups and the repetition count._

### 12.9 What was not measured

_To be written, from the exclusion table and from the sections left out of `BENCH_SECTIONS`.
PERF-009 (Nsight Systems and Nsight Compute evidence) and PERF-012 (a second GPU) are SHOULD and
MAY; record them as performed or not performed, with no effect on completion (§19)._
