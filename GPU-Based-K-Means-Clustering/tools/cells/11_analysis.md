## 11. Analysis — figures and tables from the measured CSVs

Everything below is computed from `bench_raw.csv`, `bench_phases.csv` and `bench_numerics.csv`
in `CSV_DIR`. The cell does not measure anything, so it runs with `BENCH = False` against the
CSVs of an earlier session; if a file is missing it says so and skips that part rather than
filling the gap (PERF-011).

**The validity gate runs first (§16.6).** A figure drawn from rows that failed the gate would be
a performance claim about a run that was not verified, so the gate comes before the first plot:

| Check | Criterion |
|---|---|
| TEST-P01 | $T_{\text{alg}} \le T_{\text{e2e}}$ in every row |
| TEST-P02 | every `--fixed-iters` row performed exactly $I_{\max}$ updates and reports `ITERATION_LIMIT` |
| TEST-P04 | at least 10 measured repetitions per configuration, with the coefficient of variation of `t_alg_ms` reported |
| PERF-001 | rows whose `correctness_status` is not `PASS` are named and then dropped from every figure |

**What is plotted.** Median $T_{\text{alg}}$ against $N$, $D$ and $K$ on log–log axes, four series
per figure — Approach A and Approach B, each with the baseline and the shared-memory assignment
kernel. Each figure is followed by the table it was drawn from, which is also the accessible
reading of it. Then the two ratio tables PERF-012 names, $T_{\text{alg}}(B)/T_{\text{alg}}(A)$
and $T_{\text{alg}}(\texttt{smem})/T_{\text{alg}}(\texttt{baseline})$; the block-size, $G$ and
tile tables; per-iteration time $T_{\text{alg}}/(t+1)$ for the fixed-iteration matrix and for the
converging runs with their iteration counts (PERF-004); the PERF-007 phase breakdown with the
unattributed remainder shown rather than distributed; and the TEST-I09 table comparing the two
strategies' final assignments, counts, centroids and objective.

The figures identify a series by colour **and** marker shape, and every figure is paired with its
table, so nothing here depends on colour alone. No figure mixes two y-scales, and the colour of a
series is fixed to the (strategy, kernel) pair it denotes, so a sweep in which one series is
missing does not repaint the others.

**This cell draws no conclusion.** Section 12 is where the interpretation goes, and the numbers
there come from these tables.
