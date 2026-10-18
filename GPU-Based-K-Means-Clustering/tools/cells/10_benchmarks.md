## 10. Benchmarks (§13: PERF-001 … PERF-008, PERF-011)

This section measures the PERF-003 matrix and writes the §13.1 CSVs. It is **off by default**:
set `BENCH = True` in the cell below and re-run it. With `BENCH = False` the cell prints the
plan — every configuration it would run and every one it would exclude — and measures nothing,
so a development pass over the notebook stays fast (BLD-007).

**One command per configuration.** Each configuration is run once as

```
./kmeans --data …_X.kmd --centroids …_C0.kmd --k K --max-iter 20 --tol TOL \
         --strategy {atomic|reduction} --assign {baseline|smem} [--block-size B] \
         [--reduce-chunks G] --fixed-iters --verify final --warmup 3 --reps 10 --csv bench_raw.csv
```

which satisfies three requirements at once:

* **PERF-001** (specification v1.9) — `--verify final` runs the P1/P2 check once, after the
  repetitions, on the host. It costs no device time and stamps every row of the configuration
  `correctness_status=PASS`, `PASS_SAMPLED`, `FAIL` or `FAIL_SAMPLED`, with the detail in the
  `p1_check_mode` column. P2, the centroid–mean check, is always performed in full. P1 is
  performed in full when $NKD \le 2\times10^{10}$ and on a fixed seeded stratified sample of
  100 000 points above that, because the full P1 is a single-threaded host pass of $NKD$
  binary32 distance evaluations and at the top of the matrix it costs more than the run it
  guards. A sampled gate is evidence about the sampled points only; the unsampled P1 evidence
  for the kernels is the §16.1–§16.2 suite in section 8, which runs at sizes where the full
  check is affordable. A `FAIL` configuration keeps its rows and is excluded from the analysis,
  never deleted.
* **PERF-004** — `--fixed-iters` with $I_{\max} = 20$, so Approach A and Approach B perform the
  same number of updates and the comparison is about the kernels, not about one strategy stopping
  a step earlier. The converging runs are a separate section with their own iteration counts, and
  per-iteration time $T_{\text{alg}}/(t+1)$ is derived in section 11.
* **PERF-006** — 3 warm-up runs that are not recorded, then 10 measured repetitions in the same
  process on data that stays loaded; mean, median, standard deviation, min and max are computed
  from the raw rows in `bench_summary.csv`.

**PERF-002** holds by construction: the variants of one sweep point share the dataset file, the
same $C^{(0)}$ (the generator writes it once), the same $I_{\max}$, $\varepsilon$ and build.

**Files written to `CSV_DIR`** (Drive, BLD-001c):

| File | Contents |
|---|---|
| `bench_raw.csv` | §13.1 raw schema, one row per measured repetition |
| `bench_summary.csv` | one row per configuration: count, mean, median, std, min, max of `t_alg_ms` and `t_e2e_ms` |
| `bench_phases.csv` | the PERF-007 `--profile-phases` rows, in the same schema, kept apart because the extra events perturb the totals |
| `bench_numerics.csv` | the TEST-I09 comparison of Approach A against Approach B at benchmark scale |
| `bench_excluded.csv` | every configuration that was *not* run, with the reason (PERF-011) |
| `bench_progress.json` | the resume index: which configurations are already measured and which rows are theirs |

**Resume.** Colab disconnects. Before running a configuration the cell looks it up in
`bench_progress.json` and skips it when every `run_id` recorded there is still present in
`bench_raw.csv`; a partial match purges those rows and re-measures. The index is needed because
the requested $G$ (`auto`, ¼×, 4×) cannot be recovered from the CSV — two requests can select the
same $G$. `BENCH_SECTIONS` narrows the plan further, so the matrix can be spread over several
sessions.

**Sizing (PERF-003 "sizes are adjusted to the confirmed GPU memory").** Before each run the cell
evaluates the MEM-002 model $8ND + 4N + 16KD + 4K + [8GKD + 4GK]$ and compares it against 95 % of
the free device memory — the same budget `check_capacity` applies — and checks free disk for the
$4ND$ dataset file. Anything over budget is skipped and listed in the exclusion table with its
required bytes and the reason. The list is computed from what this session actually has, not
predicted here, and nothing is reported as measured that was not (PERF-011).

**Tile sizes are checked before they are timed.** `assign_smem` is a template on $T_K$ and
$T_D$, so every pair in the tile sweep is a different kernel instantiation with a different
overhang pattern. The section 8 suite establishes SMEM-004 for the default $8\times32$ build
only; this cell builds a `test_gpu` for each other pair and runs `--only smem` — the hand cases,
the dataset sweep and the TEST-E16 non-aligned size matrix — before any configuration of that
build is measured. A pair that fails is blocked and all of its configurations appear in the
exclusion table, because a timing of a kernel that disagrees with the baseline measures the
wrong thing (SMEM-004 before SMEM-005).

**Estimated run time.** The plan print below gives, per section, how many configurations are not
yet measured and roughly how long they will take, so a section can be given a session of its own
through `BENCH_SECTIONS` when the whole matrix does not fit one. The estimate is a model — the
assignment pass at a nominal rate, plus the PERF-001 gate at a nominal host rate — and it says
so; once `bench_progress.json` holds three or more completed configurations the model is
calibrated against their measured wall times and the print says what it was calibrated on. It is
an estimate for planning, never a measurement (PERF-011).
