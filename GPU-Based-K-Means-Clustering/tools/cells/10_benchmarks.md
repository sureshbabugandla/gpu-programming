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

* **PERF-001** — `--verify final` runs the P1/P2 check once, after the repetitions, on the host.
  It costs no device time and stamps every row of the configuration `correctness_status=PASS` or
  `FAIL`. A `FAIL` configuration keeps its rows and is excluded from the analysis, never deleted.
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

**Cost, and where it goes.** The plan print below reports, per section, the work the PERF-001
gate implies: `verify_final` is a single-threaded host pass of $N\,K\,D$ binary32 distance
evaluations per configuration. For most of the matrix that is small beside the measured run; for
the tile sweep at $D = 512$, $K = 1024$ it is the dominant cost of the whole section. If a
session is short, give that section one of its own — `BENCH_SECTIONS = ("tiles",)` — and let the
resume index join the results to the rest.
