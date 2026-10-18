# Section 12 support: the measured tables that belong under each heading above, printed from the
# CSVs so the written evaluation can be written against numbers rather than recollection. This
# cell draws no conclusion and invents nothing (PERF-010, PERF-011).
import os

import pandas as pd

if "RAW_CSV" not in globals():
    raise RuntimeError("Run the benchmark cell (section 10) first: it defines the CSV paths.")

print("Tables for section 12. Each block is headed by the subsection it belongs to; the text of "
      "that subsection is written by hand from these numbers.\n")


def block(title):
    print(f"\n{'-' * 100}\n{title}\n{'-' * 100}")


def table(frame, columns=None, note=None):
    if frame is None or not len(frame):
        print("  not measured")
        if note:
            print(f"  {note}")
        return
    print((frame[columns] if columns else frame).to_string(index=False))
    if note:
        print(f"  {note}")


# PERF-001 (v1.9): a sampled gate is still a pass, and is kept, but it is never silently
# merged with a full one -- the p1_check_mode column travels with every table below.
PASSING = ["PASS", "PASS_SAMPLED"]

raw = pd.read_csv(RAW_CSV, keep_default_na=False) if os.path.exists(RAW_CSV) else None
summary = pd.read_csv(SUMMARY_CSV) if os.path.exists(SUMMARY_CSV) else None
phases = pd.read_csv(PHASES_CSV, keep_default_na=False) if os.path.exists(PHASES_CSV) else None
numerics = pd.read_csv(NUMERICS_CSV) if os.path.exists(NUMERICS_CSV) else None
excluded = pd.read_csv(EXCLUDED_CSV) if os.path.exists(EXCLUDED_CSV) else None

# --- 12.1 ----------------------------------------------------------------------------------------
block("12.1  what was measured")
if raw is None or not len(raw):
    print("  no bench_raw.csv: nothing was measured in this or an earlier session")
else:
    env_cols = ["gpu_name", "compute_capability", "sm_count", "driver_version", "cuda_runtime",
                "nvcc_version", "host_compiler", "os", "build_flags", "git_commit"]
    print("PERF-001 gate modes over the rows (v1.9: P2 always full, P1 full or sampled):")
    table(raw.groupby(["correctness_status", "p1_check_mode"]).size()
          .reset_index(name="rows"))
    print("\nPERF-008 environment of the rows in bench_raw.csv (one line per distinct record):")
    table(raw[env_cols].drop_duplicates())
    print(f"\nrows: {len(raw)};  configurations: "
          f"{len(summary) if summary is not None else 'unknown'};  "
          f"repetitions per configuration: {sorted(raw.groupby('run_id').size().unique())[0]} row "
          f"per run_id by construction")
    print("\nconfigurations per shape:")
    table(raw.groupby(["N", "D", "K"]).agg(
        rows=("run_id", "count"),
        configurations=("run_id", lambda s: s.str.rsplit("-r", n=1).str[0].nunique()),
    ).reset_index())

# --- 12.2 and 12.3 --------------------------------------------------------------------------------
block("12.2  Approach A against Approach B, and 12.3  baseline against assign_smem")
if summary is None:
    print("  no bench_summary.csv")
else:
    keyed = summary[(summary["fixed_iters"] == 1) &
                    (summary["correctness_status"].isin(PASSING))].copy()
    pivot_strategy = keyed.pivot_table(index=["N", "D", "K", "assign_kernel", "block_size"],
                                       columns="strategy", values="t_alg_median",
                                       aggfunc="median")
    if {"atomic", "reduction"} <= set(pivot_strategy.columns):
        pivot_strategy["T_alg(B)/T_alg(A)"] = (pivot_strategy["reduction"] /
                                               pivot_strategy["atomic"])
    print("median T_alg (ms) per strategy, and their ratio:")
    table(pivot_strategy.reset_index())

    pivot_kernel = keyed.pivot_table(index=["N", "D", "K", "strategy", "block_size"],
                                     columns="assign_kernel", values="t_alg_median",
                                     aggfunc="median")
    if {"baseline", "smem"} <= set(pivot_kernel.columns):
        pivot_kernel["T_alg(smem)/T_alg(baseline)"] = (pivot_kernel["smem"] /
                                                       pivot_kernel["baseline"])
    print("\nmedian T_alg (ms) per assignment kernel, and their ratio (SMEM-005):")
    table(pivot_kernel.reset_index())

# --- 12.4 -----------------------------------------------------------------------------------------
block("12.4  scaling in N, D and K")
if summary is None:
    print("  no bench_summary.csv")
else:
    sweeps = summary[(summary["fixed_iters"] == 1) &
                     (summary["correctness_status"].isin(PASSING)) &
                     (summary["block_size"] == 256)]
    table(sweeps.sort_values(["D", "K", "N", "strategy", "assign_kernel"]),
          ["N", "D", "K", "strategy", "assign_kernel", "reduce_chunks_G", "reps",
           "t_alg_median", "t_alg_std", "t_alg_cv", "t_alg_per_iteration_ms", "t_e2e_median",
           "p1_check_mode"],
          "t_alg_per_iteration_ms is T_alg / (t + 1) (PERF-004).")

# --- 12.5 -----------------------------------------------------------------------------------------
block("12.5  block size, reduction chunks G, tile sizes")
if summary is None:
    print("  no bench_summary.csv")
else:
    ok = summary[summary["correctness_status"].isin(PASSING)]
    print("block size:")
    table(ok[ok["block_size"] != 256].sort_values(["N", "D", "K", "block_size", "strategy"]),
          ["N", "D", "K", "strategy", "assign_kernel", "block_size", "t_alg_median", "t_alg_cv"],
          "the 256 rows are the base point above (CUDA-007 default).")
    print("\nreduction chunks G as selected (RED-006; the requested value is in "
          "bench_progress.json):")
    table(ok[ok["strategy"] == "reduction"].sort_values(["N", "D", "K", "reduce_chunks_G"]),
          ["N", "D", "K", "assign_kernel", "reduce_chunks_G", "t_alg_median", "t_alg_cv"])
    print("\ntile sizes (SMEM-001, one build per pair):")
    table(ok[ok["assign_kernel"] == "smem"].sort_values(["N", "D", "K", "tile_k", "tile_d"]),
          ["N", "D", "K", "strategy", "tile_k", "tile_d", "t_alg_median", "t_alg_cv"])

# --- 12.6 -----------------------------------------------------------------------------------------
block("12.6  phase breakdown (PERF-007)")
if phases is None or not len(phases):
    print("  no bench_phases.csv: the phase pass was not run")
else:
    grouped = phases.groupby(["N", "D", "K", "strategy", "assign_kernel"]).agg(
        reps=("t_alg_ms", "count"), t_alg=("t_alg_ms", "median"),
        assign=("t_assign_ms", "median"), accumulate=("t_accum_ms", "median"),
        update=("t_update_ms", "median"), convergence=("t_conv_ms", "median"),
    ).reset_index()
    grouped["unattributed"] = grouped["t_alg"] - grouped[["assign", "accumulate", "update",
                                                          "convergence"]].sum(axis=1)
    table(grouped, note="perturbed totals, not comparable with the headline numbers (PERF-007).")

# --- 12.7 -----------------------------------------------------------------------------------------
block("12.7  numerical agreement between the strategies (TEST-I09)")
table(numerics, note="TEST-R01 and TEST-R02 are in the section 8 test-suite output, not here: "
                     "they are tests, and their numbers are read from that log.")

# --- 12.9 -----------------------------------------------------------------------------------------
block("12.9  what was not measured (PERF-011)")
table(excluded, note="sections left out of BENCH_SECTIONS do not appear here at all; name them "
                     "in 12.9 from the plan printed by the benchmark cell.")
