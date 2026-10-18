# Analysis — figures and tables from the measured CSVs. Computes nothing it cannot read from
# bench_raw.csv, bench_phases.csv and bench_numerics.csv (PERF-011).
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if "CSV_DIR" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")
if "RAW_CSV" not in globals():
    raise RuntimeError("Run the benchmark cell (section 10) first: it defines the CSV paths. "
                       "It does not have to measure anything -- BENCH = False is enough.")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

# --- presentation -------------------------------------------------------------------------------
# A categorical palette, fixed per (strategy, kernel) pair rather than per position, so a sweep
# in which one series has no data does not recolour the others. Four hues, validated as a set
# for all pairs under normal vision and the three common colour-vision deficiencies; the marker
# shape repeats the identity, which is what keeps the figures readable where colour is not.
SERIES = {
    ("atomic", "baseline"): ("Approach A - baseline", "#2a78d6", "o"),
    ("atomic", "smem"): ("Approach A - smem", "#eb6834", "s"),
    ("reduction", "baseline"): ("Approach B - baseline", "#1baf7a", "^"),
    ("reduction", "smem"): ("Approach B - smem", "#4a3aa7", "D"),
}
PHASE_COLORS = {"assign": "#2a78d6", "accumulate": "#eb6834", "update": "#1baf7a",
                "convergence": "#4a3aa7", "unattributed": "#9a9a94"}
GRID_KW = dict(color="#d9d8d2", linewidth=0.8)
plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 7,
})


def heading(text):
    print(f"\n{'=' * 104}\n{text}\n{'=' * 104}")


def show(frame, note=None):
    """Print a table under a figure: the figure's accessible reading, not an extra."""
    if frame is None or len(frame) == 0:
        print("  (no measured rows for this table)")
        return
    print(frame.to_string(index=False))
    if note:
        print(f"  {note}")


# --- load ----------------------------------------------------------------------------------------

def load_raw(path):
    # keep_default_na=False: §13.1 writes the literal string NA for a column that describes
    # nothing, and turning it into a missing value would be a different claim.
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, keep_default_na=False)
    return frame if len(frame) else None


def annotate(frame):
    """Attach the requested G and the plan section to each row, from the resume index.

    The §13.1 schema records the G that RED-006 *selected*, not the one that was asked for, so
    without this the G-sweep rows at the base point are indistinguishable from the automatic
    ones and would be averaged into the base point's number.
    """
    frame = frame.copy()
    frame["requested_G"] = "unknown"
    frame["plan_section"] = "unknown"
    if not os.path.exists(PROGRESS_JSON):
        print(f"warning: {PROGRESS_JSON} is missing, so a row's requested G is unknown. Rows are "
              "treated as automatic-G rows, which is wrong if the G sweep was measured.")
        frame["requested_G"] = "auto"
        return frame
    with open(PROGRESS_JSON) as f:
        progress = json.load(f)
    by_run = {}
    for record in progress.values():
        for run_id in record.get("run_ids", []):
            by_run[run_id] = (str(record.get("g", "auto")), record.get("section", "unknown"))
    frame["requested_G"] = frame["run_id"].map(lambda r: by_run.get(r, ("unknown", ""))[0])
    frame["plan_section"] = frame["run_id"].map(lambda r: by_run.get(r, ("", "unknown"))[1])
    return frame


RAW = load_raw(RAW_CSV)
PHASES = load_raw(PHASES_CSV)
NUMERICS = pd.read_csv(NUMERICS_CSV) if os.path.exists(NUMERICS_CSV) else None
EXCLUDED = pd.read_csv(EXCLUDED_CSV) if os.path.exists(EXCLUDED_CSV) else None

if RAW is None:
    print(f"No {RAW_CSV}: the benchmark cell has not written any rows in this or an earlier "
          "session. Set BENCH = True in section 10 and run it; nothing below can be shown "
          "without measurements (PERF-011).")
else:
    RAW = annotate(RAW)

# --- the §16.6 validity gate ----------------------------------------------------------------------

if RAW is not None:
    heading("validity of the measurements (§16.6: TEST-P01, TEST-P02, TEST-P04, PERF-001)")
    gate_failures = []

    violations = RAW[RAW["t_alg_ms"] > RAW["t_e2e_ms"]]
    print(f"TEST-P01  t_alg <= t_e2e: {len(RAW) - len(violations)} of {len(RAW)} rows hold")
    if len(violations):
        gate_failures.append(f"TEST-P01: {len(violations)} row(s) with t_alg > t_e2e")
        show(violations[["run_id", "N", "D", "K", "strategy", "t_alg_ms", "t_e2e_ms"]].head(10))

    fixed = RAW[RAW["fixed_iters"] == 1]
    bad_iters = fixed[(fixed["iterations"] != fixed["max_iter"]) |
                      (fixed["status"] != "ITERATION_LIMIT")]
    print(f"TEST-P02  --fixed-iters rows perform exactly max_iter updates and report "
          f"ITERATION_LIMIT: {len(fixed) - len(bad_iters)} of {len(fixed)} rows hold")
    if len(bad_iters):
        gate_failures.append(f"TEST-P02: {len(bad_iters)} fixed-iteration row(s) disagree")
        show(bad_iters[["run_id", "N", "D", "K", "strategy", "iterations", "max_iter",
                        "status"]].head(10))

    CONFIG_COLS = ["N", "D", "K", "strategy", "assign_kernel", "block_size", "tile_k", "tile_d",
                   "requested_G", "reduce_chunks_G", "max_iter", "fixed_iters",
                   "correctness_status", "p1_check_mode"]
    per_config = RAW.groupby(CONFIG_COLS, dropna=False).agg(
        reps=("t_alg_ms", "count"),
        t_alg_median=("t_alg_ms", "median"), t_alg_mean=("t_alg_ms", "mean"),
        t_alg_std=("t_alg_ms", "std"), t_alg_min=("t_alg_ms", "min"),
        t_alg_max=("t_alg_ms", "max"), t_e2e_median=("t_e2e_ms", "median"),
        iterations=("iterations", "median"), objective_J=("objective_J", "median"),
    ).reset_index()
    per_config["t_alg_cv"] = per_config["t_alg_std"] / per_config["t_alg_mean"]
    per_config["t_alg_per_iter_ms"] = per_config["t_alg_median"] / (per_config["iterations"] + 1)

    short = per_config[per_config["reps"] < 10]
    print(f"TEST-P04  at least 10 measured repetitions: "
          f"{len(per_config) - len(short)} of {len(per_config)} configurations hold; "
          f"coefficient of variation of t_alg_ms: median {per_config['t_alg_cv'].median():.4f}, "
          f"worst {per_config['t_alg_cv'].max():.4f}")
    if len(short):
        gate_failures.append(f"TEST-P04: {len(short)} configuration(s) with fewer than 10 reps")
        show(short[CONFIG_COLS + ["reps"]])

    # PERF-001 (specification v1.9): PASS_SAMPLED is a pass whose P1 half ran on the fixed
    # sample rather than on every point. It counts as verified for the purpose of keeping the
    # row, and the count of such configurations is printed because the distinction is the whole
    # reason the column exists -- a sampled gate must never be read as a full one.
    PASSING = ["PASS", "PASS_SAMPLED"]
    failed = per_config[~per_config["correctness_status"].isin(PASSING)]
    sampled = per_config[per_config["correctness_status"] == "PASS_SAMPLED"]
    print(f"PERF-001  correctness gate: {len(per_config) - len(failed)} of {len(per_config)} "
          f"configurations passed --verify final; of those, {len(sampled)} had P1 checked on "
          f"the sample rather than on every point (P2 is full everywhere)")
    if len(sampled):
        show(sampled[["N", "D", "K", "strategy", "assign_kernel", "p1_check_mode",
                      "correctness_status", "reps"]],
             "P1 for these configurations is evidence about the sampled points only; the "
             "unsampled P1 evidence for the kernels is the §16.1-§16.2 suite.")
    if len(failed):
        print("  the following are excluded from every figure below and from any performance "
              "conclusion:")
        show(failed[CONFIG_COLS + ["reps", "t_alg_median"]])

    print()
    if gate_failures:
        for line in gate_failures:
            print(f"  NOT VALID: {line}")
        print("  The figures below are drawn anyway, but these rows are not benchmark evidence "
              "until the cause is understood.")
    else:
        print("  every validity check holds for the rows in this file")

    # Everything below works on the verified rows only.
    OK = per_config[per_config["correctness_status"].isin(PASSING)].copy()
    DEFAULT_TILE = (OK["tile_k"].astype(str) == "NA") | \
                   ((OK["tile_k"].astype(str) == "8") & (OK["tile_d"].astype(str) == "32"))
    AUTO_G = OK["requested_G"].isin(["auto", "unknown"])
    MATRIX = OK[(OK["fixed_iters"] == 1) & (OK["block_size"] == 256) & DEFAULT_TILE & AUTO_G]


# --- sweeps ----------------------------------------------------------------------------------------

def sweep_frame(axis, held):
    """Rows of one one-factor-at-a-time sweep: `axis` varies, everything in `held` is fixed."""
    frame = MATRIX
    for column, value in held.items():
        frame = frame[frame[column] == value]
    return frame.sort_values([axis, "strategy", "assign_kernel"])


def plot_sweep(frame, axis, title, xlabel):
    """Median T_alg against one factor, one line per (strategy, kernel) pair."""
    if frame is None or len(frame) == 0:
        print(f"  ({title}: not measured in this session)")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for (strategy, kernel), (label, color, marker) in SERIES.items():
        series = frame[(frame["strategy"] == strategy) & (frame["assign_kernel"] == kernel)]
        if not len(series):
            continue
        ax.plot(series[axis], series["t_alg_median"], label=label, color=color, marker=marker,
                markeredgecolor="#ffffff", markeredgewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("median $T_{alg}$  (ms, log scale)")
    ax.set_title(title)
    ax.grid(True, which="major", **GRID_KW)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def ratio_table(frame, axis, numerator, denominator, column, label):
    """One ratio per sweep point, e.g. T_alg(B)/T_alg(A) at the same point and kernel."""
    if frame is None or len(frame) == 0:
        return None
    pivot = frame.pivot_table(index=[axis, "assign_kernel"] if column == "strategy"
                              else [axis, "strategy"],
                              columns=column, values="t_alg_median", aggfunc="median")
    if numerator not in pivot.columns or denominator not in pivot.columns:
        return None
    out = pivot.copy()
    out[label] = pivot[numerator] / pivot[denominator]
    return out.reset_index()


REPORT_COLS = ["N", "D", "K", "strategy", "assign_kernel", "block_size", "reduce_chunks_G",
               "reps", "t_alg_median", "t_alg_mean", "t_alg_std", "t_alg_min", "t_alg_max",
               "t_alg_cv", "t_alg_per_iter_ms", "t_e2e_median", "objective_J"]

if RAW is not None:
    for axis, held, title, xlabel in (
        ("N", {"D": BASE_D, "K": BASE_K},
         f"Scaling in N  (D = {BASE_D}, K = {BASE_K}, fixed {MAX_ITER} iterations)",
         "N  (points, log scale)"),
        ("D", {"N": BASE_N, "K": BASE_K},
         f"Scaling in D  (N = {BASE_N:.0e}, K = {BASE_K}, fixed {MAX_ITER} iterations)",
         "D  (dimensions, log scale)"),
        ("K", {"N": BASE_N, "D": BASE_D},
         f"Scaling in K  (N = {BASE_N:.0e}, D = {BASE_D}, fixed {MAX_ITER} iterations)",
         "K  (clusters, log scale)"),
    ):
        heading(f"{axis} sweep")
        held = {k: v for k, v in held.items() if k != axis}
        frame = sweep_frame(axis, held)
        plot_sweep(frame, axis, title, xlabel)
        show(frame[REPORT_COLS])

        ba = ratio_table(frame, axis, "reduction", "atomic", "strategy", "T_alg(B)/T_alg(A)")
        sb = ratio_table(frame, axis, "smem", "baseline", "assign_kernel",
                         "T_alg(smem)/T_alg(baseline)")
        print(f"\n  ratio of the two strategies at each {axis}:")
        show(ba)
        print(f"\n  ratio of the two assignment kernels at each {axis}:")
        show(sb)

    # --- block size, G and tiles
    heading("block size (CUDA-007), reduction chunks G (RED-006) and the SMEM-001 tiles")
    base = ((OK["N"] == BASE_N) & (OK["D"] == BASE_D) & (OK["K"] == BASE_K) &
            (OK["fixed_iters"] == 1))
    blocks = OK[base & DEFAULT_TILE & AUTO_G].sort_values(["block_size", "strategy",
                                                           "assign_kernel"])
    print("block size at the base point:")
    show(blocks[REPORT_COLS])

    g_rows = OK[base & DEFAULT_TILE & (OK["strategy"] == "reduction")].sort_values(
        ["requested_G", "assign_kernel"])
    print("\nreduction chunks G at the base point (requested vs. selected, RED-006/RED-007):")
    show(g_rows[["requested_G", "reduce_chunks_G", "assign_kernel", "reps", "t_alg_median",
                 "t_alg_cv", "t_alg_per_iter_ms"]])

    tiles = OK[(OK["assign_kernel"] == "smem") & (OK["fixed_iters"] == 1) &
               (OK["block_size"] == 256)].sort_values(["N", "D", "K", "tile_k", "tile_d",
                                                       "strategy"])
    print("\nassign_smem tile sizes (each pair is its own build, SMEM-001):")
    show(tiles[["N", "D", "K", "strategy", "tile_k", "tile_d", "reps", "t_alg_median",
                "t_alg_cv", "t_alg_per_iter_ms"]])

    # --- the small full grid
    heading("small full grid (PERF-003 row 8)")
    grid_n, grid_d, grid_k = ({n for n, _, _ in GRID_SWEEP}, {d for _, d, _ in GRID_SWEEP},
                              {k for _, _, k in GRID_SWEEP})
    grid = OK[(OK["fixed_iters"] == 1) & (OK["assign_kernel"] == "baseline") & AUTO_G &
              OK["N"].isin(grid_n) & OK["D"].isin(grid_d) &
              OK["K"].isin(grid_k)].sort_values(["N", "D", "K", "strategy"])
    show(grid[REPORT_COLS])
    if len(grid):
        print("\n  T_alg(B)/T_alg(A) over the grid:")
        wide = grid.pivot_table(index=["N", "D", "K"], columns="strategy",
                                values="t_alg_median", aggfunc="median")
        if {"atomic", "reduction"} <= set(wide.columns):
            wide["T_alg(B)/T_alg(A)"] = wide["reduction"] / wide["atomic"]
        show(wide.reset_index())

    # --- PERF-004: per-iteration time and the converging runs
    heading("PERF-004: per-iteration time, and the converging runs reported separately")
    print("fixed-iteration matrix, T_alg / (t + 1) per configuration (ms):")
    show(MATRIX.sort_values(["N", "D", "K", "strategy", "assign_kernel"])[
        ["N", "D", "K", "strategy", "assign_kernel", "iterations", "t_alg_median",
         "t_alg_per_iter_ms"]])
    converging = OK[OK["fixed_iters"] == 0].sort_values(["N", "D", "K", "strategy",
                                                         "assign_kernel"])
    print("\nconverging runs (no --fixed-iters), with the iteration count each one reached:")
    show(converging[["N", "D", "K", "strategy", "assign_kernel", "iterations", "reps",
                     "t_alg_median", "t_alg_cv", "t_alg_per_iter_ms", "objective_J"]],
         "iterations is the median over the repetitions; a strategy that converges in fewer "
         "iterations does less work, which is why PERF-004 compares the fixed-iteration runs "
         "and reports these separately.")

# --- PERF-007 phase breakdown -----------------------------------------------------------------------

heading("PERF-007: where the time goes inside T_alg")
if PHASES is None:
    print("No bench_phases.csv: the phase-timing pass was not run in this or an earlier session.")
else:
    phase_cfg = PHASES.groupby(["N", "D", "K", "strategy", "assign_kernel"], dropna=False).agg(
        reps=("t_alg_ms", "count"),
        t_alg=("t_alg_ms", "median"), assign=("t_assign_ms", "median"),
        accumulate=("t_accum_ms", "median"), update=("t_update_ms", "median"),
        convergence=("t_conv_ms", "median"),
    ).reset_index()
    phase_cfg["unattributed"] = (phase_cfg["t_alg"] - phase_cfg[["assign", "accumulate",
                                                                 "update", "convergence"]]
                                 .sum(axis=1))
    for column in ("assign", "accumulate", "update", "convergence", "unattributed"):
        phase_cfg[column + "_pct"] = 100.0 * phase_cfg[column] / phase_cfg["t_alg"]
    show(phase_cfg)
    print("  the extra events perturb T_alg, so these totals are not comparable with the "
          "headline numbers above (PERF-007); 'unattributed' is the per-iteration 16-byte "
          "transfer and the host decision gap, left as measured rather than distributed.")

    labels = [f"N={int(r.N):.0e} D={int(r.D)} K={int(r.K)}\n{r.strategy}/{r.assign_kernel}"
              for r in phase_cfg.itertuples()]
    fig, ax = plt.subplots(figsize=(9.0, 0.42 * len(phase_cfg) + 2.0))
    left = np.zeros(len(phase_cfg))
    for column in ("assign", "accumulate", "update", "convergence", "unattributed"):
        ax.barh(labels, phase_cfg[column], left=left, color=PHASE_COLORS[column], label=column,
                height=0.62, edgecolor="#ffffff", linewidth=2.0)
        left = left + phase_cfg[column].to_numpy()
    ax.set_xlabel("milliseconds, summed over the run")
    ax.set_title("PERF-007 phase breakdown of $T_{alg}$ (phase-timing runs, perturbed totals)")
    ax.grid(True, axis="x", **GRID_KW)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    plt.show()

# --- TEST-I09 ------------------------------------------------------------------------------------------

heading("TEST-I09: Approach A against Approach B on identical inputs")
if NUMERICS is None:
    print("No bench_numerics.csv: the comparison was not run in this or an earlier session.")
else:
    show(NUMERICS[["N", "D", "K", "max_iter", "assignment_disagreements",
                   "assignment_disagreement_fraction", "counts_equal", "max_abs_dC",
                   "objective_atomic", "objective_reduction", "objective_rel_difference",
                   "correctness_atomic", "correctness_reduction"]],
         "Both strategies ran on the same file, the same C0 and the same 20 fixed iterations "
         "(PERF-002, FR-020). A non-zero disagreement is not by itself an error: once the two "
         "disagree on one point their trajectories separate and the remaining iterations are "
         "computed on different partitions (EDGE-32). The per-iteration comparison on identical "
         "A(t), which is the strict form of ATOM-007/RED-009, is TEST-I09 in test_edge.")

# --- what was not measured ---------------------------------------------------------------------------

heading("configurations that were NOT measured (PERF-011)")
if EXCLUDED is None or not len(EXCLUDED):
    print("bench_excluded.csv is absent or empty: no configuration of the attempted sections was "
          "skipped. Sections left out of BENCH_SECTIONS are simply not in the CSVs.")
else:
    show(EXCLUDED)
