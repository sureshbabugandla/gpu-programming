# Benchmark harness — §13: the PERF-003 matrix, the PERF-001 gate, PERF-004 fixed-iteration
# comparisons, PERF-006 repetitions, PERF-007 phase timing and the §13.1 CSVs.
import json
import os
import shutil
import struct
import subprocess
import time

import numpy as np
import pandas as pd

if "CSV_DIR" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

# BLD-007: the benchmark section is skippable, and it is skipped unless this is switched on.
# Deliberately independent of QUICK: a QUICK = False run that only wants the §16 test evidence
# should not pay for the matrix, and a benchmark session should not be forced to re-run it.
BENCH = False

# Which parts of the matrix to measure. Narrow this to spread the matrix over several Colab
# sessions; the resume index below makes the parts additive.
BENCH_SECTIONS = ("base", "N", "D", "K", "block", "G", "tiles", "grid", "converge", "phases",
                  "numerics")

# --- the PERF-003 base point and the sweeps around it -------------------------------------------
BASE_N, BASE_D, BASE_K = 1_000_000, 32, 64
BASE_BLOCK = 256
N_SWEEP = (10_000, 100_000, 1_000_000, 10_000_000)
D_SWEEP = (2, 8, 32, 128, 512)
K_SWEEP = (2, 8, 64, 256, 1024)
BLOCK_SWEEP = (128, 256, 512)
TILE_SWEEP = tuple((tk, td) for tk in (4, 8, 16) for td in (16, 32))
TILE_POINTS = ((BASE_N, BASE_D, BASE_K), (BASE_N, 512, 1024))
GRID_SWEEP = tuple((n, d, k) for n in (10_000, 1_000_000) for d in (4, 64) for k in (4, 256))
PHASE_POINTS = ((BASE_N, BASE_D, BASE_K), (10_000_000, BASE_D, BASE_K),
                (BASE_N, 512, BASE_K), (BASE_N, BASE_D, 1024))
NUMERIC_POINTS = ((10_000, BASE_D, BASE_K), (BASE_N, BASE_D, BASE_K),
                  (10_000_000, BASE_D, BASE_K), (BASE_N, 2, BASE_K), (BASE_N, 512, BASE_K),
                  (BASE_N, BASE_D, 2), (BASE_N, BASE_D, 1024))

VARIANTS = (("atomic", "baseline"), ("atomic", "smem"),
            ("reduction", "baseline"), ("reduction", "smem"))

# PERF-003: Gaussian blobs (FR-003) with K_true = K and a fixed seed per configuration.
GEN_SEP, GEN_SPREAD = 10.0, 1.0

# PERF-006 and PERF-004. TOL is in data units (§2.6: Delta is an L2 distance between successive
# centroids) and only has an effect in the converging section, since --fixed-iters disables both
# convergence tests; it is passed everywhere so every run of the matrix shares one epsilon
# (PERF-002).
REPS, WARMUP, MAX_ITER, TOL = 10, 3, 20, 1e-3

# KMEANS_TILE_K / KMEANS_TILE_D defaults of launch_config.h: the `kmeans` binary built in
# section 6 is the (8, 32) build, so the tile sweep only has to compile the other five.
DEFAULT_TILES = (8, 32)

# A configuration that has not finished in this long is recorded as an exclusion rather than
# allowed to hold the session open.
PER_CONFIG_TIMEOUT_S = 7200

RAW_CSV = os.path.join(CSV_DIR, "bench_raw.csv")
SUMMARY_CSV = os.path.join(CSV_DIR, "bench_summary.csv")
PHASES_CSV = os.path.join(CSV_DIR, "bench_phases.csv")
NUMERICS_CSV = os.path.join(CSV_DIR, "bench_numerics.csv")
EXCLUDED_CSV = os.path.join(CSV_DIR, "bench_excluded.csv")
PROGRESS_JSON = os.path.join(CSV_DIR, "bench_progress.json")
BENCH_DATA_DIR = os.path.join(DATA_DIR, "bench")
BENCH_OUT_DIR = os.path.join(OUT_DIR, "bench")


# --- the plan -----------------------------------------------------------------------------------

def config(section, N, D, K, strategy, assign, block=BASE_BLOCK, g="auto", tiles=DEFAULT_TILES,
           fixed=True, phases=False):
    return {"section": section, "N": N, "D": D, "K": K, "strategy": strategy, "assign": assign,
            "block": block, "g": g, "tile_k": tiles[0], "tile_d": tiles[1], "fixed": fixed,
            "phases": phases}


def config_key(c):
    """Identity of a configuration, and the resume key.

    The requested G is part of it and cannot be recovered from a CSV row: `auto`, a quarter of
    auto and four times auto are three different requests that may select the same G, and a row
    only records what was selected (RED-006, RED-007).
    """
    return (f"n{c['N']}_d{c['D']}_k{c['K']}|{c['strategy']}|{c['assign']}|b{c['block']}"
            f"|g{c['g']}|tk{c['tile_k']}td{c['tile_d']}|i{MAX_ITER}"
            f"|{'fixed' if c['fixed'] else 'converge'}|{'phases' if c['phases'] else 'plain'}")


def build_plan():
    """The PERF-003 matrix, in run order, with the base point measured once.

    Sweeps are one-factor-at-a-time around the base point and therefore overlap in it; a
    duplicate would be a second set of rows for an identical configuration, which would weight
    that point twice in every aggregate. The first occurrence wins and later sweeps reuse it.
    """
    plan, seen = [], set()

    def add(c):
        if c["section"] not in BENCH_SECTIONS:
            return
        key = config_key(c)
        if key in seen:
            return
        seen.add(key)
        plan.append(c)

    for strategy, assign in VARIANTS:
        add(config("base", BASE_N, BASE_D, BASE_K, strategy, assign))
    for n in N_SWEEP:
        for strategy, assign in VARIANTS:
            add(config("N", n, BASE_D, BASE_K, strategy, assign))
    for d in D_SWEEP:
        for strategy, assign in VARIANTS:
            add(config("D", BASE_N, d, BASE_K, strategy, assign))
    for k in K_SWEEP:
        for strategy, assign in VARIANTS:
            add(config("K", BASE_N, BASE_D, k, strategy, assign))
    for block in BLOCK_SWEEP:
        for strategy, assign in VARIANTS:
            add(config("block", BASE_N, BASE_D, BASE_K, strategy, assign, block=block))
    # The G sweep is Approach B only: Approach A allocates no partials (RED-006). The two
    # non-auto values are filled in once auto G is known for the base point.
    for g in ("auto", "quarter", "four"):
        for assign in ("baseline", "smem"):
            add(config("G", BASE_N, BASE_D, BASE_K, "reduction", assign, g=g))
    # SMEM-001 makes the tiles compile-time constants, so each pair is a separate binary; only
    # assign_smem reads them, so the baseline kernel is not in this sweep.
    for n, d, k in TILE_POINTS:
        for tiles in TILE_SWEEP:
            for strategy in ("atomic", "reduction"):
                add(config("tiles", n, d, k, strategy, "smem", tiles=tiles))
    # PERF-003 small grid: strategies only, at the default kernel and block size.
    for n, d, k in GRID_SWEEP:
        for strategy in ("atomic", "reduction"):
            add(config("grid", n, d, k, strategy, "baseline"))
    # PERF-004: the converging counterpart of the base point, reported separately with its
    # iteration counts.
    for strategy, assign in VARIANTS:
        add(config("converge", BASE_N, BASE_D, BASE_K, strategy, assign, fixed=False))
    # PERF-007: phase timing is its own pass into its own file.
    for n, d, k in PHASE_POINTS:
        for strategy, assign in VARIANTS:
            add(config("phases", n, d, k, strategy, assign, phases=True))
    return plan


PLAN = build_plan()


# --- resources: device memory (MEM-002) and disk -------------------------------------------------

def device_memory_required(N, D, K, G):
    """MEM-002: 8ND + 4N + 16KD + 4K + [8GKD + 4GK] + the movement scratch.

    The same model `device_memory_bytes` (common.h) evaluates inside the program, repeated here
    so a configuration that cannot fit is named before it is launched rather than after it exits
    with code 3.
    """
    total = 8 * N * D + 4 * N + 16 * K * D + 4 * K + 8 * 1024 + 32
    if G:
        total += 8 * G * K * D + 4 * G * K
    return total


def free_device_bytes():
    """Free device memory now, from nvidia-smi (ENV-01). Falls back to the total if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60)
        return int(out.stdout.split()[0]) * 1024 * 1024
    except Exception:
        return 0


def free_disk_bytes(path):
    return shutil.disk_usage(path).free


def capacity_excuse(c, g_value):
    """None when the configuration fits, otherwise the PERF-011 reason it was not run."""
    need = device_memory_required(c["N"], c["D"], c["K"], g_value)
    free = free_device_bytes()
    budget = int(free * 0.95)  # the budget check_capacity applies (device_context.cuh)
    if free and need > budget:
        return (f"needs {need / 2**30:.2f} GiB of device memory, usable budget is "
                f"{budget / 2**30:.2f} GiB (95 % of {free / 2**30:.2f} GiB free)")
    return None


# --- datasets (FR-003) ---------------------------------------------------------------------------

def dataset_seed(N, D, K):
    """A fixed seed per configuration (PERF-003), derived from the shape so it is reproducible
    from the notebook alone and stable across sessions."""
    h = 1469598103934665603
    for value in (N, D, K):
        for byte in str(value).encode() + b"|":
            h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h % 1_000_000_007


def dataset_paths(N, D, K):
    seed = dataset_seed(N, D, K)
    prefix = os.path.join(BENCH_DATA_DIR, f"blob_n{N}_d{D}_k{K}_s{seed}")
    return prefix, prefix + "_X.kmd", prefix + "_C0.kmd"


def ensure_dataset(N, D, K, needed_later):
    """Generate the dataset if it is not on disk. Returns (x_path, c_path, None) or (None, None,
    reason). Datasets no longer needed by the remaining plan are deleted when disk runs short;
    they are regenerable from their seeds, so disk is not worth defending at the cost of a
    configuration."""
    prefix, x_path, c_path = dataset_paths(N, D, K)
    if os.path.exists(x_path) and os.path.exists(c_path):
        return x_path, c_path, None

    need = 4 * N * D + 4 * K * D + 1024
    if free_disk_bytes(BENCH_DATA_DIR) < need * 1.1:
        for stale_prefix in list(GENERATED):
            shape = GENERATED[stale_prefix]
            if shape in needed_later:
                continue
            for suffix in ("_X.kmd", "_C0.kmd"):
                try:
                    os.remove(stale_prefix + suffix)
                except OSError:
                    pass
            del GENERATED[stale_prefix]
            print(f"    freed {os.path.basename(stale_prefix)}: no later configuration needs it")
            if free_disk_bytes(BENCH_DATA_DIR) >= need * 1.1:
                break
    if free_disk_bytes(BENCH_DATA_DIR) < need * 1.1:
        return None, None, (f"the {need / 2**30:.2f} GiB dataset does not fit in the "
                            f"{free_disk_bytes(BENCH_DATA_DIR) / 2**30:.2f} GiB of free disk")

    seed = dataset_seed(N, D, K)
    proc = subprocess.run(["./kmeans-gen", "--n", str(N), "--d", str(D), "--k", str(K),
                           "--sep", str(GEN_SEP), "--spread", str(GEN_SPREAD), "--seed", str(seed),
                           "--out-prefix", prefix], capture_output=True, text=True)
    if proc.returncode != 0:
        return None, None, f"kmeans-gen exited {proc.returncode}: {(proc.stdout + proc.stderr).strip()[:300]}"
    GENERATED[prefix] = (N, D, K)
    return x_path, c_path, None


# --- the resume index -----------------------------------------------------------------------------

def load_progress():
    if os.path.exists(PROGRESS_JSON):
        try:
            with open(PROGRESS_JSON) as f:
                return json.load(f)
        except Exception as exc:
            print(f"warning: {PROGRESS_JSON} is unreadable ({exc!r}); starting a fresh index. "
                  "Configurations already in the CSVs will be measured again.")
    return {}


def save_progress(progress):
    with open(PROGRESS_JSON, "w") as f:
        json.dump(progress, f, indent=1, sort_keys=True)


def csv_run_ids(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, keep_default_na=False)["run_id"].astype(str))


def purge_rows(path, run_ids):
    """Drop rows of a half-finished configuration so a re-measurement does not double-count it."""
    frame = pd.read_csv(path, keep_default_na=False)
    keep = frame[~frame["run_id"].astype(str).isin(run_ids)]
    keep.to_csv(path, index=False)
    print(f"    purged {len(frame) - len(keep)} stale row(s) of this configuration")


# --- running one configuration -------------------------------------------------------------------

def binary_for(tiles):
    """The executable built with these compile-time tiles (SMEM-001)."""
    return "./kmeans" if tuple(tiles) == DEFAULT_TILES else f"./kmeans_tk{tiles[0]}_td{tiles[1]}"


def build_tile_binaries(plan):
    """Compile one executable per tile pair the plan needs, with the section 6 compile helper."""
    wanted = {(c["tile_k"], c["tile_d"]) for c in plan} - {DEFAULT_TILES}
    if not wanted:
        return
    if "compile_cuda" not in globals():
        raise RuntimeError("The tile sweep needs the CUDA compile cell (section 6) to have run: "
                           "compile_cuda is not defined.")
    for tiles in sorted(wanted):
        target = binary_for(tiles)[len("./"):]
        if os.path.exists(target):
            continue
        print(f"\nbuilding {target} (SMEM-001: T_K and T_D are compile-time constants)")
        compile_cuda(target, ["kmeans_main.cu"] + CUDA_SOURCES + HOST_SOURCES,
                     extra_flags=[f"-DKMEANS_TILE_K={tiles[0]}", f"-DKMEANS_TILE_D={tiles[1]}"])


def auto_chunks(N, D, K, x_path, c_path):
    """The G that RED-006 selects for this shape, read from a run that does no iterations.

    --max-iter 0 is legal (VAL-004) and still allocates the partial buffers and prints the
    selection, so this costs one finalization pass and tells the G sweep what a quarter and four
    times auto are.
    """
    proc = subprocess.run(["./kmeans", "--data", x_path, "--centroids", c_path, "--k", str(K),
                           "--max-iter", "0", "--tol", str(TOL), "--strategy", "reduction"],
                          capture_output=True, text=True, timeout=PER_CONFIG_TIMEOUT_S)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith("reduction: G = "):
            return int(line.split("=", 1)[1].split("(")[0].strip())
    return None


def resolve_chunks(c, auto_g):
    """The --reduce-chunks value for this configuration, or None for automatic selection."""
    if c["strategy"] != "reduction" or c["g"] == "auto":
        return None
    if auto_g is None:
        return None
    return max(1, auto_g // 4) if c["g"] == "quarter" else auto_g * 4


def run_one(c, auto_g, needed_later, progress, excluded):
    """Measure one configuration. Appends rows to the raw or phase CSV, or an exclusion."""
    key = config_key(c)
    csv_path = PHASES_CSV if c["phases"] else RAW_CSV
    label = (f"{c['section']:9s} N={c['N']:<9d} D={c['D']:<4d} K={c['K']:<5d} "
             f"{c['strategy']:9s} {c['assign']:8s} block={c['block']:<4d} g={c['g']:<7s} "
             f"tiles={c['tile_k']}x{c['tile_d']} {'fixed' if c['fixed'] else 'converge'}"
             f"{' phases' if c['phases'] else ''}")

    # Resume: the index is authoritative about which rows belong to this configuration.
    done = progress.get(key)
    if done:
        present = csv_run_ids(csv_path)
        recorded = set(done.get("run_ids", []))
        if recorded and recorded <= present:
            print(f"  skip  {label}  [{len(recorded)} rows already measured]")
            return
        if recorded & present:
            purge_rows(csv_path, recorded & present)

    g_value = resolve_chunks(c, auto_g)
    if c["strategy"] == "reduction" and c["g"] != "auto" and g_value is None:
        excluded.append({**c, "reason": "the automatic G for this point is unknown, so a "
                                        "quarter of it and four times it are undefined"})
        print(f"  SKIP  {label}  [auto G unknown]")
        return

    # Approach A allocates no partials; for Approach B use the requested G, falling back to
    # the automatic one and then to 1, which is the smallest G RED-006 can select.
    g_for_memory = 0
    if c["strategy"] == "reduction":
        g_for_memory = g_value or auto_g or 1
    excuse = capacity_excuse(c, g_for_memory)
    if excuse:
        excluded.append({**c, "reason": excuse})
        print(f"  SKIP  {label}\n        {excuse}")
        return

    x_path, c_path, reason = ensure_dataset(c["N"], c["D"], c["K"], needed_later)
    if reason:
        excluded.append({**c, "reason": reason})
        print(f"  SKIP  {label}\n        {reason}")
        return

    argv = [binary_for((c["tile_k"], c["tile_d"])), "--data", x_path, "--centroids", c_path,
            "--k", str(c["K"]), "--max-iter", str(MAX_ITER), "--tol", str(TOL),
            "--strategy", c["strategy"], "--assign", c["assign"],
            "--block-size", str(c["block"]),
            "--warmup", str(WARMUP), "--reps", str(REPS),
            "--verify", "final",          # PERF-001: the gate, once, after the repetitions
            "--csv", csv_path]
    if c["fixed"]:
        argv.append("--fixed-iters")      # PERF-004
    if c["phases"]:
        argv.append("--profile-phases")   # PERF-007
    if g_value is not None:
        argv += ["--reduce-chunks", str(g_value)]

    before = csv_run_ids(csv_path)
    started = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=PER_CONFIG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        excluded.append({**c, "reason": f"did not finish within {PER_CONFIG_TIMEOUT_S} s"})
        print(f"  SKIP  {label}\n        timed out after {PER_CONFIG_TIMEOUT_S} s")
        return
    elapsed = time.time() - started
    output = (proc.stdout + proc.stderr).strip()

    # Exit 6 is a verification failure (BLD-005). Its rows are written and carry
    # correctness_status=FAIL, which is exactly what PERF-001 asks for: recorded, and excluded
    # from the performance conclusions by the analysis cell rather than deleted here.
    if proc.returncode not in (0, 6):
        excluded.append({**c, "reason": f"exit code {proc.returncode}: "
                                        f"{diagnostic(output)}"})
        print(f"  FAIL  {label}\n        exit {proc.returncode}: {diagnostic(output)}")
        return

    new_ids = sorted(csv_run_ids(csv_path) - before)
    if len(new_ids) != REPS:
        print(f"  warn  {label}\n        {len(new_ids)} new row(s), expected {REPS}")
    # The requested G is recorded next to the rows because the §13.1 schema keeps only the G
    # that was selected; the analysis cell needs the request to tell a G-sweep row from an
    # automatic one at the same point.
    progress[key] = {"run_ids": new_ids, "csv": os.path.basename(csv_path),
                     "section": c["section"], "g": c["g"], "block": c["block"],
                     "tiles": [c["tile_k"], c["tile_d"]], "seconds": round(elapsed, 1)}
    save_progress(progress)

    median = next((line for line in output.splitlines() if line.startswith("t_alg_ms")), "")
    verify = next((line for line in output.splitlines() if line.startswith("verify final")), "")
    print(f"  ok    {label}  [{elapsed:.1f} s]")
    if median:
        print(f"        {median}")
    if proc.returncode == 6:
        print(f"        correctness_status=FAIL (PERF-001): {verify or diagnostic(output)}")


def diagnostic(output):
    """The error line of a failed run (BLD-006), or the last line of output."""
    for line in output.splitlines():
        if line.startswith("error["):
            return line
    lines = output.splitlines()
    return lines[-1] if lines else "(no output)"


# --- TEST-I09 at benchmark scale: Approach A against Approach B ------------------------------------

def read_kmd_file(path):
    with open(path, "rb") as f:
        if f.read(4) != b"KMD1":
            raise RuntimeError(f"{path}: not a BLD-004 .kmd file")
        struct.unpack("<I", f.read(4))
        rows, cols = struct.unpack("<QQ", f.read(16))
        payload = np.fromfile(f, dtype="<f4")
    return payload.reshape(int(rows), int(cols))


def read_kma_file(path):
    with open(path, "rb") as f:
        if f.read(4) != b"KMA1":
            raise RuntimeError(f"{path}: not a BLD-004 .kma file")
        struct.unpack("<I", f.read(4))
        struct.unpack("<Q", f.read(8))
        return np.fromfile(f, dtype="<i4")


def compare_strategies(N, D, K, needed_later, excluded):
    """One row of the TEST-I09 table: both strategies on the same data, same C0, same I_max.

    A difference here is not automatically an error. Once the two strategies disagree on one
    assignment their trajectories separate and everything after that is computed on different
    partitions (EDGE-32, permitted); what the table reports is how far apart they ended, judged
    against the NUM-006 scale, with the per-iteration comparison on identical A(t) left to
    TEST-I09 in test_edge.
    """
    x_path, c_path, reason = ensure_dataset(N, D, K, needed_later)
    if reason:
        excluded.append({"section": "numerics", "N": N, "D": D, "K": K, "reason": reason})
        return None

    results = {}
    for strategy in ("atomic", "reduction"):
        prefix = os.path.join(BENCH_OUT_DIR, f"i09_n{N}_d{D}_k{K}_{strategy}")
        proc = subprocess.run(["./kmeans", "--data", x_path, "--centroids", c_path, "--k", str(K),
                               "--max-iter", str(MAX_ITER), "--tol", str(TOL),
                               "--strategy", strategy, "--fixed-iters", "--verify", "final",
                               "--out-prefix", prefix],
                              capture_output=True, text=True, timeout=PER_CONFIG_TIMEOUT_S)
        if proc.returncode not in (0, 6):
            excluded.append({"section": "numerics", "N": N, "D": D, "K": K,
                             "strategy": strategy,
                             "reason": f"exit {proc.returncode}: "
                                       f"{diagnostic(proc.stdout + proc.stderr)}"})
            return None
        with open(prefix + "_summary.json") as f:
            summary = json.load(f)
        results[strategy] = {"A": read_kma_file(prefix + "_A.kma"),
                             "C": read_kmd_file(prefix + "_C.kmd"),
                             "counts": np.asarray(summary["counts"], dtype=np.int64),
                             "J": summary["objective"],
                             "correctness": summary["correctness_status"]}

    a, b = results["atomic"], results["reduction"]
    dC = np.abs(a["C"].astype(np.float64) - b["C"].astype(np.float64))
    disagree = int(np.count_nonzero(a["A"] != b["A"]))
    return {"N": N, "D": D, "K": K, "max_iter": MAX_ITER, "fixed_iters": 1,
            "assignment_disagreements": disagree,
            "assignment_disagreement_fraction": disagree / float(N),
            "counts_equal": int(np.array_equal(a["counts"], b["counts"])),
            "max_abs_dC": float(dC.max()),
            "objective_atomic": a["J"], "objective_reduction": b["J"],
            "objective_rel_difference": abs(a["J"] - b["J"]) / max(abs(a["J"]), 1e-300),
            "correctness_atomic": a["correctness"], "correctness_reduction": b["correctness"]}


# --- §13.1 summary ----------------------------------------------------------------------------------

# `status` is not a grouping column: §2.7 permits Approach A to stop a step earlier than a
# repeat of itself (ATOM-007), and splitting one configuration into two summary rows over that
# would hide it. It is aggregated instead, with the number of distinct values beside it so a
# split shows up as distinct_statuses > 1. `correctness_status` *is* a key: PERF-001 forbids
# averaging a FAIL run together with a PASS one.
SUMMARY_GROUP = ["dataset_id", "N", "D", "K", "strategy", "assign_kernel", "accum_precision",
                 "block_size", "tile_k", "tile_d", "reduce_chunks_G", "max_iter", "tolerance",
                 "fixed_iters", "correctness_status", "gpu_name", "compute_capability",
                 "sm_count", "driver_version", "cuda_runtime", "nvcc_version", "host_compiler",
                 "os", "build_flags"]


def write_summary(raw_path=RAW_CSV, summary_path=SUMMARY_CSV):
    """§13.1: one row per configuration, with the PERF-006 statistics over its repetitions."""
    if not os.path.exists(raw_path):
        print(f"no {os.path.basename(raw_path)} yet: nothing to summarise")
        return None
    raw = pd.read_csv(raw_path, keep_default_na=False)
    stats = raw.groupby(SUMMARY_GROUP, dropna=False).agg(
        reps=("t_alg_ms", "count"),
        status=("status", "first"),
        distinct_statuses=("status", "nunique"),
        t_alg_mean=("t_alg_ms", "mean"), t_alg_median=("t_alg_ms", "median"),
        t_alg_std=("t_alg_ms", "std"), t_alg_min=("t_alg_ms", "min"),
        t_alg_max=("t_alg_ms", "max"),
        t_e2e_mean=("t_e2e_ms", "mean"), t_e2e_median=("t_e2e_ms", "median"),
        t_e2e_std=("t_e2e_ms", "std"), t_e2e_min=("t_e2e_ms", "min"),
        t_e2e_max=("t_e2e_ms", "max"),
        iterations=("iterations", "median"),
        objective_J=("objective_J", "median"),
    ).reset_index()
    # TEST-P04 reports the coefficient of variation; it belongs next to the statistics it is
    # computed from rather than only in the analysis cell.
    stats["t_alg_cv"] = stats["t_alg_std"] / stats["t_alg_mean"]
    stats["t_alg_per_iteration_ms"] = stats["t_alg_median"] / (stats["iterations"] + 1)
    stats.to_csv(summary_path, index=False)
    print(f"{summary_path}: {len(stats)} configuration(s) from {len(raw)} raw row(s)")
    return stats


# --- the plan, printed whether or not it is run ------------------------------------------------------

GENERATED = {}
for _prefix, _shape in [(dataset_paths(n, d, k)[0], (n, d, k))
                        for n, d, k in {(c["N"], c["D"], c["K"]) for c in PLAN}]:
    if os.path.exists(_prefix + "_X.kmd"):
        GENERATED[_prefix] = _shape

print(f"PERF-003 plan: {len(PLAN)} configuration(s) x {REPS} measured repetitions "
      f"({WARMUP} warm-ups each), I_max = {MAX_ITER}, tol = {TOL}")
# The PERF-001 gate is a host pass of N*K*D binary32 distance evaluations per configuration
# (verify_final in kmeans_gpu.cu), single-threaded, and at the top of the matrix it costs more
# than the measured run itself. Printing the work per section is what makes the cost visible
# before a session is spent on it; a heavy section can be given a session of its own through
# BENCH_SECTIONS.
for section in BENCH_SECTIONS:
    members = [c for c in PLAN if c["section"] == section]
    if members:
        shapes = sorted({(c["N"], c["D"], c["K"]) for c in members})
        gate = sum(c["N"] * c["K"] * c["D"] for c in members)
        print(f"  {section:9s} {len(members):3d} configuration(s) over {len(shapes)} dataset(s), "
              f"PERF-001 gate work {gate:.3e} point-centroid distances")
_dataset_bytes = sum(4 * n * d + 4 * k * d for n, d, k in {(c["N"], c["D"], c["K"]) for c in PLAN})
print(f"  datasets   {_dataset_bytes / 2**30:.2f} GiB on local disk at {BENCH_DATA_DIR}")
print(f"  CSVs       {CSV_DIR}")

if not BENCH:
    print("\nBENCH = False: nothing was measured. Set BENCH = True in this cell and re-run it "
          "to measure the matrix; the section-11 analysis cell reads the CSVs this would write "
          "and works from a previous session's results.")
else:
    os.makedirs(BENCH_DATA_DIR, exist_ok=True)
    os.makedirs(BENCH_OUT_DIR, exist_ok=True)
    progress = load_progress()
    if not progress and os.path.exists(RAW_CSV) and len(csv_run_ids(RAW_CSV)):
        print(f"warning: {RAW_CSV} holds rows but the resume index is empty, so every "
              "configuration will be measured again and the file will hold two sets of rows for "
              "each. Move the old CSVs aside first if that is not what you want.")
    excluded = []
    started_all = time.time()

    build_tile_binaries(PLAN)

    # The G sweep needs the automatic G of the base point before it can ask for a quarter of it.
    auto_g = None
    if any(c["section"] == "G" for c in PLAN):
        x_path, c_path, reason = ensure_dataset(BASE_N, BASE_D, BASE_K,
                                                {(c["N"], c["D"], c["K"]) for c in PLAN})
        if reason:
            print(f"warning: the G sweep has no dataset: {reason}")
        else:
            auto_g = auto_chunks(BASE_N, BASE_D, BASE_K, x_path, c_path)
            print(f"\nRED-006 selects G = {auto_g} at the base point; the sweep also measures "
                  f"{max(1, (auto_g or 4) // 4)} and {(auto_g or 1) * 4}")

    print(f"\n{'=' * 110}\nmeasuring\n{'=' * 110}")
    for index, c in enumerate(PLAN):
        needed_later = {(later["N"], later["D"], later["K"]) for later in PLAN[index + 1:]}
        run_one(c, auto_g, needed_later, progress, excluded)

    # --- TEST-I09 at benchmark scale
    numeric_rows = []
    if "numerics" in BENCH_SECTIONS:
        print(f"\n{'=' * 110}\nTEST-I09: Approach A against Approach B on identical inputs\n"
              f"{'=' * 110}")
        existing = (pd.read_csv(NUMERICS_CSV) if os.path.exists(NUMERICS_CSV) else pd.DataFrame())
        for n, d, k in NUMERIC_POINTS:
            if len(existing) and ((existing["N"] == n) & (existing["D"] == d) &
                                  (existing["K"] == k)).any():
                print(f"  skip  N={n} D={d} K={k}  [already compared]")
                continue
            row = compare_strategies(n, d, k, set(), excluded)
            if row is None:
                continue
            numeric_rows.append(row)
            print(f"  ok    N={n:<9d} D={d:<4d} K={k:<5d} "
                  f"assignment disagreements {row['assignment_disagreements']} "
                  f"({row['assignment_disagreement_fraction']:.3e} of N), "
                  f"max |dC| {row['max_abs_dC']:.3e}, counts "
                  f"{'equal' if row['counts_equal'] else 'DIFFER'}")
        if numeric_rows:
            frame = pd.DataFrame(numeric_rows)
            if len(existing):
                frame = pd.concat([existing, frame], ignore_index=True)
            frame.to_csv(NUMERICS_CSV, index=False)
            print(f"{NUMERICS_CSV}: {len(frame)} comparison(s)")

    # --- §13.1 summaries and the PERF-011 exclusion record
    print(f"\n{'=' * 110}\nsummaries\n{'=' * 110}")
    write_summary(RAW_CSV, SUMMARY_CSV)
    if os.path.exists(PHASES_CSV):
        write_summary(PHASES_CSV, os.path.join(CSV_DIR, "bench_phases_summary.csv"))

    if excluded:
        pd.DataFrame(excluded).to_csv(EXCLUDED_CSV, index=False)
        print(f"\n{len(excluded)} configuration(s) NOT run (PERF-011), written to "
              f"{EXCLUDED_CSV}:")
        for row in excluded:
            print(f"  N={row.get('N')} D={row.get('D')} K={row.get('K')} "
                  f"{row.get('strategy', '')} {row.get('assign', '')}: {row['reason']}")
    else:
        print("\nno configuration was excluded: every configuration in the plan was measured.")

    print(f"\ntotal benchmark time: {(time.time() - started_all) / 60:.1f} min")
