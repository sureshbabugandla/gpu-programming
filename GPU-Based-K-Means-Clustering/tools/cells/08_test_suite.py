# The full test suite (TEST-B02): §16.1-§16.5. Raises on the first failure (NFR-015).
import json
import os
import struct
import subprocess

import numpy as np
import pandas as pd

if "QUICK" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

SUITE_DIR = os.path.join(DATA_DIR, "suite")
os.makedirs(SUITE_DIR, exist_ok=True)
failures = []


def run(cmd, env_extra=None, expect=0, label=None):
    """Run a command and return (exit code, combined output). Records a failure on a surprise."""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = proc.stdout + proc.stderr
    if expect is not None and proc.returncode != expect:
        failures.append(f"{label or ' '.join(cmd)}: exit code {proc.returncode}, expected "
                        f"{expect}\n{output.strip()[:2000]}")
    return proc.returncode, output


def check(condition, what):
    if not condition:
        failures.append(what)


# --- BLD-004 writers, so the malformed inputs below are malformed in exactly one way -----------

def write_kmd(path, rows, cols, values):
    """Binary .kmd per BLD-004: magic, uint32 version, uint64 rows, uint64 cols, float32 payload.

    rows and cols are written as given rather than derived from `values`, which is what lets the
    EDGE-01 and EDGE-02 cases declare zero rows or zero columns in the header.
    """
    with open(path, "wb") as f:
        f.write(b"KMD1")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", rows))
        f.write(struct.pack("<Q", cols))
        np.asarray(values, dtype="<f4").tofile(f)


def read_kmd(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"KMD1":
            raise RuntimeError(f"{path}: magic is {magic!r}, expected b'KMD1' (BLD-004)")
        version, = struct.unpack("<I", f.read(4))
        rows, cols = struct.unpack("<QQ", f.read(16))
        payload = np.fromfile(f, dtype="<f4")
    if version != 1:
        raise RuntimeError(f"{path}: version {version}, expected 1")
    if payload.size != rows * cols:
        raise RuntimeError(f"{path}: {payload.size} values, header says {rows} x {cols}")
    return payload.reshape(int(rows), int(cols))


def read_kma(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"KMA1":
            raise RuntimeError(f"{path}: magic is {magic!r}, expected b'KMA1' (BLD-004)")
        version, = struct.unpack("<I", f.read(4))
        n, = struct.unpack("<Q", f.read(8))
        payload = np.fromfile(f, dtype="<i4")
    if version != 1 or payload.size != n:
        raise RuntimeError(f"{path}: version {version}, {payload.size} values, header says {n}")
    return payload


# --- 1. the three test executables --------------------------------------------------------------

for label, cmd in (
    ("CPU suite (§16.0, §16.3, TEST-U08/U09/U11/U12/U13, I08, BLD-003a)", ["./test_cpu"]),
    ("GPU kernels (TEST-I01..I07, I10, U01, U03..U07, E16, SMEM-004)",
     ["./test_gpu"] + (["--quick"] if QUICK else [])),
    ("corner cases, cross-strategy, reproducibility (§9.2, TEST-I09/I11/R01/R02)",
     ["./test_edge"] + (["--quick"] if QUICK else [])),
):
    print(f"\n{'=' * 100}\n== {label}\n{'=' * 100}")
    code, output = run(cmd, label=label)
    print(output, end="")

# --- 2. a small valid dataset for the CLI cases -------------------------------------------------

print(f"\n{'=' * 100}\n== BLD-003a / BLD-005: the program as a program\n{'=' * 100}")

PREFIX = os.path.join(SUITE_DIR, "cli")
code, output = run(["./kmeans-gen", "--n", "4000", "--d", "8", "--k", "6", "--sep", "20",
                    "--spread", "1", "--seed", "77", "--out-prefix", PREFIX],
                   label="kmeans-gen for the CLI cases")
print(output, end="")
GOOD_X = PREFIX + "_X.kmd"
GOOD_C = PREFIX + "_C0.kmd"

# The malformed inputs. Each file is wrong in exactly one way, so the rule that rejects it is
# unambiguous.
ZERO_ROWS = os.path.join(SUITE_DIR, "zero_rows_X.kmd")
write_kmd(ZERO_ROWS, 0, 8, [])
ZERO_COLS = os.path.join(SUITE_DIR, "zero_cols_X.kmd")
write_kmd(ZERO_COLS, 10, 0, [])
NAN_DATA = os.path.join(SUITE_DIR, "nan_X.kmd")
_nan_values = np.arange(10 * 4, dtype="<f4")
_nan_values[13] = np.nan                      # row 3, column 1
write_kmd(NAN_DATA, 10, 4, _nan_values)
INF_CENTROIDS = os.path.join(SUITE_DIR, "inf_C0.kmd")
_inf_values = np.zeros(3 * 4, dtype="<f4")
_inf_values[5] = np.inf
write_kmd(INF_CENTROIDS, 3, 4, _inf_values)
SMALL_X = os.path.join(SUITE_DIR, "small_X.kmd")
write_kmd(SMALL_X, 10, 4, np.arange(10 * 4, dtype="<f4"))
SMALL_C = os.path.join(SUITE_DIR, "small_C0.kmd")
write_kmd(SMALL_C, 3, 4, np.arange(3 * 4, dtype="<f4"))
TRUNCATED = os.path.join(SUITE_DIR, "truncated_X.kmd")
with open(GOOD_X, "rb") as _src, open(TRUNCATED, "wb") as _dst:
    _dst.write(_src.read()[:-17])             # a payload short of what the header promises

BASE = ["--max-iter", "10", "--tol", "0.0", "--strategy", "atomic"]

# Each row: EDGE/rule id, why, argv tail, expected exit code, a substring the message must carry.
cases = [
    ("EDGE-01", "N = 0 in the data header (VAL-001)",
     ["--data", ZERO_ROWS, "--centroids", GOOD_C, "--k", "6"], 2, "N"),
    ("EDGE-02", "D = 0 in the data header (VAL-001)",
     ["--data", ZERO_COLS, "--centroids", GOOD_C, "--k", "6"], 2, "D"),
    ("EDGE-03", "K = 0 (VAL-001)",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "0"], 2, "K"),
    ("EDGE-20", "NaN in X, with the offending row and column named (VAL-006)",
     ["--data", NAN_DATA, "--centroids", SMALL_C, "--k", "3"], 2, "VAL-006"),
    ("EDGE-20", "+Inf in C0 (VAL-006)",
     ["--data", SMALL_X, "--centroids", INF_CENTROIDS, "--k", "3"], 2, "VAL-006"),
    ("EDGE-21", "a negative tolerance (VAL-005)",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6", "--tol", "-1"], 2, "VAL-005"),
    ("EDGE-21", "a NaN tolerance (VAL-005)",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6", "--tol", "nan"], 2, "VAL-005"),
    ("EDGE-21", "an unparsable tolerance (BLD-003a)",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6", "--tol", "tiny"], 2, "--tol"),
    ("VAL-004", "a negative iteration limit",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6", "--max-iter", "-1"], 2, "VAL-004"),
    ("VAL-008", "a centroid file whose shape is not K x D",
     ["--data", GOOD_X, "--centroids", SMALL_C, "--k", "6"], 2, None),
    ("VAL-010", "a block size that is not a multiple of 32 (CUDA-007)",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6", "--block-size", "100"], 2, "VAL-010"),
    ("VAL-010", "a block size outside [32, 1024] (CUDA-007)",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6", "--block-size", "2048"], 2, "VAL-010"),
    ("BLD-003a", "an unknown option",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6", "--threads", "8"], 2, "unknown option"),
    ("BLD-003a", "a missing required option", ["--data", GOOD_X, "--centroids", GOOD_C], 2,
     "--k is required"),
    ("BLD-005", "a data file that does not exist (exit 5, I/O)",
     ["--data", os.path.join(SUITE_DIR, "absent.kmd"), "--centroids", GOOD_C, "--k", "6"], 5,
     None),
    ("BLD-005", "a data file shorter than its header promises (exit 5, I/O)",
     ["--data", TRUNCATED, "--centroids", GOOD_C, "--k", "6"], 5, None),
    ("RED-007", "a G beyond N: insufficient resources, exit 3",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6", "--strategy", "reduction",
      "--reduce-chunks", "999999"], 3, "RED-007"),
    ("PERF-007", "--profile-phases with --verify lockstep, whose per-iteration copies would "
                 "land inside the measured phases",
     ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6", "--profile-phases",
      "--verify", "lockstep"], 2, "PERF-007"),
]

def diagnostic_line(output):
    """The error or warning line, which is what these cases are about (BLD-006).

    Not simply the first line: a case that fails inside the run has already printed the
    configuration and the device to stdout, so the first line would be that instead.
    """
    lines = output.strip().splitlines()
    for line in lines:
        if line.startswith("error[") or line.startswith("warning["):
            return line
    return lines[0] if lines else "(no output)"


def argv_for(tail):
    """`tail` plus every BASE option it does not already set."""
    argv = ["./kmeans"] + list(tail)
    for i in range(0, len(BASE), 2):
        if BASE[i] not in tail:
            argv += [BASE[i], BASE[i + 1]]
    return argv


for edge_id, why, tail, expected, substring in cases:
    code, output = run(argv_for(tail), expect=expected, label=f"{edge_id}: {why}")
    if substring is not None and substring not in output:
        failures.append(f"{edge_id}: {why}: the message should mention {substring!r}, got:\n"
                        f"{output.strip()[:600]}")
    print(f"  {'ok  ' if code == expected else 'FAIL'} exit {code} ({edge_id}) {why}")
    print(f"       {diagnostic_line(output)}")

# --- 3. EDGE-27 / EDGE-28: the MEM-008 fault-injection paths ------------------------------------

print("\nEDGE-27 / EDGE-28: injected allocation and launch failures (MEM-008, CUDA-008)")
VALID_TAIL = ["--data", GOOD_X, "--centroids", GOOD_C, "--k", "6"]
REDUCTION_TAIL = VALID_TAIL + ["--strategy", "reduction"]

for label, tail, env_extra in (
    ("EDGE-27: the 1st device allocation fails", VALID_TAIL, {"KMEANS_FAIL_ALLOC": "1"}),
    ("EDGE-27: the 5th device allocation fails, with four already live", VALID_TAIL,
     {"KMEANS_FAIL_ALLOC": "5"}),
    ("EDGE-28: the assign launch fails", VALID_TAIL, {"KMEANS_FAIL_LAUNCH": "assign"}),
    ("EDGE-28: the assign_smem launch fails", VALID_TAIL + ["--assign", "smem"],
     {"KMEANS_FAIL_LAUNCH": "assign_smem"}),
    ("EDGE-28: the reduce_stage1 launch fails", REDUCTION_TAIL,
     {"KMEANS_FAIL_LAUNCH": "reduce_stage1"}),
):
    code, output = run(argv_for(tail), env_extra=env_extra, expect=4, label=label)
    check("MEM-008" in output,
          f"{label}: the message should name MEM-008, got: {output.strip()[:400]}")
    print(f"  {'ok  ' if code == 4 else 'FAIL'} exit {code}  {label}")
    print(f"       {diagnostic_line(output)}")

# A hook that leaked into normal operation would be worse than no hook at all, so the same command
# is run once with nothing injected.
code, output = run(argv_for(VALID_TAIL), label="control run with no hooks set")
print(f"  {'ok  ' if code == 0 else 'FAIL'} exit {code}  control run with no hooks set")

# --- 4. TEST-I08: the FR-015 outputs, read back ------------------------------------------------

print(f"\n{'=' * 100}\n== TEST-I08: the FR-015 outputs and the §13.1 CSV, read back\n{'=' * 100}")

OUT_PREFIX = os.path.join(OUT_DIR, "suite_i08")
CSV_PATH = os.path.join(CSV_DIR, "suite_i08_raw.csv")
if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)

code, output = run(["./kmeans", "--data", GOOD_X, "--centroids", GOOD_C, "--k", "6",
                    "--max-iter", "50", "--tol", "0.0", "--strategy", "reduction",
                    "--assign", "smem", "--verify", "final", "--out-prefix", OUT_PREFIX],
                   label="TEST-I08 run")
print(output, end="")

if code == 0:
    summary = json.load(open(OUT_PREFIX + "_summary.json"))
    assignments = read_kma(OUT_PREFIX + "_A.kma")
    centroids = read_kmd(OUT_PREFIX + "_C.kmd")
    cfg = summary["configuration"]

    check(assignments.size == cfg["N"], f"the .kma holds N = {cfg['N']} assignments, "
                                        f"got {assignments.size}")
    check(assignments.min() >= 0 and assignments.max() < cfg["K"],
          f"every assignment is in [0, K): got [{assignments.min()}, {assignments.max()}]")
    check(centroids.shape == (cfg["K"], cfg["D"]),
          f"the .kmd holds K x D centroids: got {centroids.shape}, want {(cfg['K'], cfg['D'])}")
    check(np.isfinite(centroids).all(), "every reported centroid is finite")

    counts = np.asarray(summary["counts"], dtype=np.int64)
    check(counts.size == cfg["K"], "the summary reports one count per cluster")
    check(counts.sum() == cfg["N"], f"the counts sum to N (FR-009): got {counts.sum()}")
    expected_counts = np.bincount(assignments, minlength=cfg["K"])
    check(np.array_equal(counts, expected_counts),
          "the reported counts equal a bincount of the reported assignments (§2.8 step 1)")

    for field in ("status", "iterations", "centroid_consistent", "assignment_consistent",
                  "last_delta", "objective", "correctness_status", "timing_ms", "environment",
                  "run_id"):
        check(field in summary, f"FR-015 field {field!r} is present in the summary")
    check(summary["correctness_status"] == "PASS",
          f"--verify final reported {summary['correctness_status']}")
    # PERF-001 (v1.9): N*K*D here is far below the 2e10 threshold, so the gate must have checked
    # every point. A sampled result on a dataset this small would mean the threshold logic is
    # wrong, which matters because the benchmark rows are judged by the same code.
    check(summary["p1_check_mode"] == "full",
          f"P1 was checked in full on a {cfg['N']}x{cfg['D']} dataset with K={cfg['K']}: "
          f"got {summary['p1_check_mode']}")
    check(summary["assignment_consistent"] is True, "P1 holds by construction (§2.8)")
    timing = summary["timing_ms"]
    check(timing["t_alg"] <= timing["t_e2e"],
          f"t_alg <= t_e2e (PERF-005): {timing['t_alg']} vs {timing['t_e2e']}")
    check(timing["valid_as_benchmark_sample"] is True,
          "a --verify final run is still a valid timing sample (only lockstep is not)")
    env_fields = summary["environment"]
    check(env_fields["gpu_name"] == ENV["gpu"]["name"],
          f"the summary records the GPU this ran on: {env_fields['gpu_name']}")
    check(env_fields["git_commit"] == ENV["run_version"],
          "the summary records the run version from the configuration cell (PERF-008)")
    print(f"  ok   status {summary['status']}, t = {summary['iterations']}, "
          f"J = {summary['objective']:.6e}, G = {cfg['reduce_chunks_G']}, "
          f"tiles {cfg['tile_k']}x{cfg['tile_d']}")

# FR-020 and the §13.1 raw file: exactly max_iter updates, one CSV row per measured repetition.
reps = 10
code, output = run(["./kmeans", "--data", GOOD_X, "--centroids", GOOD_C, "--k", "6",
                   "--max-iter", "7", "--tol", "0.0", "--strategy", "atomic", "--fixed-iters",
                   "--reps", str(reps), "--warmup", "3", "--csv", CSV_PATH],
                   label="FR-020 / §13.1 run")
print(output, end="")

if code == 0:
    with open(CSV_PATH) as f:
        lines = [line for line in f.read().splitlines() if line]
    header = lines[0].split(",")
    check(len(lines) == reps + 1,
          f"the raw file has one header and {reps} rows: got {len(lines)} lines")
    for line in lines[1:]:
        # Field counting has to respect quoting, which build_flags needs.
        fields, quoted = 1, False
        for ch in line:
            if ch == '"':
                quoted = not quoted
            elif ch == "," and not quoted:
                fields += 1
        check(fields == len(header),
              f"a row has {fields} fields, the header has {len(header)} columns")
    # keep_default_na=False matters: §13.1 writes the literal string NA for a column that
    # describes nothing, and pandas would otherwise turn it into a missing value, which is a
    # different claim.
    raw = pd.read_csv(CSV_PATH, keep_default_na=False)
    check(sorted(raw["rep_index"].tolist()) == list(range(1, reps + 1)),
          f"the repetition indices are 1..{reps}: got {sorted(raw['rep_index'].tolist())}")
    check((raw["iterations"] == 7).all(),
          f"--fixed-iters performs exactly max_iter updates (FR-020): "
          f"got {raw['iterations'].unique().tolist()}")
    check((raw["status"] == "ITERATION_LIMIT").all(),
          "--fixed-iters reports ITERATION_LIMIT (FR-020)")
    check((raw["t_alg_ms"] <= raw["t_e2e_ms"]).all(), "t_alg <= t_e2e in every row (PERF-005)")
    check(raw["run_id"].nunique() == reps, "every row has its own run_id (§13.1)")
    check((raw["tile_k"].astype(str) == "NA").all(),
          "the tile columns are NA for the baseline assignment kernel")
    print(f"  ok   {len(raw)} raw rows, t_alg median {raw['t_alg_ms'].median():.3f} ms "
          "(plumbing check, not a benchmark result)")

# PERF-007: the phase columns of §13.1, which are NA in every row above. The phases cover the
# T_alg region and cannot sum past it; what they leave out is the per-iteration 16-byte transfer
# and the host decision gap, which belong to no phase.
PHASE_CSV = os.path.join(CSV_DIR, "suite_phases_raw.csv")
if os.path.exists(PHASE_CSV):
    os.remove(PHASE_CSV)

code, output = run(["./kmeans", "--data", GOOD_X, "--centroids", GOOD_C, "--k", "6",
                    "--max-iter", "7", "--tol", "0.0", "--strategy", "reduction",
                    "--assign", "smem", "--fixed-iters", "--profile-phases",
                    "--reps", str(reps), "--warmup", "1", "--csv", PHASE_CSV],
                   label="PERF-007 phase-timing run")
print(output, end="")

if code == 0:
    check(any(line.startswith("phases:") for line in output.splitlines()),
          "the run prints a phase breakdown")
    check("warning[PERF-007]" in output,
          "and says that phase rows are perturbed and not headline numbers")
    phase_raw = pd.read_csv(PHASE_CSV, keep_default_na=False)
    phase_cols = ["t_assign_ms", "t_accum_ms", "t_update_ms", "t_conv_ms"]
    check(len(phase_raw) == reps, f"the phase run wrote {reps} rows: got {len(phase_raw)}")
    numeric = phase_raw[phase_cols].apply(pd.to_numeric, errors="coerce")
    check(numeric.notna().all().all(),
          f"every phase column holds a number, not NA: {phase_raw[phase_cols].head(1).to_dict()}")
    check((numeric >= 0).all().all(), "no phase sum is negative")
    total = numeric.sum(axis=1)
    # Not a per-phase "> 0": on a dataset this small the update and convergence kernels run in a
    # few microseconds and a phase could legitimately round to zero at the event resolution. The
    # claim that matters is that the phases are inside T_alg and account for most of it.
    check((total > 0).all(), "the phases account for some of t_alg")
    check((total <= phase_raw["t_alg_ms"] * 1.001).all(),
          "the phases sum to no more than t_alg, which is the region they cover: "
          f"worst ratio {(total / phase_raw['t_alg_ms']).max():.4f}")
    print(f"  ok   phases over {len(phase_raw)} rows: median sum "
          f"{total.median():.3f} ms of t_alg median {phase_raw['t_alg_ms'].median():.3f} ms "
          "(plumbing check, not a benchmark result)")

# --- the verdict --------------------------------------------------------------------------------

print(f"\n{'=' * 100}")
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    raise RuntimeError(
        f"{len(failures)} test failure(s); see the list above. Each one names the requirement it "
        "violates. Fix the source in src/ or tests/, regenerate the notebook with "
        "tools/build_notebook.py, and re-run this cell before trusting anything below it."
    )

print("TEST-B02: the full suite passed." +
      ("  QUICK = True, so test_gpu and test_edge ran their reduced selections; set QUICK = False "
       "for the §16 acceptance evidence." if QUICK else
       "  QUICK = False: this is the §16.1-§16.5 acceptance evidence for this GPU."))
