# Configuration — requirements.md §15.3 part 3; BLD-001a, BLD-001c, BLD-002, CUDA-001, NUM-002/003.
import os
import re
import subprocess

USE_DRIVE = True  # BLD-001c: persist results to Drive; falls back to local disk on failure.
DRIVE_RESULTS_DIR = "/content/drive/MyDrive/kmeans_cuda"
BUILD_DIR = "/content/kmeans"  # always local: see the note on exec permissions below

# QUICK keeps a development run to the compile, hand-dataset and lockstep cells: the checks that
# decide whether the kernels are correct at all. It drops the longer sweeps (the G sweep, partial
# Stage 2 tiles and the non-aligned size matrix) and, from M5/M6, the sanitizer and benchmark
# sections. Set it to False for a run that produces submission evidence (BLD-007).
QUICK = True

if "ENV" not in globals():
    raise RuntimeError("Run the environment cell (section 2) first: ENV is not defined.")

# --- ENV-01 -> SM_ARCH (BLD-001c: derived, never hard-coded) ------------------------------------
if not ENV.get("gpu"):
    raise RuntimeError(
        "No NVIDIA GPU visible (ENV-01). Set Runtime -> Change runtime type -> Hardware "
        "accelerator: GPU, then re-run from section 2. Probe said: "
        f"{ENV.get('gpu_probe', 'unknown')}"
    )

COMPUTE_CAP = ENV["gpu"]["compute_cap"].strip()
_cc_match = re.fullmatch(r"(\d+)\.(\d+)", COMPUTE_CAP)
if not _cc_match:
    raise RuntimeError(f"Could not parse compute capability {COMPUTE_CAP!r} reported by nvidia-smi.")
CC_MAJOR, CC_MINOR = int(_cc_match.group(1)), int(_cc_match.group(2))
CC_VALUE = CC_MAJOR + CC_MINOR / 10.0
SM_ARCH = f"sm_{CC_MAJOR}{CC_MINOR}"

# CUDA-001: ATOM-003 needs native atomicAdd(double), which arrived with compute capability 6.0.
if CC_VALUE < 6.0:
    raise RuntimeError(
        f"Compute capability {COMPUTE_CAP} is below the 6.0 minimum (CUDA-001): native "
        "atomicAdd(double) required by ATOM-003 is unavailable. Use a newer GPU runtime."
    )

# ENV-02: whatever nvcc reports is what is used; no toolkit version is assumed.
NVCC_VERSION = ENV.get("nvcc")
if not NVCC_VERSION:
    raise RuntimeError("nvcc is not available in this runtime (ENV-02); CUDA code cannot be built.")
_rel = re.search(r"release (\d+)\.(\d+)", NVCC_VERSION)
CUDA_TOOLKIT = f"{_rel.group(1)}.{_rel.group(2)}" if _rel else "unknown"

# BLD-001a: CUDA 13.x dropped offline compilation for compute capability < 7.5.
if _rel and int(_rel.group(1)) >= 13 and CC_VALUE < 7.5:
    raise RuntimeError(
        f"CUDA toolkit {CUDA_TOOLKIT} cannot compile offline for compute capability "
        f"{COMPUTE_CAP} (BLD-001a). Use a CUDA 12.x toolkit, or a GPU with CC >= 7.5."
    )

# --- BLD-002 build flags ------------------------------------------------------------------------
# No --use_fast_math anywhere (NUM-002): it would change the distance semantics of §2.2.
NVCC_COMMON = ["-std=c++17", f"-arch={SM_ARCH}", "-lineinfo"]
NVCC_RELEASE_FLAGS = NVCC_COMMON + ["-O3", "-Xptxas", "-v"]          # CUDA-010 usage in build log
NVCC_DEBUG_FLAGS = NVCC_COMMON + ["-O3", "-DKMEANS_SYNC_CHECK"]      # CUDA-008 localises faults
# NUM-003: the CPU reference is the oracle and must not contract multiply-adds.
NVCC_HOST_FP_FLAGS = ["-Xcompiler", "-ffp-contract=off"]
GXX_REFERENCE_FLAGS = ["-O2", "-std=c++17", "-ffp-contract=off"]

# --- BLD-001c paths ----------------------------------------------------------------------------
# Sources, binaries and generated datasets stay on local disk and only results go to Drive, for
# two reasons: the Drive FUSE mount does not permit executing binaries, so a kmeans built there
# could not be run, and datasets reach tens of GB at the top of the PERF-003 matrix while being
# reproducible from their seeds (FR-003), so they are not worth persisting.
RESULTS_ROOT, DRIVE_STATUS = BUILD_DIR, "disabled"
if USE_DRIVE:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        RESULTS_ROOT, DRIVE_STATUS = DRIVE_RESULTS_DIR, "mounted"
    except Exception as exc:
        DRIVE_STATUS = f"unavailable: {exc!r}"
        print(
            "warning[BLD-001c]: Google Drive not mounted; results stay in "
            f"{BUILD_DIR} and are lost if the runtime disconnects, so download anything you "
            f"need before closing the session. Reason: {exc!r}"
        )

WORK_DIR = BUILD_DIR                              # cwd for the %%writefile cells and the compiler
DATA_DIR = os.path.join(BUILD_DIR, "data")        # generated datasets (FR-003), regenerable
OUT_DIR = os.path.join(RESULTS_ROOT, "out")       # .kma/.kmd outputs and JSON summaries (BLD-004)
CSV_DIR = os.path.join(RESULTS_ROOT, "csv")       # benchmark CSVs (section 13.1)
LOG_DIR = os.path.join(RESULTS_ROOT, "logs")      # test and sanitizer logs (section 1.5)
for _d in (BUILD_DIR, DATA_DIR, OUT_DIR, CSV_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)
os.chdir(WORK_DIR)  # the %%writefile cells below use flat names relative to this directory

# PERF-008 asks for a build identity per benchmark row. In Colab there is no git checkout, so the
# executed notebook saved under results/ is itself the evidence (DEC-012); record what is knowable.
try:
    RUN_VERSION = "git:" + subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30
    ).stdout.strip()
except Exception:
    RUN_VERSION = "unknown"
if RUN_VERSION == "git:":
    RUN_VERSION = "unknown"

# PERF-008: the executables record the build identity in their JSON summary and CSV rows, and
# neither the source revision nor the compiler flags are knowable from inside the compiler, so
# they are passed through the environment rather than invented by the program.
os.environ["KMEANS_RUN_VERSION"] = RUN_VERSION
os.environ["KMEANS_BUILD_FLAGS"] = " ".join(NVCC_RELEASE_FLAGS + NVCC_HOST_FP_FLAGS)

ENV["drive_status"] = DRIVE_STATUS          # BLD-001c: unpersisted results stay identifiable
ENV["quick"] = QUICK                        # which sections this run actually executed
ENV["work_dir"] = WORK_DIR
ENV["results_root"] = RESULTS_ROOT
ENV["sm_arch"] = SM_ARCH
ENV["cuda_toolkit"] = CUDA_TOOLKIT
ENV["run_version"] = RUN_VERSION

print(f"GPU             : {ENV['gpu']['name']} (compute capability {COMPUTE_CAP})")
print(f"SM_ARCH         : {SM_ARCH}   [derived from ENV-01, BLD-001c]")
print(f"CUDA toolkit    : {CUDA_TOOLKIT} (as reported by nvcc)")
print(f"Drive           : {DRIVE_STATUS}")
print(f"QUICK           : {QUICK}   [True = compile, hand datasets and lockstep only]")
print(f"build dir (cwd) : {WORK_DIR}   [local: Drive mounts are not executable]")
print(f"results root    : {RESULTS_ROOT}")
print(f"data / out / csv: {DATA_DIR} | {OUT_DIR} | {CSV_DIR}")
print(f"run version     : {RUN_VERSION}")
print(f"nvcc release    : {' '.join(NVCC_RELEASE_FLAGS)}")
print(f"nvcc debug      : {' '.join(NVCC_DEBUG_FLAGS)}")
print(f"host FP (oracle): {' '.join(NVCC_HOST_FP_FLAGS)} / g++ {' '.join(GXX_REFERENCE_FLAGS)}")
