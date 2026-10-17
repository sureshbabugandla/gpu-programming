# Environment record — requirements.md §1.6.2 (ENV-01..ENV-03, ENV-08), PERF-008.
# Every probe is tolerant: a missing tool is recorded as unavailable, never raised. Nothing here
# assumes a GPU model, a toolkit version or a driver version; the reported strings are the record.
import json
import platform
import shutil
import subprocess
import sys


def _run(cmd, timeout=120):
    """Run a probe command. Returns (ok, text); never raises."""
    if shutil.which(cmd[0]) is None:
        return False, "not installed"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # a probe must not break the notebook
        return False, f"probe failed: {exc!r}"
    out = (p.stdout or "").strip() or (p.stderr or "").strip()
    return p.returncode == 0, out


def _first_line(text):
    return text.splitlines()[0].strip() if text else ""


def _version_of(cmd):
    """Version string of a tool invoked as `<tool> --version`, or None."""
    ok, out = _run([cmd, "--version"])
    return _first_line(out) if ok else None


ENV = {}

# ENV-03 — host OS and host compiler of the runtime VM.
ENV["os"] = {
    "platform": platform.platform(),
    "release": _run(["cat", "/etc/os-release"])[1].replace("\n", "; ") or None,
    "python": sys.version.split()[0],
}
ENV["host_compiler"] = _version_of("g++")

# ENV-01 — GPU model and compute capability.
gpu_ok, gpu_csv = _run(
    ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
     "--format=csv,noheader"]
)
ENV["gpu"] = None
if gpu_ok and gpu_csv:
    fields = [f.strip() for f in gpu_csv.splitlines()[0].split(",")]
    if len(fields) == 4:
        ENV["gpu"] = {
            "name": fields[0],
            "compute_cap": fields[1],
            "driver_version": fields[2],
            "memory_total": fields[3],
        }
        ENV["gpu_count"] = len(gpu_csv.splitlines())
if ENV["gpu"] is None:
    ENV["gpu_probe"] = gpu_csv or "nvidia-smi unavailable"

# ENV-02 — CUDA toolkit and the sanitizer the §16.4 suite needs.
nvcc_ok, nvcc_out = _run(["nvcc", "--version"])
ENV["nvcc"] = nvcc_out.replace("\n", " | ") if nvcc_ok else None
ENV["compute_sanitizer"] = _version_of("compute-sanitizer")

# ENV-08 — profiler availability; absence is a recorded limitation, not an error.
ENV["profilers"] = {"nsys": _version_of("nsys"), "ncu": _version_of("ncu")}

# Notebook analysis stack (§1.5 deliverables 4 and 5; no other dependency is permitted).
ENV["python_packages"] = {}
for _pkg in ("numpy", "pandas", "matplotlib"):
    try:
        ENV["python_packages"][_pkg] = __import__(_pkg).__version__
    except Exception:
        ENV["python_packages"][_pkg] = None

print(json.dumps(ENV, indent=2))
