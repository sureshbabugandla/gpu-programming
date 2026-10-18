# CUDA compile cell — BLD-002 release flags, CUDA-010 register/shared usage in the build log.
import re
import subprocess

if "NVCC_RELEASE_FLAGS" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

# Section 3 owns the flags. The CPU reference is linked into the GPU test binary, so the host FP
# flags of NUM-003 apply to the whole compile: a contracted multiply-add in distance2_fp32 would
# change the oracle the device is judged against.
CUDA_SOURCES = ["kernels.cu", "kmeans_gpu.cu"]
HOST_SOURCES = ["cpu_reference.cpp", "validate.cpp", "io_kmd.cpp", "gen.cpp"]


def compile_cuda(output, sources, extra_flags=()):
    """Compile one nvcc executable; raise on a non-zero exit. Returns the combined output."""
    cmd = (["nvcc"] + NVCC_RELEASE_FLAGS + NVCC_HOST_FP_FLAGS + list(extra_flags) +
           ["-I", "."] + sources + ["-o", output])
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    diagnostics = (proc.stdout + proc.stderr).strip()
    if diagnostics:
        print(diagnostics)
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvcc failed with exit code {proc.returncode} building {output}. Fix the source in "
            "src/ or tests/, regenerate the notebook with tools/build_notebook.py, and re-run."
        )
    print(f"  -> {output}")
    return diagnostics


build_log = compile_cuda("test_gpu", ["test_gpu.cu"] + CUDA_SOURCES + HOST_SOURCES)

# CUDA-010 / NFR-010: surface the per-kernel resource usage rather than leaving it buried in the
# -Xptxas -v noise, and flag any spill to local memory, which SMEM/PERF conclusions depend on.
print("\nper-kernel resource usage (-Xptxas -v, CUDA-010):")
usage = re.findall(
    r"Function properties for (\S+).*?(\d+) bytes stack frame, (\d+) bytes spill stores, "
    r"(\d+) bytes spill loads\s*ptxas info\s*:\s*Used (\d+) registers",
    build_log,
    re.DOTALL,
)
if usage:
    for name, stack, spill_store, spill_load, registers in usage:
        spilled = int(spill_store) + int(spill_load)
        flag = "   <-- SPILLS TO LOCAL MEMORY" if spilled else ""
        print(f"  {name:<48} {registers:>4} registers, {stack:>4} B stack, "
              f"{spilled:>4} B spill{flag}")
else:
    # Not a failure: ptxas wording varies between toolkit versions, and the raw log is above.
    print("  (could not parse the ptxas lines; read the raw -Xptxas -v output printed above)")

print("\nCUDA build clean: test_gpu")
