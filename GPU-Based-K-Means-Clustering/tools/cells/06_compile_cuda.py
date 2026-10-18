# CUDA compile cell — BLD-002 release flags, CUDA-010 register/shared usage in the build log.
import re
import subprocess

if "NVCC_RELEASE_FLAGS" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

# Section 3 owns the flags. The CPU reference is linked into every binary here, so the host FP
# flags of NUM-003 apply to the whole compile: a contracted multiply-add in distance2_fp32 would
# change the oracle the device is judged against.
CUDA_SOURCES = ["kernels.cu", "kmeans_gpu.cu"]
HOST_SOURCES = ["cpu_reference.cpp", "validate.cpp", "io_kmd.cpp", "gen.cpp", "cli_options.cpp",
                "report.cpp", "lockstep.cpp"]


def compile_cuda(output, sources, extra_flags=(), flags=None):
    """Compile one nvcc executable; raise on a non-zero exit. Returns the combined output."""
    cmd = (["nvcc"] + (NVCC_RELEASE_FLAGS if flags is None else list(flags)) +
           NVCC_HOST_FP_FLAGS + list(extra_flags) + ["-I", "."] + sources + ["-o", output])
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


def report_kernel_usage(build_log):
    """CUDA-010 / NFR-010: per-kernel registers, shared memory and spills, not buried in the log.

    A spill in assign_smem is singled out because SMEM-001 keeps the T_K partial distances in
    registers on purpose: if they land in local memory instead, the SMEM-005 comparison is
    measuring something other than what it claims to (CUDA-010).
    """
    print("\nper-kernel resource usage (-Xptxas -v, CUDA-010):")
    usage = re.findall(
        r"Function properties for (\S+).*?(\d+) bytes stack frame, (\d+) bytes spill stores, "
        r"(\d+) bytes spill loads\s*ptxas info\s*:\s*Used (\d+) registers"
        r"(?:, used \d+ barriers)?(?:, (\d+) bytes smem)?",
        build_log,
        re.DOTALL,
    )
    if not usage:
        # Not a failure: the ptxas wording varies between toolkit versions, and the raw log is
        # printed above.
        print("  (could not parse the ptxas lines; read the raw -Xptxas -v output above)")
        return
    spilling = []
    for name, stack, spill_store, spill_load, registers, smem in usage:
        spilled = int(spill_store) + int(spill_load)
        flag = "   <-- SPILLS TO LOCAL MEMORY" if spilled else ""
        print(f"  {name:<52} {registers:>4} registers, {smem or '0':>5} B smem, "
              f"{stack:>4} B stack, {spilled:>4} B spill{flag}")
        if spilled and "assign_smem" in name:
            spilling.append(name)
    if spilling:
        print(f"\nwarning[CUDA-010]: {', '.join(spilling)} spills to local memory. SMEM-001 keeps "
              "the T_K partials in registers; report this with the SMEM-005 benchmark results and "
              "consider a smaller T_K.")


# The three executables of §15.3: the test drivers and the BLD-003a program itself.
build_log = compile_cuda("test_gpu", ["test_gpu.cu"] + CUDA_SOURCES + HOST_SOURCES)
compile_cuda("test_edge", ["test_edge.cu"] + CUDA_SOURCES + HOST_SOURCES)
compile_cuda("kmeans", ["kmeans_main.cu"] + CUDA_SOURCES + HOST_SOURCES)

report_kernel_usage(build_log)

print("\nCUDA build clean: test_gpu, test_edge, kmeans")
