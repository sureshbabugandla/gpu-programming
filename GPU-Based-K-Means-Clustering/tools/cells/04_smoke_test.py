# Toolchain smoke test — requirements.md §15.3 part 4. Raises on any failure so that
# "restart kernel and run all" stops here instead of inside a larger build (NFR-014).
import subprocess

if "SM_ARCH" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first: SM_ARCH is not defined.")

compile_cmd = ["nvcc"] + NVCC_RELEASE_FLAGS + ["-o", "hello", "hello.cu"]
print("$ " + " ".join(compile_cmd))
compile_proc = subprocess.run(compile_cmd, capture_output=True, text=True)
print(compile_proc.stdout + compile_proc.stderr, end="")  # -Xptxas -v usage report (CUDA-010)
if compile_proc.returncode != 0:
    raise RuntimeError(
        f"nvcc failed with exit code {compile_proc.returncode} for -arch={SM_ARCH} "
        f"(toolkit {CUDA_TOOLKIT}, device CC {COMPUTE_CAP}). See the output above; check BLD-001a "
        "if the toolkit is too new for this device."
    )

run_proc = subprocess.run(["./hello"], capture_output=True, text=True)
print(run_proc.stdout + run_proc.stderr, end="")
if run_proc.returncode != 0:
    raise RuntimeError(
        f"./hello exited with code {run_proc.returncode} "
        "(4 = CUDA runtime error per BLD-005). See the output above."
    )

smoke = dict(
    line.split("=", 1) for line in run_proc.stdout.splitlines() if "=" in line
)
if smoke.get("kernel.check") != "OK":
    raise RuntimeError(
        f"Kernel result wrong: {smoke.get('kernel.mismatches')} of "
        f"{smoke.get('kernel.elements')} elements mismatched. The toolchain builds but the device "
        "does not compute correctly; do not trust any later result."
    )
if smoke.get("device.compute_capability") != COMPUTE_CAP:
    raise RuntimeError(
        f"Device reports compute capability {smoke.get('device.compute_capability')} to host code "
        f"but nvidia-smi reported {COMPUTE_CAP} in section 2, so SM_ARCH={SM_ARCH} was derived "
        "from a different device (BLD-001c). Re-run from section 2."
    )

# SM count feeds CUDA-007 grid sizing and the Approach B chunk count G (RED-006); the shared-memory
# limits bound the assign_smem tiles (SMEM-003, MEM-004). Recorded now, used from M2 on.
ENV["device_properties"] = smoke
SM_COUNT = int(smoke["device.sm_count"])
SHARED_MEM_PER_BLOCK = int(smoke["device.shared_mem_per_block_bytes"])

print(
    f"\nsmoke test OK: {smoke['device.name']}, CC {smoke['device.compute_capability']}, "
    f"{SM_COUNT} SMs, {SHARED_MEM_PER_BLOCK} B shared memory per block, "
    f"CUDA runtime {smoke['runtime.cuda_version']}, driver {smoke['driver.cuda_version']}, "
    f"kernel sum {smoke['kernel.sum']} == expected {smoke['kernel.expected_sum']}"
)
