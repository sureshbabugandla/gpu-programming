# Host compile cell — BLD-002 host flags plus -Wall -Wextra; no warnings tolerated (NFR-014).
import subprocess

if "GXX_REFERENCE_FLAGS" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

# One source of truth: section 3 owns -O2 -std=c++17 -ffp-contract=off (NUM-003, BLD-002).
GXX_FLAGS = GXX_REFERENCE_FLAGS + ["-Wall", "-Wextra"]
CPU_LIB_SOURCES = ["validate.cpp", "io_kmd.cpp", "cpu_reference.cpp", "gen.cpp"]


def compile_host(output, sources):
    """Compile one host executable; raise on a non-zero exit or on any compiler output."""
    cmd = ["g++"] + GXX_FLAGS + ["-I", "."] + sources + ["-o", output]
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    diagnostics = (proc.stdout + proc.stderr).strip()
    if diagnostics:
        print(diagnostics)
    if proc.returncode != 0:
        raise RuntimeError(f"g++ failed with exit code {proc.returncode} building {output}")
    if diagnostics:
        raise RuntimeError(
            f"g++ emitted diagnostics while building {output}; NFR-014 requires a warning-free "
            "host build. Fix the source in src/ or tests/, regenerate the notebook, and re-run."
        )
    print(f"  -> {output}")


compile_host("kmeans-gen", ["gen_main.cpp", "gen.cpp", "io_kmd.cpp", "validate.cpp"])
compile_host("test_cpu", ["test_cpu.cpp"] + CPU_LIB_SOURCES)
print("\nhost build clean: kmeans-gen and test_cpu, no warnings")
