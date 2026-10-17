# CPU-level test suite: §16.0 hand datasets plus TEST-U08/U09/U11/U12/U13. Raises on any failure.
import os
import subprocess

if "DATA_DIR" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

tests = subprocess.run(["./test_cpu"], capture_output=True, text=True)
print(tests.stdout + tests.stderr, end="")
if tests.returncode != 0:
    raise RuntimeError(
        f"./test_cpu exited with code {tests.returncode}: the CPU reference does not reproduce the "
        "hand-verified results of requirements.md §16.0. Every GPU comparison depends on this "
        "oracle, so fix it in src/ before continuing."
    )

# FR-003 / BLD-003a: one small dataset, to confirm the generator and the BLD-004 format work here.
prefix = os.path.join(DATA_DIR, "demo_n2000_d4_k3")
gen = subprocess.run(
    ["./kmeans-gen", "--n", "2000", "--d", "4", "--k", "3", "--sep", "10", "--spread", "1",
     "--seed", "20261017", "--out-prefix", prefix],
    capture_output=True, text=True,
)
print("\n$ kmeans-gen --n 2000 --d 4 --k 3 --sep 10 --spread 1 --seed 20261017")
print(gen.stdout + gen.stderr, end="")
if gen.returncode != 0:
    raise RuntimeError(f"kmeans-gen exited with code {gen.returncode} (see BLD-005 for the codes)")

for suffix, expected_rows in (("_X.kmd", 2000), ("_C0.kmd", 3)):
    path = prefix + suffix
    # BLD-004: 24-byte header (magic, version, rows, cols) then rows*cols binary32.
    expected_bytes = 24 + expected_rows * 4 * 4
    actual = os.path.getsize(path)
    if actual != expected_bytes:
        raise RuntimeError(f"{path} is {actual} B, expected {expected_bytes} B for "
                           f"{expected_rows}x4 binary32 values plus the BLD-004 header")
    print(f"  {os.path.basename(path)}: {actual} B as the format requires")
