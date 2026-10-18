# TEST-I01 / TEST-I02: the §16.0 hand datasets on the device, both strategies.
import subprocess

if "WORK_DIR" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

hand = subprocess.run(["./test_gpu", "--only", "hand"], capture_output=True, text=True)
print(hand.stdout + hand.stderr, end="")
if hand.returncode != 0:
    raise RuntimeError(
        f"./test_gpu --only hand exited with code {hand.returncode}: a kernel disagrees with the "
        "hand-derived results of requirements.md §16.0. Fix the source in src/, regenerate the "
        "notebook, and re-run; do not continue to the kernel checks or the benchmarks."
    )
