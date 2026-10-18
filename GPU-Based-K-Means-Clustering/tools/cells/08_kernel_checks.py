# TEST-U06 and TEST-I10 / §10.3: buffer initialisation and lockstep verification.
# QUICK (section 3) drops the TEST-U05 / U07 / E16 sweeps; the lockstep run itself always runs.
import subprocess

if "QUICK" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

cmd = ["./test_gpu", "--only", "kernels"] + (["--quick"] if QUICK else [])
print("$ " + " ".join(cmd))
checks = subprocess.run(cmd, capture_output=True, text=True)
print(checks.stdout + checks.stderr, end="")
if checks.returncode != 0:
    raise RuntimeError(
        f"{' '.join(cmd)} exited with code {checks.returncode}: the device disagrees with the CPU "
        "reference beyond what NUM-005/NUM-006 allow, or a poisoned buffer reached a result. The "
        "failure lines above name the kernel and the quantity; fix it in src/, regenerate the "
        "notebook, and re-run."
    )

if QUICK:
    print("\nQUICK = True: TEST-U05 (G sweep), TEST-U07 (partial tiles) and TEST-E16 "
          "(non-aligned sizes) were skipped. Set QUICK = False in section 3 to run them.")
