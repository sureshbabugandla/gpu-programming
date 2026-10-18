# SMEM-004: assign_smem against the baseline assign on every test dataset.
# QUICK (section 3) drops the TEST-E16 size matrix and the assign_smem lockstep run.
import subprocess

if "QUICK" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

cmd = ["./test_gpu", "--only", "smem"] + (["--quick"] if QUICK else [])
print("$ " + " ".join(cmd))
smem = subprocess.run(cmd, capture_output=True, text=True)
print(smem.stdout + smem.stderr, end="")
if smem.returncode != 0:
    raise RuntimeError(
        f"{' '.join(cmd)} exited with code {smem.returncode}: assign_smem does not reproduce the "
        "baseline assign bit for bit (SMEM-004). The failure lines above name the dataset and the "
        "first differing quantity. Either the tiled kernel changed the accumulation order "
        "(NUM-002) or ptxas contracted a multiply-add in one kernel and not the other; the second "
        "must be shown in the -Xptxas -v output and bounded by NUM-005 before it is accepted. Fix "
        "it in src/, regenerate the notebook, and re-run."
    )

if QUICK:
    print("\nQUICK = True: the TEST-E16 size matrix through both kernels, the three larger tile "
          "shapes and the assign_smem lockstep run were skipped. Set QUICK = False in section 3 "
          "to run them.")
