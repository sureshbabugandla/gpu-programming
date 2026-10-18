### 6b. CUDA compile cell (BLD-002, CUDA-010)

Builds the M2–M4 GPU test executable with the release flags of §15.3 part 3:
`-O3 -lineinfo -arch=$SM_ARCH -std=c++17`, **no `--use_fast_math`** (it would change the distance
semantics of §2.2, NUM-002), and `-Xcompiler -ffp-contract=off` so the CPU reference compiled into
the same binary stays the oracle it is meant to be (NUM-003).

`-Xptxas -v` prints per-kernel register and shared-memory usage, which CUDA-010 requires in the
build log and NFR-010 reports alongside benchmark results. Two lines matter in particular:
`assign_smem` holds `T_K` partial distances in registers and must show **0 bytes spill** — a spill
there would move them to local memory and invalidate the SMEM-005 comparison — and its shared
memory should read 3072 B at the defaults (1 KiB tile + 2 KiB change-count reduction).

The cell raises on a non-zero `nvcc` exit code, so "restart kernel and run all" stops here rather
than at a confusing runtime failure (NFR-014).
