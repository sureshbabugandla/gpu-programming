## 3. Configuration

Derives everything machine-dependent from the record above — nothing is hard-coded before ENV-01
is known (BLD-001c):

- **`SM_ARCH`** from the reported compute capability (`7.5` → `sm_75`).
- **Fail fast** when the environment cannot run the assignment: no GPU attached, compute capability
  below 6.0 (CUDA-001 — `atomicAdd` on `double`, which ATOM-003 requires, is not native below
  6.0), or a CUDA 13.x toolkit paired with a device below CC 7.5, whose offline compilation that
  toolkit removed (BLD-001a). Each case raises with the action to take, so "restart kernel and run
  all" stops here rather than failing later inside a build.
- **Build flags** (BLD-002): a release build `-O3 -lineinfo -arch=$SM_ARCH -std=c++17` with
  `-Xptxas -v` so register and shared-memory usage appear in the build log (CUDA-010), and a
  sanitizer/debug build that adds `-DKMEANS_SYNC_CHECK` (CUDA-008). `--use_fast_math` is never
  used (NUM-002), and the CPU reference is compiled with FP contraction disabled (NUM-003).
- **`QUICK`**: `True` keeps the run to the compile, hand-dataset and lockstep cells — the checks
  that decide whether the kernels are correct at all. It skips the longer sweeps (the `G` sweep,
  partial Stage 2 tiles, the non-aligned size matrix) and, from M5/M6, the sanitizer and benchmark
  sections (BLD-007). Set it to `False` for a run that is meant to produce submission evidence.
- **Paths**: results — benchmark CSVs, output files and logs — go to Google Drive when
  `USE_DRIVE` is enabled and the mount succeeds, so a runtime disconnect does not lose them;
  otherwise a warning is printed, local disk is used, and the fallback is recorded in `ENV` so
  results that were not persisted stay identifiable (BLD-001c). Sources, binaries and generated
  datasets always stay on local disk: the Drive FUSE mount does not permit executing binaries, so
  anything built there could not be run, and datasets are reproducible from their seeds (FR-003)
  while reaching tens of GB at the top of the PERF-003 matrix. The cell `chdir`s into the build
  directory, which is why the `%%writefile` cells use flat file names.
