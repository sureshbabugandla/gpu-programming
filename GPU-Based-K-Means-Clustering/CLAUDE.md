# Problem 5: GPU-Based K-Means Clustering — rules for Claude Code

## Contract
- `requirements.md` is the specification and the single source of truth. Read the relevant sections before each milestone.
- Cite requirement IDs (FR-, CUDA-, ATOM-, RED-, SMEM-, NUM-, PERF-, TEST-) in commit messages, and in code comments only where they explain a non-obvious choice.
- Never change convergence, empty-cluster, tie-breaking, iteration-counting, tolerance or finalization semantics in code. If a requirement looks wrong or impractical, stop and propose a spec change per requirements.md §18.4 (reason, old text, new text, test impact) and wait for approval.

## Environment
- This laptop has no NVIDIA GPU. CUDA code may be compiled locally if `nvcc` is available, but it is executed only on Google Colab (Tesla T4).
- Never claim that a GPU test, sanitizer run or benchmark passed unless an executed notebook in `results/` shows it.
- CPU-only parts (validation, file I/O, generator, CPU reference, H1–H6 reference tests) must be compiled and run locally with g++ before committing.

## Layout and workflow (requirements.md DEC-011, DEC-012, §15.2a)
- Source lives in `src/` and `tests/`. `tools/build_notebook.py` generates `kmeans_cuda.ipynb` with one `%%writefile` cell per file, in the §15.3 cell order. Never hand-edit the notebook; regenerate it.
- After each milestone: regenerate the notebook, commit, and tell me exactly which notebook sections to run on Colab and what output to expect.
- Executed notebooks come back in `results/`. When asked, read their outputs and fix failures in `src/`.
- Never commit credentials or tokens.

## Work order (requirements.md §18.2)
Work one milestone at a time and stop after each:
M0 skeleton + notebook generator → M1 validation, I/O, generator, CPU reference, H1–H6 →
M2 device setup, baseline `assign`, Approach A, update/convergence/finalization →
M3 Approach B → M4 `assign_smem` → M5 full tests + sanitizer cells → M6 benchmarks + analysis cells →
M7 traceability matrix update.

## Code style
- CUDA C++ with C++17 host code. Clear names, simple control flow.
- Comments only where they explain synchronization, data ownership, numerical choices or non-obvious indexing.
- No new frameworks or dependencies beyond the CUDA runtime, the C++ standard library, and numpy/pandas/matplotlib in the notebook.
- After writing each kernel, be ready to explain its thread mapping, ownership and synchronization so I can defend it.
