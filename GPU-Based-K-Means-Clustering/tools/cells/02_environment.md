## 2. Environment record

Records the machine-dependent items left open by requirements.md §1.6.2 — ENV-01 (GPU model and
compute capability), ENV-02 (CUDA toolkit, driver, `compute-sanitizer`), ENV-03 (host OS and host
compiler) and ENV-08 (profiler availability) — into the `ENV` dictionary that every later cell
reads, and that PERF-008 requires in each benchmark CSV row.

No version is assumed anywhere: whatever `nvidia-smi`, `nvcc` and `g++` report is what is recorded
and printed. A missing tool is recorded as unavailable rather than raising, because ENV-08 allows
the profilers to be absent (PERF-009 and TEST-P03 then fall back to CUDA-event phase timing, which
is reported as a limitation). The configuration cell, not this one, decides whether the recorded
environment is usable.

`nvidia-smi` does not report the SM count, and ENV-06 (FP64:FP32 throughput ratio) is a vendor
specification figure, not a queryable property. The SM count comes from
`cudaGetDeviceProperties` — printed by the smoke test below and required of the program itself by
CUDA-001 — and ENV-06 is cited from the vendor specification in the evaluation section.
