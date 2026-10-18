### 6c. Sanitizer build (BLD-002)

`compute-sanitizer` runs an ordinary binary, but two things make a separate build worth the extra
minute. `-lineinfo` lets the tool name the source line of a hazard instead of only a PTX offset,
and `KMEANS_SYNC_CHECK` makes `KMEANS_CHECK_LAUNCH` synchronise after **every** launch, so an
asynchronous fault is attributed to the kernel that caused it rather than to the next
synchronisation point several kernels later (CUDA-008). Neither belongs in a benchmark build: the
per-launch synchronisation alone would dominate the measurements.

The cell builds `kmeans_dbg`, `test_gpu_dbg` and `test_edge_dbg` with the debug flags of §15.3
part 3 — still `-O3`, still no `--use_fast_math`, because a sanitizer run on code the optimiser
never saw would not be checking the code that ships (NUM-002, BLD-002).

With `QUICK = True` this section and the sanitizer section below are skipped: they add several
minutes and they check the same kernels the correctness cells already ran.
