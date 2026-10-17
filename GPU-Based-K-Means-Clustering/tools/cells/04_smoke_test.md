## 4. Toolchain smoke test

A nine-line kernel is cheaper to debug than a multi-file build, so before any program source is
compiled this section writes `hello.cu`, builds it with the **same release flags the real build
uses** (BLD-002, so a bad flag is caught here), runs it, and checks three things:

1. `nvcc` accepts `-arch=$SM_ARCH` for the attached device and the binary links and runs;
2. a kernel launches and its writes are visible to the host (`kernel.check=OK`, verified
   element-wise, not just by a checksum);
3. the compute capability the device reports to host code equals the one `nvidia-smi` reported in
   section 2 — if these disagree, `SM_ARCH` was derived from the wrong device.

The cell raises on a non-zero compile or run exit code, on a mismatch, or on a missing `OK`, so
"restart kernel and run all" halts at the first toolchain fault (NFR-014). It also prints the SM
count, shared memory per block, warp size and the CUDA runtime and driver versions — the
`cudaGetDeviceProperties` fields that `nvidia-smi` does not expose, which CUDA-001 requires the
program to query, MEM-004 and SMEM-003 check tile sizes against, and CUDA-007 uses to size grids.
Whatever the runtime reports is printed verbatim; no CUDA version is assumed.
