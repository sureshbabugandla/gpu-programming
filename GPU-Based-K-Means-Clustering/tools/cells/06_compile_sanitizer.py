# Sanitizer build — BLD-002 debug flags: -lineinfo for source attribution, KMEANS_SYNC_CHECK so
# an asynchronous fault is reported at the launch that caused it (CUDA-008).
if "NVCC_DEBUG_FLAGS" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")
if "compile_cuda" not in globals():
    raise RuntimeError("Run the CUDA compile cell (section 6b) first: it defines compile_cuda, "
                       "CUDA_SOURCES and HOST_SOURCES, which this cell reuses so that the release "
                       "and debug builds cannot drift apart.")

if QUICK:
    print("QUICK = True: the sanitizer build is skipped. Set QUICK = False in section 3 to run "
          "TEST-S01..S04.")
else:
    compile_cuda("kmeans_dbg", ["kmeans_main.cu"] + CUDA_SOURCES + HOST_SOURCES,
                 flags=NVCC_DEBUG_FLAGS)
    compile_cuda("test_gpu_dbg", ["test_gpu.cu"] + CUDA_SOURCES + HOST_SOURCES,
                 flags=NVCC_DEBUG_FLAGS)
    compile_cuda("test_edge_dbg", ["test_edge.cu"] + CUDA_SOURCES + HOST_SOURCES,
                 flags=NVCC_DEBUG_FLAGS)
    print("\nsanitizer build clean: kmeans_dbg, test_gpu_dbg, test_edge_dbg")
