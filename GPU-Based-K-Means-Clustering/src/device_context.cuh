// device_context.cuh — device selection, capability record and capacity checks (ARC-04,
// CUDA-001, VAL-009, MEM-002, MEM-004, SMEM-003, RED-006/007).
//
// Everything here runs before the first kernel launch. The order matters: the device is queried,
// the capability gate of CUDA-001 is applied, the chunk count G of RED-006 is chosen from the free
// memory actually reported, and only then is the MEM-002 total compared with free memory
// (VAL-009). No allocation happens until all of that holds, which is what lets EDGE-26 fail with
// exit code 3 and a byte count instead of a CUDA out-of-memory error.
//
// Header-only: it is pure host code over the CUDA runtime API, so inlining it avoids a translation
// unit that exists only to call cudaGetDeviceProperties.

#ifndef KMEANS_DEVICE_CONTEXT_CUH
#define KMEANS_DEVICE_CONTEXT_CUH

#include <cuda_runtime.h>

#include <string>

#include "common.h"
#include "cuda_util.cuh"
#include "launch_config.h"

namespace kmeans {

// CUDA-001: the properties recorded at startup. These are the same fields the environment record
// of PERF-008 carries, so a benchmark row can be traced to the device that produced it.
struct DeviceInfo {
    int device = 0;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    int max_threads_per_block = 0;
    int warp_size = 0;
    size_t shared_mem_per_block = 0;      // default limit
    size_t shared_mem_per_block_optin = 0;  // opt-in maximum
    size_t total_global_mem = 0;
    size_t free_global_mem = 0;
    int runtime_version = 0;
    int driver_version = 0;

    double compute_capability() const { return cc_major + cc_minor / 10.0; }
    // DEC-007 / ATOM-003: native atomicAdd(double) exists from compute capability 6.0.
    bool has_native_fp64_atomics() const { return compute_capability() >= 6.0; }
};

inline DeviceInfo query_device(int device)
{
    DeviceInfo info;
    int count = 0;
    KMEANS_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count <= 0) {
        throw CudaError("no CUDA device is visible (CUDA-001)", kExitUnsupportedDevice);
    }
    if (device < 0 || device >= count) {
        throw CudaError("device " + std::to_string(device) + " does not exist; " +
                            std::to_string(count) + " device(s) visible",
                        kExitUnsupportedDevice);
    }
    KMEANS_CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    KMEANS_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    info.device = device;
    info.name = prop.name;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    info.shared_mem_per_block = prop.sharedMemPerBlock;
    info.shared_mem_per_block_optin = prop.sharedMemPerBlockOptin;
    info.total_global_mem = prop.totalGlobalMem;
    KMEANS_CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    KMEANS_CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));

    size_t free_bytes = 0, total_bytes = 0;
    KMEANS_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    info.free_global_mem = free_bytes;

    // CUDA-001: reject below compute capability 6.0 with exit code 3. Below 6.0 there is no native
    // atomicAdd(double), which ATOM-003 requires and for which this project has no CAS fallback.
    if (info.compute_capability() < 6.0) {
        throw CudaError("device '" + info.name + "' has compute capability " +
                            std::to_string(info.cc_major) + "." + std::to_string(info.cc_minor) +
                            ", below the 6.0 minimum of CUDA-001 (native atomicAdd(double) is "
                            "required by ATOM-003)",
                        kExitUnsupportedDevice);
    }
    return info;
}

inline std::string describe_device(const DeviceInfo &info)
{
    return info.name + " (cc " + std::to_string(info.cc_major) + "." +
           std::to_string(info.cc_minor) + ", " + std::to_string(info.sm_count) + " SMs, " +
           std::to_string(info.free_global_mem / (1024 * 1024)) + " MiB free of " +
           std::to_string(info.total_global_mem / (1024 * 1024)) + " MiB, shared/block " +
           std::to_string(info.shared_mem_per_block) + " B)";
}

// MEM-004, SMEM-003: every per-block shared-memory request is checked against the device limit at
// startup rather than discovered as a launch failure. The movement and assign reductions scale
// with the block size, so the check depends on the validated block size.
inline void check_shared_memory(const DeviceInfo &info, int block_size, int tile_k, int tile_d)
{
    const int64_t limit = static_cast<int64_t>(info.shared_mem_per_block);
    struct Request {
        const char *rule;
        const char *what;
        int64_t bytes;
    };
    const Request requests[] = {
        {"MEM-004", "assign change-count reduction", block_reduction_bytes(block_size)},
        {"MEM-004", "movement block maximum reduction", block_reduction_bytes(block_size)},
        // max_reduce sizes its single block from the number of per-block maxima, not from the
        // configured block size, so it is checked at its own maximum rather than at block_size.
        {"MEM-004", "max_reduce final reduction", block_reduction_bytes(kMaxMovementBlocks)},
        {"MEM-004", "reduce_stage2 sums and counts", stage2_shared_bytes()},
        {"SMEM-003", "assign_smem centroid tile", assign_smem_bytes(tile_k, tile_d)},
        // assign_smem asks for the tile and its change-count reduction in one allocation, so the
        // sum is what the launch actually requests and what has to fit.
        {"SMEM-003", "assign_smem tile and change-count reduction",
         assign_smem_shared_bytes(tile_k, tile_d, block_size)},
    };
    for (const Request &r : requests) {
        if (r.bytes > limit) {
            throw CudaError(std::string("[") + r.rule + "] the " + r.what + " needs " +
                                std::to_string(r.bytes) +
                                " B of shared memory per block, but the device offers only " +
                                std::to_string(limit) + " B",
                            kExitInvalidInput);
        }
    }
}

// Bytes of MEM-001 other than the Approach B partial buffers P and Q. This is the `other_bytes`
// input of the MEM-003 reduction budget: the budget is a quarter of what is left after the
// buffers every run needs.
inline uint64_t non_reduction_bytes(int64_t N, int64_t D, int64_t K, int64_t movement_blocks)
{
    const MemoryModel m = device_memory_bytes(N, D, K, 0, movement_blocks);
    return m.overflow ? UINT64_MAX : m.total;
}

// RED-006 / RED-007 for a concrete device. Approach A passes `strategy == kAtomic` and gets G = 0,
// since it allocates no partial buffers.
inline ChunkSelection select_chunks(const DeviceInfo &info, const Config &config,
                                    uint64_t user_cap_bytes)
{
    ChunkSelection sel;
    if (config.strategy != Strategy::kReduction) {
        sel.G = 0;
        return sel;
    }
    const int movement_blocks = movement_grid_size(config.K, config.block_size, info.sm_count);
    const uint64_t other = non_reduction_bytes(config.N, config.D, config.K, movement_blocks);
    sel = choose_reduce_chunks(config.N, config.D, config.K, info.sm_count,
                               static_cast<uint64_t>(info.free_global_mem), other, user_cap_bytes,
                               config.reduce_chunks);
    if (!sel.ok) {
        // RED-007: fail before allocation with the required byte count, as exit code 3.
        throw CudaError("[RED-007] " + sel.message, kExitUnsupportedDevice);
    }
    return sel;
}

// VAL-009 against the free memory of this device, with the selected G included in the model.
inline void check_capacity(const DeviceInfo &info, const Config &config, int64_t reduce_chunks)
{
    const int movement_blocks = movement_grid_size(config.K, config.block_size, info.sm_count);
    const MemoryModel m =
        device_memory_bytes(config.N, config.D, config.K, reduce_chunks, movement_blocks);
    if (m.overflow) {
        throw CudaError("[VAL-009] required device memory overflows 64-bit arithmetic for N=" +
                            std::to_string(config.N) + " D=" + std::to_string(config.D) +
                            " K=" + std::to_string(config.K) +
                            " G=" + std::to_string(reduce_chunks),
                        kExitInvalidInput);
    }
    const uint64_t budget =
        static_cast<uint64_t>(static_cast<double>(info.free_global_mem) * 0.95);
    if (m.total > budget) {
        throw CudaError("[VAL-009] required " + std::to_string(m.total) +
                            " B exceeds the usable " + std::to_string(budget) + " B (95 % of " +
                            std::to_string(info.free_global_mem) + " B free on " + info.name + ")",
                        kExitUnsupportedDevice);
    }
}

}  // namespace kmeans

#endif  // KMEANS_DEVICE_CONTEXT_CUH
