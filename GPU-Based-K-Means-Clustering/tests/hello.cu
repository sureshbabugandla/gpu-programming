// Toolchain smoke test (requirements.md §15.3 part 4).
//
// Proves, before any program source is compiled, that the SM_ARCH derived from ENV-01 matches the
// attached device, that the toolkit reported by nvcc can build and link for it, and that a kernel
// launches and its writes are visible to the host. It also prints the device properties that
// nvidia-smi does not expose (SM count, shared memory per block) and that CUDA-001 requires the
// program itself to query, plus the CUDA runtime and driver versions exactly as reported — no
// version is assumed anywhere.
//
// Output is `key=value` lines so the notebook cell can assert on them. Exit codes follow BLD-005:
// 0 success, 4 CUDA runtime error.

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// CUDA-008: every runtime call is checked; failure reports file, line, call and the error string.
#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "error[CUDA-008]: %s:%d: %s failed: %s\n", __FILE__,          \
                         __LINE__, #call, cudaGetErrorString(err_));                           \
            std::exit(4);                                                                      \
        }                                                                                     \
    } while (0)

static const int kElements = 256;

// One thread per element, grid-stride (CUDA-007), bounds-checked (CUDA-005). The value is i*i so
// the host can verify every element independently of launch geometry.
__global__ void square_indices(int *out, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = i * i;
    }
}

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int runtime_version = 0;
    int driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    std::printf("device.name=%s\n", prop.name);
    std::printf("device.compute_capability=%d.%d\n", prop.major, prop.minor);
    std::printf("device.sm_count=%d\n", prop.multiProcessorCount);
    std::printf("device.global_mem_bytes=%zu\n", prop.totalGlobalMem);
    std::printf("device.shared_mem_per_block_bytes=%zu\n", prop.sharedMemPerBlock);
    std::printf("device.shared_mem_per_block_optin_bytes=%d\n", prop.sharedMemPerBlockOptin);
    std::printf("device.warp_size=%d\n", prop.warpSize);
    std::printf("device.max_threads_per_block=%d\n", prop.maxThreadsPerBlock);
    std::printf("runtime.cuda_version=%d\n", runtime_version);
    std::printf("driver.cuda_version=%d\n", driver_version);

    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, kElements * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out, 0, kElements * sizeof(int)));

    square_indices<<<4, 64>>>(d_out, kElements);
    CUDA_CHECK(cudaGetLastError());  // launch error (CUDA-008)

    int host_out[kElements];
    CUDA_CHECK(cudaMemcpy(host_out, d_out, sizeof(host_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    long long sum = 0;
    long long expected_sum = 0;
    int mismatches = 0;
    for (int i = 0; i < kElements; ++i) {
        sum += host_out[i];
        expected_sum += static_cast<long long>(i) * i;
        if (host_out[i] != i * i) {
            ++mismatches;
        }
    }

    std::printf("kernel.elements=%d\n", kElements);
    std::printf("kernel.sum=%lld\n", sum);
    std::printf("kernel.expected_sum=%lld\n", expected_sum);
    std::printf("kernel.mismatches=%d\n", mismatches);
    std::printf("kernel.check=%s\n", (mismatches == 0 && sum == expected_sum) ? "OK" : "FAIL");

    return (mismatches == 0 && sum == expected_sum) ? 0 : 1;
}
