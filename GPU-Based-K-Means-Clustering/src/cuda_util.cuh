// cuda_util.cuh — CUDA error checking (CUDA-008) and RAII ownership of device resources
// (ARC-04, MEM-007).
//
// Two rules this header exists to enforce:
//
//   * Every CUDA runtime call is checked, and a failure names the file, line, call text and
//     cudaGetErrorString (CUDA-008). Kernel launches are followed by cudaGetLastError, which
//     catches launch-configuration faults; asynchronous execution faults surface at the
//     per-iteration stream synchronisation, or immediately when KMEANS_SYNC_CHECK is defined.
//   * Every device and pinned allocation is owned by a destructor, so an error path that returns
//     early still frees it (MEM-007, verified by --leak-check full in TEST-S01).
//
// Failures are reported through a thrown CudaError rather than exit(): the caller (the driver or a
// test) decides whether to translate it into exit code 4, because a test needs to keep running
// after a deliberately provoked failure (EDGE-27, EDGE-28).

#ifndef KMEANS_CUDA_UTIL_CUH
#define KMEANS_CUDA_UTIL_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "common.h"

namespace kmeans {

// BLD-006 message shape, carried as an exception so RAII owners unwind on the way out.
class CudaError : public std::runtime_error {
public:
    CudaError(const std::string &message, int exit_code)
        : std::runtime_error(message), exit_code_(exit_code)
    {
    }
    int exit_code() const { return exit_code_; }

private:
    int exit_code_;
};

inline void cuda_throw(cudaError_t status, const char *call, const char *file, int line)
{
    const std::string message = std::string("CUDA call '") + call + "' failed at " + file + ":" +
                                std::to_string(line) + ": " + cudaGetErrorName(status) + ": " +
                                cudaGetErrorString(status);
    // Clear the sticky error so a test that provoked this deliberately can continue.
    cudaGetLastError();
    throw CudaError(message, kExitCudaError);
}

#define KMEANS_CUDA_CHECK(call)                                                   \
    do {                                                                          \
        const cudaError_t kmeans_status_ = (call);                                \
        if (kmeans_status_ != cudaSuccess) {                                      \
            kmeans::cuda_throw(kmeans_status_, #call, __FILE__, __LINE__);         \
        }                                                                         \
    } while (0)

// --- MEM-008 fault injection -------------------------------------------------------------------
//
// EDGE-27 (allocation failure despite the VAL-009 pre-check) and EDGE-28 (launch failure) cannot
// be provoked by any legal input: the pre-check is there precisely to make the first impossible,
// and a launch configuration this code computes is always valid. They are reached through two
// environment variables instead of by exhausting real memory, which keeps the test cheap, keeps
// it repeatable, and keeps the hook out of the way of every normal run:
//
//   KMEANS_FAIL_ALLOC=n       the n-th device allocation of the process fails (1-based)
//   KMEANS_FAIL_LAUNCH=name   the next launch of that kernel fails
//
// Both paths go through the ordinary CUDA-008 error handling, so what the tests observe is the
// real cleanup path: the RAII owners unwind, the message names the rule, and the exit code is 4.
// The variables are read once per process.

inline int64_t fault_injection_alloc_index()
{
    static const int64_t index = []() -> int64_t {
        const char *text = std::getenv("KMEANS_FAIL_ALLOC");
        if (text == nullptr) {
            return 0;
        }
        const long long value = std::strtoll(text, nullptr, 10);
        return value > 0 ? static_cast<int64_t>(value) : 0;
    }();
    return index;
}

inline const char *fault_injection_launch_name()
{
    static const char *const name = std::getenv("KMEANS_FAIL_LAUNCH");
    return name;
}

// True when this allocation is the one MEM-008 was asked to fail. The counter advances on every
// device allocation, so "the n-th" is counted in allocation order, which is the order of the
// MEM-001 table in kmeans_gpu.cu.
inline bool fault_injection_should_fail_alloc()
{
    const int64_t target = fault_injection_alloc_index();
    if (target == 0) {
        return false;
    }
    static int64_t seen = 0;
    return ++seen == target;
}

inline bool fault_injection_should_fail_launch(const char *kernel_name)
{
    const char *target = fault_injection_launch_name();
    return target != nullptr && kernel_name != nullptr && std::strcmp(target, kernel_name) == 0;
}

inline void check_injected_launch_failure(const char *kernel_name)
{
    if (fault_injection_should_fail_launch(kernel_name)) {
        throw CudaError(std::string("[MEM-008] launch of '") + kernel_name +
                            "' failed (fault injected by KMEANS_FAIL_LAUNCH)",
                        kExitCudaError);
    }
}

// CUDA-008: after every launch, cudaGetLastError reports launch errors. With KMEANS_SYNC_CHECK
// defined (the debug build of BLD-002) a synchronisation follows, so an asynchronous fault is
// attributed to the kernel that caused it instead of to the next synchronisation point.
//
// MEM-008: a kernel named by KMEANS_FAIL_LAUNCH raises here instead. The failure is reported as a
// launch error with exit code 4 and the same message shape as a real one, because what EDGE-28
// exercises is the handling, not the fault.
#ifdef KMEANS_SYNC_CHECK
#define KMEANS_CHECK_LAUNCH(name)                                                 \
    do {                                                                          \
        kmeans::check_injected_launch_failure(name);                              \
        KMEANS_CUDA_CHECK(cudaGetLastError());                                    \
        KMEANS_CUDA_CHECK(cudaDeviceSynchronize());                               \
    } while (0)
#else
#define KMEANS_CHECK_LAUNCH(name)                                                 \
    do {                                                                          \
        kmeans::check_injected_launch_failure(name);                              \
        KMEANS_CUDA_CHECK(cudaGetLastError());                                    \
    } while (0)
#endif

// --- device memory -----------------------------------------------------------------------------

// Owning, move-only device array. Copying is deleted because two owners would double-free.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) { allocate(count); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(size_t count)
    {
        release();
        if (count == 0) {
            return;  // a zero-element buffer is legal and stays null; no kernel indexes into it
        }
        // MEM-008: EDGE-27 is an allocation that fails although VAL-009 said it would fit. The
        // injected failure is raised before cudaMalloc, so the buffers allocated before this one
        // are still live and the unwind exercises the real MEM-007 cleanup path.
        if (fault_injection_should_fail_alloc()) {
            throw CudaError("[MEM-008] device allocation of " +
                                std::to_string(count * sizeof(T)) +
                                " B failed (fault injected by KMEANS_FAIL_ALLOC)",
                            kExitCudaError);
        }
        void *raw = nullptr;
        KMEANS_CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
        ptr_ = static_cast<T *>(raw);
        count_ = count;
    }

    void release()
    {
        if (ptr_ != nullptr) {
            // A free failure during unwinding must not throw again, so it is reported, not raised.
            const cudaError_t status = cudaFree(ptr_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "error[MEM-007]: cudaFree failed: %s\n",
                             cudaGetErrorString(status));
            }
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T *get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

    void copy_from_host(const T *src, size_t count, cudaStream_t stream)
    {
        if (count == 0) {
            return;
        }
        KMEANS_CUDA_CHECK(
            cudaMemcpyAsync(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host(T *dst, size_t count, cudaStream_t stream) const
    {
        if (count == 0) {
            return;
        }
        KMEANS_CUDA_CHECK(
            cudaMemcpyAsync(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    // Byte-pattern fill. Used for the zeroing of ATOM-001 / MEM-005 and, with 0xFF, for the
    // buffer poisoning of TEST-U06.
    void fill_bytes(int value, cudaStream_t stream)
    {
        if (ptr_ == nullptr || count_ == 0) {
            return;
        }
        KMEANS_CUDA_CHECK(cudaMemsetAsync(ptr_, value, count_ * sizeof(T), stream));
    }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

// Pinned host memory for the 16-byte per-iteration convergence copy (CUDA-011, MEM-001).
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count) { allocate(count); }
    ~PinnedBuffer() { release(); }

    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer &operator=(const PinnedBuffer &) = delete;

    void allocate(size_t count)
    {
        release();
        if (count == 0) {
            return;
        }
        void *raw = nullptr;
        KMEANS_CUDA_CHECK(cudaHostAlloc(&raw, count * sizeof(T), cudaHostAllocDefault));
        ptr_ = static_cast<T *>(raw);
        count_ = count;
    }

    void release()
    {
        if (ptr_ != nullptr) {
            const cudaError_t status = cudaFreeHost(ptr_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "error[MEM-007]: cudaFreeHost failed: %s\n",
                             cudaGetErrorString(status));
            }
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T *get() const { return ptr_; }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

// DEC-010: one stream realises every global phase boundary, so the stream is a run-scoped
// resource with the same ownership discipline as the buffers.
class CudaStream {
public:
    CudaStream() { KMEANS_CUDA_CHECK(cudaStreamCreate(&stream_)); }

    ~CudaStream()
    {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    cudaStream_t get() const { return stream_; }
    void synchronize() const { KMEANS_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_ = nullptr;
};

// PERF-005: T_alg is measured with CUDA events recorded on the stream.
class CudaEvent {
public:
    CudaEvent() { KMEANS_CUDA_CHECK(cudaEventCreate(&event_)); }

    ~CudaEvent()
    {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;

    void record(cudaStream_t stream) { KMEANS_CUDA_CHECK(cudaEventRecord(event_, stream)); }
    cudaEvent_t get() const { return event_; }

    // PERF-005: elapsed time is read only after synchronising on the end event.
    static float elapsed_ms(const CudaEvent &start, const CudaEvent &end)
    {
        KMEANS_CUDA_CHECK(cudaEventSynchronize(end.event_));
        float ms = 0.0f;
        KMEANS_CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
        return ms;
    }

private:
    cudaEvent_t event_ = nullptr;
};

}  // namespace kmeans

#endif  // KMEANS_CUDA_UTIL_CUH
