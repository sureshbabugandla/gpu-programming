// test_device.cuh — the device-side half of the test scaffolding, shared by test_gpu and
// test_edge.
//
// Separate from test_harness.h because this part needs the CUDA runtime: it exists so that a
// deliberately provoked device failure (EDGE-27, EDGE-28, an oversized configuration) is recorded
// as a test failure or an expected outcome instead of aborting the executable, which would stop
// every later case from reporting.

#ifndef KMEANS_TESTS_TEST_DEVICE_CUH
#define KMEANS_TESTS_TEST_DEVICE_CUH

#include <string>

#include "common.h"
#include "cuda_util.cuh"
#include "kmeans_gpu.h"
#include "test_harness.h"

namespace kmeans {
namespace tests {

inline const char *strategy_name(Strategy s)
{
    return s == Strategy::kAtomic ? "atomic" : "reduction";
}

inline const char *assign_name(AssignKernel a)
{
    return a == AssignKernel::kBaseline ? "baseline" : "smem";
}

// Runs the device driver, recording a CudaError as a failed check. Returns false when the run did
// not complete, so the caller can skip the comparisons that would read an empty Result.
inline bool run_device(const float *X, const float *C0, const gpu::Options &options,
                       Result *result, gpu::RunInfo *info, const gpu::LockstepHook &hook,
                       const std::string &what)
{
    try {
        *result = gpu::run(X, C0, options, info, hook);
        return true;
    } catch (const CudaError &err) {
        check(false, what + ": " + err.what() + " (exit code " +
                         std::to_string(err.exit_code()) + ")");
        return false;
    } catch (const std::exception &err) {
        check(false, what + ": " + err.what());
        return false;
    }
}

// The mirror image: a case that MUST fail, and must fail with a particular exit code. Used by the
// §9.2 rejection cases, where "it threw something" is not the claim being tested -- the claim is
// that it threw with the code BLD-005 assigns to that rule and said why.
template <typename Callable>
inline void check_rejected(Callable &&action, int expected_exit_code, const std::string &what,
                           const std::string &expected_substring = std::string())
{
    try {
        action();
        check(false, what + ": expected a rejection with exit code " +
                         std::to_string(expected_exit_code) + ", but the call succeeded");
    } catch (const CudaError &err) {
        check(err.exit_code() == expected_exit_code,
              what + ": exit code " + std::to_string(err.exit_code()) + ", want " +
                  std::to_string(expected_exit_code) + " (" + err.what() + ")");
        if (!expected_substring.empty()) {
            const std::string message = err.what();
            check(message.find(expected_substring) != std::string::npos,
                  what + ": the message should mention '" + expected_substring + "', got: " +
                      message);
        }
    } catch (const std::exception &err) {
        check(false, what + ": expected a CudaError, got " + err.what());
    }
}

}  // namespace tests
}  // namespace kmeans

#endif  // KMEANS_TESTS_TEST_DEVICE_CUH
