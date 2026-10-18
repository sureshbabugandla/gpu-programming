// test_harness.h — the check/report scaffolding shared by the three test drivers.
//
// Every driver has the same shape (requirements.md NFR-015): one executable, a PASS or FAILED line
// per case, a count of checks and failures at the end, and a non-zero exit on any failure, so a
// single notebook cell can gate the run. That scaffolding lived in each driver separately while
// there were two; with test_cpu, test_gpu and test_edge there are three, so it lives here.
//
// Header-only with C++17 inline variables: the counters are one object across translation units,
// which is what lets a driver split its cases over several files if it ever needs to.

#ifndef KMEANS_TESTS_TEST_HARNESS_H
#define KMEANS_TESTS_TEST_HARNESS_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "common.h"

namespace kmeans {
namespace tests {

inline int g_checks = 0;
inline int g_failures = 0;
inline int g_case_failures = 0;

inline void check(bool condition, const std::string &what)
{
    ++g_checks;
    if (!condition) {
        ++g_failures;
        ++g_case_failures;
        std::printf("      FAIL  %s\n", what.c_str());
    }
}

inline void check_eq_i64(int64_t got, int64_t want, const std::string &what)
{
    check(got == want, what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
}

// %.9g round-trips a binary32 and %.17g a binary64, so a reported value can be pasted back into
// a test as a literal without losing a bit.
inline std::string float_str(float v)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    return std::string(buf);
}

inline std::string double_str(double v)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

// Scope guard for the per-case PASS/FAIL line: the title is printed on entry and the verdict on
// exit, so a case that throws still reports.
class TestScope {
public:
    TestScope(const char *id, const std::string &title)
    {
        g_case_failures = 0;
        std::printf("  %-9s %s\n", id, title.c_str());
    }
    ~TestScope() { std::printf("      %s\n", g_case_failures == 0 ? "PASS" : "-> FAILED"); }

    TestScope(const TestScope &) = delete;
    TestScope &operator=(const TestScope &) = delete;
};

// Bitwise equality of two runs, used wherever the specification asks for identity rather than
// agreement within a tolerance: SMEM-004 (the tiled assignment kernel against the baseline) and
// TEST-R01 (Approach B repeated). Centroids are compared by bit pattern, not with ==, because
// -0.0 == 0.0 would hide a difference the two requirements do not permit. `why` receives the
// first difference found.
inline bool results_bit_equal(const Result &a, const Result &b, std::string *why)
{
    if (a.status != b.status) {
        *why = std::string("status ") + status_name(a.status) + " vs " + status_name(b.status);
        return false;
    }
    if (a.iterations != b.iterations) {
        *why = "iterations " + std::to_string(a.iterations) + " vs " +
               std::to_string(b.iterations);
        return false;
    }
    if (a.centroid_consistent != b.centroid_consistent) {
        *why = "the P2 flag differs";
        return false;
    }
    if (a.has_last_delta != b.has_last_delta ||
        std::memcmp(&a.last_delta, &b.last_delta, sizeof(double)) != 0) {
        *why = "Delta " + double_str(a.last_delta) + " vs " + double_str(b.last_delta);
        return false;
    }
    if (a.assignments.size() != b.assignments.size() || a.counts.size() != b.counts.size() ||
        a.centroids.size() != b.centroids.size()) {
        *why = "result array sizes differ";
        return false;
    }
    for (size_t i = 0; i < a.assignments.size(); ++i) {
        if (a.assignments[i] != b.assignments[i]) {
            *why = "assignment of point " + std::to_string(i) + ": " +
                   std::to_string(a.assignments[i]) + " vs " + std::to_string(b.assignments[i]);
            return false;
        }
    }
    for (size_t k = 0; k < a.counts.size(); ++k) {
        if (a.counts[k] != b.counts[k]) {
            *why = "count of cluster " + std::to_string(k) + ": " + std::to_string(a.counts[k]) +
                   " vs " + std::to_string(b.counts[k]);
            return false;
        }
    }
    for (size_t e = 0; e < a.centroids.size(); ++e) {
        if (std::memcmp(&a.centroids[e], &b.centroids[e], sizeof(float)) != 0) {
            *why = "centroid element " + std::to_string(e) + ": " + float_str(a.centroids[e]) +
                   " vs " + float_str(b.centroids[e]);
            return false;
        }
    }
    return true;
}

// The closing report every driver prints. Returns the process exit code.
inline int report_totals(const char *what)
{
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("RESULT: FAILED\n");
        return 1;
    }
    std::printf("RESULT: %s\n", what);
    return 0;
}

}  // namespace tests
}  // namespace kmeans

#endif  // KMEANS_TESTS_TEST_HARNESS_H
