// test_cpu.cpp — CPU-level test driver for milestone M1.
//
// Covers requirements.md §16.0 (H1-H6 and H2-a..e against the CPU reference), §16.3 (the FP32 and
// FP64 distance paths agree except on near-ties), TEST-U08 (movement and the inclusive tolerance
// boundary), TEST-U09 (empty clusters), TEST-U11 (size and overflow arithmetic), TEST-U12 (file
// I/O) and TEST-U13 (generator determinism).
//
// One executable, non-zero exit on any failure, so a single notebook cell can gate the run
// (NFR-015). GPU-level versions of these cases arrive in later milestones.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "cpu_reference.h"
#include "gen.h"
#include "hand_data.h"
#include "io_kmd.h"
#include "validate.h"

using namespace kmeans;

namespace {

int g_checks = 0;
int g_failures = 0;
int g_case_failures = 0;

void check(bool condition, const std::string &what)
{
    ++g_checks;
    if (!condition) {
        ++g_failures;
        ++g_case_failures;
        std::printf("      FAIL  %s\n", what.c_str());
    }
}

void check_eq_i64(int64_t got, int64_t want, const std::string &what)
{
    check(got == want, what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
}

std::string float_str(float v)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    return std::string(buf);
}

std::string double_str(double v)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

// Scope guard for the per-test PASS/FAIL line.
class TestScope {
public:
    TestScope(const char *id, const std::string &title) : id_(id)
    {
        g_case_failures = 0;
        std::printf("  %-7s %s\n", id, title.c_str());
    }
    ~TestScope()
    {
        std::printf("      %s\n", g_case_failures == 0 ? "PASS" : "-> FAILED");
    }

private:
    const char *id_;
};

// --- §16.0 hand datasets -----------------------------------------------------------------------

void test_hand_cases()
{
    for (const tests::HandCase &c : tests::hand_cases()) {
        TestScope scope("H-case", c.id + (c.note.empty() ? "" : "  (" + c.note + ")"));

        // Validation first, because two of the cases exist to exercise VAL-003 and VAL-008.
        const Validation params = validate_parameters(c.N, c.D, c.K, c.max_iter, c.tol);
        check(params.ok, c.id + ": parameters accepted");
        double m_range = 0.0;
        const Validation data = validate_data(c.X, c.N, c.D, c.C0, c.K, &m_range);
        check(data.ok, c.id + ": data accepted");
        check(params.has_warning("W-KGTN") == c.expect_warning_kgtn,
              c.id + ": W-KGTN warning " + (c.expect_warning_kgtn ? "expected" : "not expected"));
        check(data.has_warning("W-DUPC") == c.expect_warning_dupc,
              c.id + ": W-DUPC warning " + (c.expect_warning_dupc ? "expected" : "not expected"));

        const Result r = cpu::run(c.X.data(), c.N, c.D, c.C0.data(), c.K, c.max_iter, c.tol,
                                  m_range, false);

        check_eq_i64(r.iterations, c.expected_iterations, c.id + ": iteration count");
        check(r.status == c.expected_status,
              c.id + ": status: got " + status_name(r.status) + ", want " +
                  status_name(c.expected_status));
        check(r.centroid_consistent == c.expected_centroid_consistent,
              c.id + ": P2 flag: got " + (r.centroid_consistent ? "true" : "false") + ", want " +
                  (c.expected_centroid_consistent ? "true" : "false"));
        check(r.assignment_consistent, c.id + ": P1 flag true by construction");

        for (int64_t i = 0; i < c.N; ++i) {
            check_eq_i64(r.assignments[static_cast<size_t>(i)], c.expected_A[static_cast<size_t>(i)],
                         c.id + ": A[" + std::to_string(i) + "]");
        }
        for (int64_t k = 0; k < c.K; ++k) {
            check_eq_i64(r.counts[static_cast<size_t>(k)], c.expected_n[static_cast<size_t>(k)],
                         c.id + ": n[" + std::to_string(k) + "]");
        }
        const double s_max = s_max_from_range(m_range);
        for (int64_t e = 0; e < c.K * c.D; ++e) {
            const float got = r.centroids[static_cast<size_t>(e)];
            const float want = c.expected_C[static_cast<size_t>(e)];
            const bool ok = c.centroids_exact ? (got == want) : centroid_agrees(got, want, s_max);
            check(ok, c.id + ": C[" + std::to_string(e) + "]: got " + float_str(got) + ", want " +
                          float_str(want) + (c.centroids_exact ? " (exact)" : " (NUM-006)"));
        }

        // Empty clusters must retain their initial centroid bit-exactly (DEC-003). This holds for
        // the H cases because no empty cluster is ever re-populated in them.
        for (int64_t k = 0; k < c.K; ++k) {
            if (c.expected_n[static_cast<size_t>(k)] != 0) {
                continue;
            }
            for (int64_t j = 0; j < c.D; ++j) {
                const size_t e = static_cast<size_t>(k * c.D + j);
                check(r.centroids[e] == c.C0[e],
                      c.id + ": empty cluster " + std::to_string(k) + " retained C0[" +
                          std::to_string(j) + "] bit-exactly");
            }
        }
    }
}

// --- TEST-U08: movement and the inclusive tolerance boundary -----------------------------------

void test_u08_movement()
{
    {
        TestScope scope("U08", "movement: exact delta, empty cluster excluded");
        // Cluster 0 moves (0,0) -> (3,4), a 3-4-5 triangle, so delta is exactly 5 in binary64.
        // Cluster 1 does not move. Cluster 2 is empty: its delta is 0 by construction and it is
        // excluded from the maximum, so a 0/0 division would show up here as NaN or 0.
        const std::vector<float> c_cur = {0.0f, 0.0f, 1.0f, 1.0f, 5.0f, 5.0f};
        const std::vector<float> c_next = {3.0f, 4.0f, 1.0f, 1.0f, 5.0f, 5.0f};
        const std::vector<uint32_t> n = {2, 1, 0};
        const double delta = cpu::movement(c_cur.data(), c_next.data(), n.data(), 3, 2);
        check(delta == 5.0, "delta: got " + double_str(delta) + ", want exactly 5");

        // With every cluster empty the maximum is over an empty set, which §2.6 cannot reach
        // because N >= 1, but the function must still return 0 rather than a NaN.
        const std::vector<uint32_t> none = {0, 0, 0};
        const double delta_none = cpu::movement(c_cur.data(), c_next.data(), none.data(), 3, 2);
        check(delta_none == 0.0, "delta with no non-empty cluster is 0");
    }

    {
        TestScope scope("U08", "inclusive comparison at delta == tol (DEC-002, ENV-10)");
        const std::vector<float> X = tests::h2_points();
        const std::vector<float> C0 = tests::h2_initial_centroids();
        double m_range = 0.0;
        const Validation v = validate_data(X, 4, 1, C0, 2, &m_range);
        check(v.ok, "H2 data accepted");

        // The first update of H2 moves a centroid by exactly 2. With tol = 2 the inclusive
        // comparison stops there; one ulp below 2 it must not.
        const Result at_boundary =
            cpu::run(X.data(), 4, 1, C0.data(), 2, 10, 2.0, m_range, false);
        check(at_boundary.status == Status::CONVERGED_MOVEMENT,
              std::string("tol == delta: status ") + status_name(at_boundary.status));
        check_eq_i64(at_boundary.iterations, 1, "tol == delta: iterations");

        const double just_below = std::nextafter(2.0, 0.0);
        const Result below =
            cpu::run(X.data(), 4, 1, C0.data(), 2, 10, just_below, m_range, false);
        check(below.status == Status::CONVERGED_MOVEMENT,
              std::string("tol one ulp below delta: status ") + status_name(below.status));
        check_eq_i64(below.iterations, 2, "tol one ulp below delta: iterations (not 1)");
        check(below.last_delta == 1.5, "tol one ulp below delta: last delta is the second one");
    }

    {
        TestScope scope("U08", "tol = 0 means bit-identical centroids");
        // H1 converges by assignment stability at t = 1 even though its delta (0.5) exceeds 0,
        // which is what DEC-002's "assignment stability takes precedence" means in practice.
        const std::vector<float> X = {0.0f, 1.0f, 9.0f, 10.0f};
        const std::vector<float> C0 = {0.0f, 10.0f};
        const Result r = cpu::run(X.data(), 4, 1, C0.data(), 2, 10, 0.0, 10.0, false);
        check(r.status == Status::CONVERGED_ASSIGNMENTS,
              std::string("status ") + status_name(r.status));
        check(r.last_delta == 0.5, "last delta: got " + double_str(r.last_delta) + ", want 0.5");
    }

    {
        TestScope scope("U08", "fixed-iteration mode disables both criteria (FR-020)");
        const std::vector<float> X = tests::h2_points();
        const std::vector<float> C0 = tests::h2_initial_centroids();
        const Result r = cpu::run(X.data(), 4, 1, C0.data(), 2, 7, 1e9, 10.0, true);
        check_eq_i64(r.iterations, 7, "exactly max_iter updates run");
        check(r.status == Status::ITERATION_LIMIT, std::string("status ") + status_name(r.status));
    }
}

// --- TEST-U09: empty clusters ------------------------------------------------------------------

void test_u09_empty_clusters()
{
    {
        TestScope scope("U09", "empty cluster at the update step retains its centroid");
        // The mid-iteration case of EDGE-14 at step level: cluster 1 held points in an earlier
        // accumulation and holds none now. update() must copy its centroid through untouched and
        // never divide, and movement() must report only cluster 0's displacement.
        const std::vector<float> c_cur = {1.0f, 7.25f};
        const std::vector<double> S = {11.0, 0.0};
        const std::vector<uint32_t> n = {4, 0};
        std::vector<float> c_next(2, -1.0f);
        cpu::update(S.data(), n.data(), 2, 1, c_cur.data(), c_next.data());

        check(c_next[0] == 2.75f, "non-empty cluster: 11/4 = 2.75 exactly, got " +
                                      float_str(c_next[0]));
        check(c_next[1] == 7.25f, "empty cluster retained bit-exactly, got " +
                                      float_str(c_next[1]));
        check(std::isfinite(c_next[1]), "empty cluster is not NaN or Inf (no 0/0 division)");

        const double delta = cpu::movement(c_cur.data(), c_next.data(), n.data(), 2, 1);
        check(delta == 1.75, "delta is cluster 0's 1.75 only, got " + double_str(delta));
    }

    {
        TestScope scope("U09", "single-point cluster is bit-exact (EDGE-15)");
        // Division by 1 is exact in binary64 and the quotient is representable, so the centroid
        // must equal the point bit for bit, not merely within NUM-006.
        const std::vector<float> X = {0.1f, 1234.5678f, -7.25f};
        const std::vector<float> C0 = {0.1f, 1234.5678f, -7.25f};
        const std::vector<int32_t> A = {0, 1, 2};
        std::vector<double> S(3, 0.0);
        std::vector<uint32_t> n(3, 0);
        cpu::accumulate(X.data(), 3, 1, A.data(), 3, S.data(), n.data());
        std::vector<float> C_next(3, 0.0f);
        cpu::update(S.data(), n.data(), 3, 1, C0.data(), C_next.data());
        for (int k = 0; k < 3; ++k) {
            check(C_next[static_cast<size_t>(k)] == X[static_cast<size_t>(k)],
                  "cluster " + std::to_string(k) + " equals its only point bit-exactly");
            check_eq_i64(n[static_cast<size_t>(k)], 1, "count of cluster " + std::to_string(k));
        }
    }

    {
        TestScope scope("U09", "K > N leaves at least K - N clusters empty (EDGE-04)");
        const std::vector<float> X = {4.0f};
        const std::vector<float> C0 = {0.0f, 1.0f, 2.0f};
        const Result r = cpu::run(X.data(), 1, 1, C0.data(), 3, 10, 0.0, 4.0, false);
        int empty = 0;
        for (uint32_t count : r.counts) {
            if (count == 0) {
                ++empty;
            }
        }
        check(empty >= 2, "at least K - N = 2 clusters empty, got " + std::to_string(empty));
        check(r.centroids[0] == 0.0f && r.centroids[1] == 1.0f,
              "both empty clusters retained their initial centroids");
    }
}

// --- TEST-U11: size and overflow arithmetic ----------------------------------------------------

void test_u11_size_calculator()
{
    {
        TestScope scope("U11", "MEM-002 byte counts");
        // N=1000, D=32, K=64, Approach A (G = 0), 1024 movement blocks:
        //   8ND = 256000, 4N = 4000, 16KD = 32768, 4K = 256, scalars = 1024*8 + 32 = 8224
        const MemoryModel m = device_memory_bytes(1000, 32, 64, 0, 1024);
        check(!m.overflow, "no overflow for a modest configuration");
        check_eq_i64(static_cast<int64_t>(m.x_col), 128000, "x_col bytes (4ND)");
        check_eq_i64(static_cast<int64_t>(m.x_row), 128000, "x_row bytes (4ND)");
        check_eq_i64(static_cast<int64_t>(m.centroids), 16384, "centroid bytes (8KD)");
        check_eq_i64(static_cast<int64_t>(m.assignments), 4000, "assignment bytes (4N)");
        check_eq_i64(static_cast<int64_t>(m.sums), 16384, "sum bytes (8KD)");
        check_eq_i64(static_cast<int64_t>(m.counts), 256, "count bytes (4K)");
        check_eq_i64(static_cast<int64_t>(m.partials), 0, "no P buffer for Approach A");
        check_eq_i64(static_cast<int64_t>(m.partial_counts), 0, "no Q buffer for Approach A");
        check_eq_i64(static_cast<int64_t>(m.total), 128000 + 128000 + 16384 + 4000 + 16384 + 256 +
                                                        8224,
                     "total bytes");
    }

    {
        TestScope scope("U11", "Approach B adds 8GKD + 4GK");
        const MemoryModel a = device_memory_bytes(1000, 32, 64, 0, 1024);
        const MemoryModel b = device_memory_bytes(1000, 32, 64, 7, 1024);
        check_eq_i64(static_cast<int64_t>(b.partials), 8 * 7 * 64 * 32, "P buffer bytes (8GKD)");
        check_eq_i64(static_cast<int64_t>(b.partial_counts), 4 * 7 * 64, "Q buffer bytes (4GK)");
        check_eq_i64(static_cast<int64_t>(b.total - a.total), 8 * 7 * 64 * 32 + 4 * 7 * 64,
                     "difference is exactly the P and Q buffers");
    }

    {
        TestScope scope("U11", "overflow is detected, not wrapped (CUDA-006, EDGE-30)");
        uint64_t out = 0;
        check(checked_mul(UINT64_MAX / 2, 2, &out), "a product that just fits is accepted");
        check(!checked_mul(UINT64_MAX / 2 + 1, 2, &out), "a product that overflows is rejected");
        check(!checked_add(UINT64_MAX, 1, &out), "a sum that overflows is rejected");
        check(checked_mul(0, UINT64_MAX, &out) && out == 0, "zero times anything is zero");

        const int64_t huge = 2147483647;  // 2^31 - 1, the VAL-002 maximum
        const MemoryModel m = device_memory_bytes(huge, huge, huge, huge, 1024);
        check(m.overflow, "N = D = K = G = 2^31-1 overflows the capacity model");
        check_eq_i64(static_cast<int64_t>(m.total), 0, "an overflowing model reports no total");

        const Validation v = validate_capacity(huge, huge, huge, huge, 16ull * 1024 * 1024 * 1024);
        check(!v.ok && v.exit_code == kExitInvalidInput,
              "VAL-009 rejects the overflowing configuration with exit 2");
    }

    {
        TestScope scope("U11", "VAL-009 compares against 95 % of free memory");
        // 15 GB free, as on the recorded Tesla T4 (§1.6.2).
        const uint64_t free_bytes = 15637086208ull;
        const Validation ok = validate_capacity(1000000, 32, 64, 0, free_bytes);
        check(ok.ok, "the PERF-003 base point fits");

        const MemoryModel m = device_memory_bytes(1000000, 32, 64, 0);
        const Validation tight = validate_capacity(1000000, 32, 64, 0, m.total);
        check(!tight.ok && tight.exit_code == kExitUnsupportedDevice,
              "exactly the required bytes free is rejected by the 5 % margin with exit 3");
    }

    {
        TestScope scope("U11", "VAL-001..VAL-010 reject what they must");
        check(!validate_parameters(0, 1, 1, 10, 0.0).ok, "N = 0 rejected (VAL-001, EDGE-01)");
        check(!validate_parameters(1, 0, 1, 10, 0.0).ok, "D = 0 rejected (VAL-001, EDGE-02)");
        check(!validate_parameters(1, 1, 0, 10, 0.0).ok, "K = 0 rejected (VAL-001, EDGE-03)");
        check(validate_parameters(1, 1, 1, 0, 0.0).ok, "max_iter = 0 accepted (EDGE-22)");
        check(!validate_parameters(1, 1, 1, -1, 0.0).ok, "negative max_iter rejected (VAL-004)");
        check(!validate_parameters(1, 1, 1, 10, -1e-30).ok, "negative tol rejected (VAL-005)");
        check(!validate_parameters(1, 1, 1, 10, std::nan("")).ok, "NaN tol rejected (VAL-005)");
        check(!validate_parameters(1, 1, 1, 10, HUGE_VAL).ok, "infinite tol rejected (VAL-005)");
        check(validate_parameters(2, 1, 5, 10, 0.0).has_warning("W-KGTN"),
              "K > N warns W-KGTN and continues (VAL-003)");
        check(validate_parameters(1, 1, 1, 2147483648ll, 0.0).ok == false,
              "max_iter above 2^31-1 rejected (VAL-004)");
        check(validate_parameters(2147483648ll, 1, 1, 10, 0.0).ok == false,
              "N above 2^31-1 rejected (VAL-002)");

        // VAL-006 and VAL-007 on the data.
        double m_range = 0.0;
        const std::vector<float> good = {1.0f, 2.0f};
        const std::vector<float> c_one = {1.0f};
        const std::vector<float> with_nan = {1.0f, std::nanf("")};
        const std::vector<float> with_inf = {1.0f, HUGE_VALF};
        check(validate_data(good, 2, 1, c_one, 1, &m_range).ok, "finite data accepted");
        check(m_range == 2.0, "M is the largest magnitude, got " + double_str(m_range));
        const Validation nan_v = validate_data(with_nan, 2, 1, c_one, 1, &m_range);
        check(!nan_v.ok && nan_v.rule == "VAL-006", "NaN in data rejected (EDGE-20)");
        const Validation inf_v = validate_data(with_inf, 2, 1, c_one, 1, &m_range);
        check(!inf_v.ok && inf_v.rule == "VAL-006", "Inf in data rejected (EDGE-20)");

        // VAL-007: 4 D M^2 <= 2^126. With D = 1 the largest admissible M is 2^62, and a value an
        // octave above it must be refused.
        const float at_bound = static_cast<float>(std::ldexp(1.0, 62));
        const std::vector<float> big = {at_bound};
        check(validate_data(big, 1, 1, big, 1, &m_range).ok, "M = 2^62 with D = 1 is accepted");
        const float over = static_cast<float>(std::ldexp(1.0, 63));
        const std::vector<float> too_big = {over};
        const Validation range_v = validate_data(too_big, 1, 1, too_big, 1, &m_range);
        check(!range_v.ok && range_v.rule == "VAL-007", "M = 2^63 with D = 1 is rejected");

        // VAL-008 duplicate centroids warn but do not fail.
        const std::vector<float> dup = {5.0f, 5.0f};
        const Validation dup_v = validate_data(good, 2, 1, dup, 2, &m_range);
        check(dup_v.ok && dup_v.has_warning("W-DUPC"), "duplicate centroids warn W-DUPC");

        // VAL-010 option ranges.
        check(validate_options(256, 0, 8, 32, 1024, 49152).ok, "the defaults are accepted");
        check(!validate_options(200, 0, 8, 32, 1024, 49152).ok, "block size 200 is not a multiple "
                                                                "of 32");
        check(!validate_options(2048, 0, 8, 32, 1024, 49152).ok, "block size above 1024 rejected");
        check(!validate_options(256, -1, 8, 32, 1024, 49152).ok, "negative G rejected");
        check(!validate_options(256, 0, 8, 48, 1024, 49152).ok, "tile_d not a multiple of 32");
        check(!validate_options(256, 0, 1024, 32, 1024, 49152).ok,
              "tiles larger than the shared-memory limit rejected (EDGE-34)");
    }
}

// --- TEST-U12: file I/O ------------------------------------------------------------------------

void test_u12_file_io()
{
    TestScope scope("U12", "BLD-004 .kmd/.kma round trip and rejection of damaged files");

    check(host_is_little_endian(), "host is little-endian, as BLD-004 requires");

    const std::string path = "test_u12.kmd";
    const std::vector<float> values = {0.0f, 1.5f, -2.25f, 1e30f, 10.0f, 0.1f};
    std::string err;
    check(write_kmd(path, 3, 2, values.data(), &err) == kExitOk, "write a 3x2 .kmd: " + err);

    Matrix m;
    check(read_kmd(path, &m, &err) == kExitOk, "read it back: " + err);
    check(m.rows == 3 && m.cols == 2, "dimensions survive the round trip");
    check(m.data == values, "every value survives the round trip bit-exactly");

    // The payload of this file contains byte 0x0A, which Windows text mode would translate. A
    // failure here means a stream was opened without std::ios::binary.
    const std::vector<float> newline_byte = {1.4012985e-45f * 10.0f};  // 0x0000000A as a float
    const float raw_newline = []() {
        uint32_t bits = 0x0000000Au;
        float f = 0.0f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }();
    const std::vector<float> with_lf = {raw_newline, raw_newline, 1.0f};
    check(write_kmd("test_u12_lf.kmd", 3, 1, with_lf.data(), &err) == kExitOk,
          "write a payload containing 0x0A bytes");
    Matrix lf;
    check(read_kmd("test_u12_lf.kmd", &lf, &err) == kExitOk, "read it back: " + err);
    check(lf.data == with_lf, "0x0A payload bytes are not translated (binary mode)");
    (void)newline_byte;

    // FR-002: the declared shape must match the header.
    check(read_kmd_expect(path, 3, 2, &m, &err) == kExitOk, "matching declared shape accepted");
    const int mismatch = read_kmd_expect(path, 4, 2, &m, &err);
    check(mismatch == kExitInvalidInput, "declared shape mismatch is exit 2, got " +
                                             std::to_string(mismatch));

    // Truncation and trailing bytes are both caught by the exact-size rule.
    {
        std::vector<unsigned char> bytes;
        std::FILE *f = std::fopen(path.c_str(), "rb");
        check(f != nullptr, "reopen the file to copy its bytes");
        if (f != nullptr) {
            unsigned char buf[512];
            const size_t got = std::fread(buf, 1, sizeof(buf), f);
            bytes.assign(buf, buf + got);
            std::fclose(f);
        }

        std::FILE *t = std::fopen("test_u12_trunc.kmd", "wb");
        if (t != nullptr && bytes.size() > 4) {
            std::fwrite(bytes.data(), 1, bytes.size() - 4, t);
            std::fclose(t);
        }
        const int code = read_kmd("test_u12_trunc.kmd", &m, &err);
        check(code == kExitIoError, "truncated file is exit 5, got " + std::to_string(code));

        std::FILE *x = std::fopen("test_u12_extra.kmd", "wb");
        if (x != nullptr) {
            std::fwrite(bytes.data(), 1, bytes.size(), x);
            const unsigned char junk[4] = {1, 2, 3, 4};
            std::fwrite(junk, 1, sizeof(junk), x);
            std::fclose(x);
        }
        const int extra = read_kmd("test_u12_extra.kmd", &m, &err);
        check(extra == kExitIoError, "trailing bytes give exit 5, got " + std::to_string(extra));

        std::FILE *b = std::fopen("test_u12_magic.kmd", "wb");
        if (b != nullptr && bytes.size() > 4) {
            std::fwrite("XXXX", 1, 4, b);
            std::fwrite(bytes.data() + 4, 1, bytes.size() - 4, b);
            std::fclose(b);
        }
        const int magic = read_kmd("test_u12_magic.kmd", &m, &err);
        check(magic == kExitIoError, "bad magic gives exit 5, got " + std::to_string(magic));
    }

    const int missing = read_kmd("test_u12_does_not_exist.kmd", &m, &err);
    check(missing == kExitIoError, "missing file is exit 5, got " + std::to_string(missing));

    // .kma assignments.
    const std::vector<int32_t> assignments = {0, 1, 2, 1, 0, 2147483647};
    check(write_kma("test_u12.kma", assignments, &err) == kExitOk, "write .kma: " + err);
    std::vector<int32_t> read_back;
    check(read_kma("test_u12.kma", &read_back, &err) == kExitOk, "read .kma: " + err);
    check(read_back == assignments, ".kma round trip is bit-exact");

    for (const char *p : {"test_u12.kmd", "test_u12_lf.kmd", "test_u12_trunc.kmd",
                          "test_u12_extra.kmd", "test_u12_magic.kmd", "test_u12.kma"}) {
        std::remove(p);
    }
}

// --- TEST-U13: generator determinism -----------------------------------------------------------

void test_u13_generator()
{
    {
        TestScope scope("U13", "identical parameters reproduce identical datasets (FR-003)");
        GenParams p;
        p.N = 500;
        p.D = 4;
        p.K = 5;
        p.sep = 10.0;
        p.spread = 1.0;
        p.seed = 20261017;

        const Dataset a = generate(p);
        const Dataset b = generate(p);
        check(a.X == b.X, "the data matrix is bit-identical across runs");
        check(a.C0 == b.C0, "the initial centroids are bit-identical across runs");
        check(a.dataset_id == b.dataset_id, "the dataset ID is stable: " + a.dataset_id);
        check_eq_i64(static_cast<int64_t>(a.X.size()), p.N * p.D, "data size is N*D");
        check_eq_i64(static_cast<int64_t>(a.C0.size()), p.K * p.D, "centroid size is K*D");

        // A different seed must change the data; otherwise the seed is being ignored.
        GenParams q = p;
        q.seed = p.seed + 1;
        const Dataset c = generate(q);
        check(c.X != a.X, "a different seed changes the data");
        check(c.dataset_id != a.dataset_id, "a different seed changes the dataset ID");

        // The written files must be byte-identical too, which is what TEST-U13 compares.
        std::string err;
        check(write_kmd("test_u13_a.kmd", static_cast<uint64_t>(p.N),
                        static_cast<uint64_t>(p.D), a.X.data(), &err) == kExitOk, "write A");
        check(write_kmd("test_u13_b.kmd", static_cast<uint64_t>(p.N),
                        static_cast<uint64_t>(p.D), b.X.data(), &err) == kExitOk, "write B");
        Matrix ma, mb;
        check(read_kmd("test_u13_a.kmd", &ma, &err) == kExitOk, "read A");
        check(read_kmd("test_u13_b.kmd", &mb, &err) == kExitOk, "read B");
        check(ma.data == mb.data, "the two files hold identical bytes");
        std::remove("test_u13_a.kmd");
        std::remove("test_u13_b.kmd");

        // Every generated value must pass the VAL rules the loader will apply.
        double m_range = 0.0;
        const Validation v = validate_data(a.X, p.N, p.D, a.C0, p.K, &m_range);
        check(v.ok, "generated data passes VAL-006 and VAL-007");
        check(a.min_centre_distance > 0.0, "a positive minimum centre distance is measured");
    }

    {
        TestScope scope("U13", "well-separated blobs are separated by at least 10 sigma (§10.3)");
        GenParams p;
        p.N = 1000;
        p.D = 8;
        p.K = 4;
        p.sep = 10.0;
        p.spread = 1.0;
        p.seed = 7;
        const Dataset d = generate(p);
        const double sigma = d.min_centre_distance / p.spread;
        check(sigma >= 10.0, "minimum centre separation is " + double_str(sigma) +
                                 " sigma, want >= 10 for the well-separated datasets");
    }

    {
        TestScope scope("U13", "the uniform draw stays inside (0, 1)");
        // log(u1) in Box-Muller must never see 0, and a draw of exactly 1 would bias the angle.
        Rng rng(12345);
        double lo = 1.0;
        double hi = 0.0;
        for (int i = 0; i < 200000; ++i) {
            const double u = rng.uniform_open();
            if (u < lo) {
                lo = u;
            }
            if (u > hi) {
                hi = u;
            }
        }
        check(lo > 0.0, "smallest draw is above 0: " + double_str(lo));
        check(hi < 1.0, "largest draw is below 1: " + double_str(hi));
    }

    {
        TestScope scope("U13", "the normal draw has the expected first two moments");
        // A sanity check on Box-Muller, not a statistical test: with 400k samples the sample mean
        // is within about 0.005 of 0 and the variance within about 0.01 of 1.
        Rng rng(999);
        const int count = 400000;
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int i = 0; i < count; ++i) {
            const double z = rng.normal();
            sum += z;
            sum_sq += z * z;
        }
        const double mean = sum / count;
        const double variance = sum_sq / count - mean * mean;
        check(std::fabs(mean) < 0.02, "sample mean " + double_str(mean) + " is near 0");
        check(std::fabs(variance - 1.0) < 0.02,
              "sample variance " + double_str(variance) + " is near 1");
    }
}

// --- §16.3: the two distance paths agree except on near-ties -----------------------------------

void test_distance_paths_agree()
{
    TestScope scope("U01-pre", "FP32 and FP64 assignment agree except on near-ties (NUM-005)");

    GenParams p;
    p.N = 2000;
    p.D = 6;
    p.K = 8;
    p.sep = 3.0;   // deliberately tight, so near-ties actually occur
    p.spread = 1.0;
    p.seed = 424242;
    const Dataset d = generate(p);

    int64_t mismatches = 0;
    int64_t near_ties = 0;
    int64_t hard_mismatches = 0;
    double worst_margin = 0.0;

    for (int64_t i = 0; i < p.N; ++i) {
        const float *x = d.X.data() + i * p.D;
        const int32_t k32 = cpu::nearest_centroid(x, p.D, d.C0.data(), p.K);
        const cpu::TieInfo info = cpu::tie_info(x, p.D, d.C0.data(), p.K);
        if (info.near_tie) {
            ++near_ties;
        }
        if (k32 != info.k1) {
            ++mismatches;
            if (!info.near_tie) {
                ++hard_mismatches;
                if (info.margin > worst_margin) {
                    worst_margin = info.margin;
                }
            }
        }
    }

    std::printf("      %lld points, %lld near-ties, %lld FP32/FP64 mismatches, "
                "%lld of them outside the near-tie band\n",
                static_cast<long long>(p.N), static_cast<long long>(near_ties),
                static_cast<long long>(mismatches), static_cast<long long>(hard_mismatches));
    check(hard_mismatches == 0,
          "no mismatch outside the near-tie band; worst offending margin " +
              double_str(worst_margin) + " against a threshold of " +
              double_str(near_tie_threshold(p.D)));

    // Random blobs rarely land on a decision boundary, so the band above is usually empty and the
    // check passes without exercising NUM-005 at all. These constructed points sit exactly on, or
    // one ulp off, the perpendicular bisector of two centroids, which is where the rule bites
    // (EDGE-13, EDGE-19; TEST-U02 repeats them on the device).
    const std::vector<float> C = {0.0f, 0.0f, 2.0f, 0.0f};  // two centroids 2 apart in 2-D
    const std::vector<float> on_bisector = {1.0f, 0.0f, 1.0f, 5.0f, 1.0f, -3.25f};
    for (int64_t i = 0; i < 3; ++i) {
        const float *x = on_bisector.data() + i * 2;
        const int32_t k = cpu::nearest_centroid(x, 2, C.data(), 2);
        const cpu::TieInfo info = cpu::tie_info(x, 2, C.data(), 2);
        check_eq_i64(k, 0, "point on the bisector takes the lowest index (DEC-005)");
        check(info.margin == 0.0, "its binary64 margin is exactly 0");
        check(info.near_tie, "it is classified as a near-tie");
    }

    // Just off the bisector the nearer centroid must win outright, and the margin must be small
    // enough to still count as a near-tie, which is what makes a device mismatch here reportable
    // rather than a failure.
    const float nudge = std::nextafter(1.0f, 2.0f);
    const std::vector<float> off_bisector = {nudge, 0.0f};
    const cpu::TieInfo off = cpu::tie_info(off_bisector.data(), 2, C.data(), 2);
    check_eq_i64(cpu::nearest_centroid(off_bisector.data(), 2, C.data(), 2), 1,
                 "one ulp past the bisector the second centroid wins");
    check(off.near_tie, "a one-ulp margin is inside the near-tie band: margin " +
                            double_str(off.margin) + " <= " + double_str(near_tie_threshold(2)));

    // A point far from the boundary must not be excused as a near-tie.
    const std::vector<float> clear = {0.25f, 0.0f};
    const cpu::TieInfo far = cpu::tie_info(clear.data(), 2, C.data(), 2);
    check(!far.near_tie, "a clearly nearer centroid is not a near-tie: margin " +
                             double_str(far.margin));
    check_eq_i64(far.k1, 0, "and it is the lower-index one here");

    // Duplicate centroids make every point an exact tie (EDGE-10, EDGE-12).
    const std::vector<float> dup = {3.0f, 3.0f, 3.0f, 3.0f};
    const std::vector<float> any_point = {-7.5f, 2.0f};
    check_eq_i64(cpu::nearest_centroid(any_point.data(), 2, dup.data(), 2), 0,
                 "with duplicate centroids the lower index always wins");
}

// --- end-to-end sanity on generated data -------------------------------------------------------

void test_end_to_end_blobs()
{
    TestScope scope("I11-pre", "well-separated blobs recover their structure on the CPU");

    GenParams p;
    p.N = 600;
    p.D = 3;
    p.K = 4;
    p.sep = 20.0;
    p.spread = 1.0;
    p.seed = 31337;
    const Dataset d = generate(p);

    double m_range = 0.0;
    const Validation v = validate_data(d.X, p.N, p.D, d.C0, p.K, &m_range);
    check(v.ok, "generated data is valid");

    const Result r = cpu::run(d.X.data(), p.N, p.D, d.C0.data(), p.K, 100, 1e-6, m_range, false);
    check(is_converged(r.status), std::string("the run converged: ") + status_name(r.status));
    check(r.centroid_consistent, "P2 holds after convergence by assignment stability");
    check(r.assignment_consistent, "P1 holds by construction");

    uint64_t total = 0;
    for (uint32_t count : r.counts) {
        total += count;
    }
    check_eq_i64(static_cast<int64_t>(total), p.N, "the counts sum to N");

    for (int64_t i = 0; i < p.N; ++i) {
        if (r.assignments[static_cast<size_t>(i)] < 0 ||
            r.assignments[static_cast<size_t>(i)] >= p.K) {
            check(false, "assignment " + std::to_string(i) + " is outside [0, K)");
            break;
        }
    }

    // P1 independently: every reported assignment is the nearest centroid of the reported
    // centroids (§2.8 verification mode, TEST-I07 at CPU level).
    int64_t p1_violations = 0;
    for (int64_t i = 0; i < p.N; ++i) {
        const int32_t expect =
            cpu::nearest_centroid(d.X.data() + i * p.D, p.D, r.centroids.data(), p.K);
        if (expect != r.assignments[static_cast<size_t>(i)]) {
            ++p1_violations;
        }
    }
    check_eq_i64(p1_violations, 0, "P1 re-checked against the reported centroids");
    check(r.objective > 0.0, "the objective J is positive and finite");
    check(std::isfinite(r.objective), "the objective J is finite");
}

}  // namespace

int main()
{
    std::printf("M1 CPU test suite (requirements.md §16.0, §16.3, TEST-U08/U09/U11/U12/U13)\n\n");

    test_hand_cases();
    std::printf("\n");
    test_u08_movement();
    test_u09_empty_clusters();
    test_u11_size_calculator();
    test_u12_file_io();
    test_u13_generator();
    test_distance_paths_agree();
    test_end_to_end_blobs();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("RESULT: FAILED\n");
        return 1;
    }
    std::printf("RESULT: all CPU-level tests passed\n");
    return 0;
}
