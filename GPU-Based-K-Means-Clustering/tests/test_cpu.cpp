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

#include "cli_options.h"
#include "common.h"
#include "cpu_reference.h"
#include "gen.h"
#include "hand_data.h"
#include "io_kmd.h"
#include "launch_config.h"
#include "report.h"
#include "test_harness.h"
#include "validate.h"

using namespace kmeans;

namespace {

using kmeans::tests::check;
using kmeans::tests::check_eq_i64;
using kmeans::tests::double_str;
using kmeans::tests::float_str;
using kmeans::tests::TestScope;

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

// --- launch geometry and the choice of G (CUDA-007, RED-006, MEM-003, DEC-001) -----------------
//
// src/launch_config.h is deliberately free of CUDA constructs so that all of this arithmetic --
// which decides how much memory Approach B allocates and in what order it sums -- is testable
// without a GPU. The kernels use the identical expressions.

void test_launch_geometry()
{
    {
        TestScope scope("U11", "CUDA-007 grid sizing");
        // grid = min(ceil(work / block), 32 * SM count). With 40 SMs the cap is 1280.
        check_eq_i64(grid_size(1000000, 256, 40), 1280, "large work is capped at 32 * SMs");
        check_eq_i64(grid_size(1000, 256, 40), 4, "small work uses ceil(work / block)");
        check_eq_i64(grid_size(256, 256, 40), 1, "exactly one block of work");
        check_eq_i64(grid_size(257, 256, 40), 2, "one element past a block rounds up");
        check_eq_i64(grid_size(1, 256, 40), 1, "a single element still launches one block");
        check_eq_i64(grid_size(0, 256, 40), 1,
                     "zero work launches one block, which the bounds checks make a no-op");

        // The movement grid is capped at 1024 so max_reduce fits in one block and the cap matches
        // the movement_blocks default of device_memory_bytes.
        check_eq_i64(movement_grid_size(100000000, 256, 40), kMaxMovementBlocks,
                     "the movement grid is capped at 1024 blocks");
        check_eq_i64(movement_grid_size(512, 256, 40), 2, "small K is not affected by the cap");
    }

    {
        TestScope scope("U11", "MEM-004 / SMEM-003 shared-memory footprints");
        // Against the 48 KiB per-block default of ENV-01.
        const int64_t limit = 49152;
        check_eq_i64(stage2_shared_bytes(), 8 * 8 * 32 + 4 * 8, "reduce_stage2 needs 2080 B");
        check(stage2_shared_bytes() <= limit, "reduce_stage2 fits the 48 KiB limit");
        check_eq_i64(assign_smem_bytes(8, 32), 1024,
                     "assign_smem needs 1 KiB at the SMEM-001 defaults T_K = 8, T_D = 32");
        // The tile sizes are compile-time (SMEM-001), so the configured build is checked against
        // the formula rather than against the default literal: a -DKMEANS_TILE_K build must still
        // satisfy SMEM-003.
        check_eq_i64(assign_smem_bytes(kAssignTileK, kAssignTileD),
                     4 * static_cast<int64_t>(kAssignTileK) * kAssignTileD,
                     "the configured tile costs 4 T_K T_D bytes");
        check(assign_smem_bytes(kAssignTileK, kAssignTileD) <= limit,
              "the configured tile fits the 48 KiB limit");
        check(assign_smem_bytes(16, 32) <= limit, "the largest tile sweep configuration fits");
        check_eq_i64(block_reduction_bytes(1024), 8192,
                     "a 1024-thread block reduction needs 8 KiB");
        check(block_reduction_bytes(1024) <= limit, "the largest block reduction fits");
        // assign_smem asks for the tile and the change-count reduction in one allocation.
        check_eq_i64(assign_smem_shared_bytes(8, 32, 256), 1024 + 2048,
                     "assign_smem needs 3 KiB at the defaults with a 256-thread block");
        check_eq_i64(assign_smem_shared_bytes(16, 32, 1024), 2048 + 8192,
                     "the largest tile with the largest block needs 10 KiB");
        check(assign_smem_shared_bytes(kAssignTileK, kAssignTileD, 1024) <= limit,
              "the configured tile plus the largest block reduction fits the 48 KiB limit");
    }

    {
        TestScope scope("U11", "Stage 2 tile geometry (RED-003)");
        check_eq_i64(stage2_dim_tiles(1), 1, "D = 1 is one partial tile");
        check_eq_i64(stage2_dim_tiles(32), 1, "D = 32 is exactly one tile");
        check_eq_i64(stage2_dim_tiles(33), 2, "D = 33 needs a second, partial tile");
        check_eq_i64(stage2_dim_tiles(65), 3, "D = 65 needs three tiles");
        check_eq_i64(stage2_tiles(1024, 65), 1024 * 3, "K * T tiles in total");
    }

    {
        TestScope scope("U11", "RED-006 choice of G");
        // 40 SMs, so T_target = 81920. With D = 32 the target is ceil(81920/32) = 2560.
        const uint64_t plenty = 16ull * 1024 * 1024 * 1024;
        const ChunkSelection big =
            choose_reduce_chunks(1000000, 32, 64, 40, plenty, 1024 * 1024, 0, 0);
        check(big.ok, "a modest configuration selects a G");
        check_eq_i64(big.g_target, 2560, "G_target = ceil(T_target / D)");
        check_eq_i64(big.G, 2560, "memory is ample, so the target wins");
        check_eq_i64(static_cast<int64_t>(big.bytes_per_chunk), 8 * 64 * 32 + 4 * 64,
                     "8KD + 4K bytes per chunk");
        check_eq_i64(static_cast<int64_t>(big.required_bytes),
                     2560 * (8 * 64 * 32 + 4 * 64), "the reported byte count is 8GKD + 4GK");

        // G <= N always, however large the target.
        const ChunkSelection small_n =
            choose_reduce_chunks(100, 32, 64, 40, plenty, 1024 * 1024, 0, 0);
        check_eq_i64(small_n.G, 100, "G is clamped to N");

        // A tight budget makes G_mem the binding constraint.
        const uint64_t per_chunk = 8 * 64 * 32 + 4 * 64;
        const ChunkSelection tight =
            choose_reduce_chunks(1000000, 32, 64, 40, 4 * 100 * per_chunk, 0, 0, 0);
        check(tight.ok, "a tight budget still selects a G");
        check_eq_i64(tight.G, 100, "G_mem binds: floor(B_red / (8KD + 4K)) with B_red = free/4");
        check_eq_i64(tight.g_memory, 100, "G_mem is reported");

        // MEM-003: an explicit user cap overrides the quarter-of-free-memory rule.
        const ChunkSelection capped =
            choose_reduce_chunks(1000000, 32, 64, 40, plenty, 0, 10 * per_chunk, 0);
        check_eq_i64(capped.G, 10, "the user cap on B_red binds");

        // RED-007: no usable G at all fails, naming the bytes the run needs.
        const ChunkSelection impossible =
            choose_reduce_chunks(1000000, 32, 64, 40, per_chunk, 0, 0, 0);
        check(!impossible.ok, "a budget below one chunk is rejected");
        check(impossible.G == 0, "no G is reported when the selection fails");
        check(impossible.required_bytes == per_chunk,
              "the message reports the cost of a single chunk");
        check(!impossible.message.empty(), "the failure carries a RED-007 explanation");

        // The --reduce-chunks override is still bounded by N and by the budget.
        const ChunkSelection override_ok =
            choose_reduce_chunks(1000000, 32, 64, 40, plenty, 1024 * 1024, 0, 7);
        check(override_ok.ok && override_ok.G == 7, "an override within the limits is honoured");
        check(override_ok.user_override, "the override is recorded");
        const ChunkSelection override_too_big =
            choose_reduce_chunks(100, 32, 64, 40, plenty, 1024 * 1024, 0, 101);
        check(!override_too_big.ok, "an override above N is rejected (DEC-001 requires G <= N)");
        const ChunkSelection override_no_memory =
            choose_reduce_chunks(1000000, 32, 64, 40, 4 * 10 * per_chunk, 0, 0, 1000);
        check(!override_no_memory.ok, "an override past the budget is rejected, not silently used");

        // Small D leans on a large G for Stage 1 parallelism (RED-008).
        const ChunkSelection small_d =
            choose_reduce_chunks(10000000, 1, 4, 40, plenty, 1024 * 1024, 0, 0);
        check_eq_i64(small_d.g_target, 81920, "D = 1 gives G_target = T_target");
    }

    {
        TestScope scope("U11", "DEC-001 chunk partition covers [0, N) exactly (RED-005)");
        const int64_t sizes[][2] = {{1, 1},   {10, 1},   {10, 10}, {10, 3},  {100, 7},
                                    {97, 13}, {1000, 1}, {7, 7},   {12345, 101}};
        for (const auto &pair : sizes) {
            const int64_t N = pair[0], G = pair[1];
            const std::string tag = "N=" + std::to_string(N) + " G=" + std::to_string(G);
            check(chunk_begin(0, N, G) == 0, tag + ": the first chunk starts at 0");
            check(chunk_end(G - 1, N, G) == N, tag + ": the last chunk ends at N");
            int64_t covered = 0;
            bool contiguous = true;
            for (int64_t g = 0; g < G; ++g) {
                const int64_t begin = chunk_begin(g, N, G);
                const int64_t end = chunk_end(g, N, G);
                if (end < begin) {
                    contiguous = false;
                }
                if (g > 0 && begin != chunk_end(g - 1, N, G)) {
                    contiguous = false;  // a gap or an overlap between consecutive chunks
                }
                covered += end - begin;
            }
            check(contiguous, tag + ": consecutive chunks share endpoints");
            check_eq_i64(covered, N, tag + ": the chunks cover exactly N points");
            if (G <= N) {
                bool all_non_empty = true;
                for (int64_t g = 0; g < G; ++g) {
                    if (chunk_end(g, N, G) <= chunk_begin(g, N, G)) {
                        all_non_empty = false;
                    }
                }
                check(all_non_empty, tag + ": G <= N means every chunk is non-empty");
            }
        }

        // CUDA-006: the products stay inside int64 at the top of the valid range, where g*N is
        // close to 2^62. A 32-bit intermediate would wrap and silently mis-partition the data.
        const int64_t huge = 2147483647;  // 2^31 - 1, the VAL-002 limit
        check(chunk_end(huge - 1, huge, huge) == huge,
              "G = N at the VAL-002 limit still ends at N");
        check(chunk_begin(huge / 2, huge, 2) >= 0, "no overflow at the top of the range");
    }
}

// --- BLD-003a: the command line of the kmeans executable ---------------------------------------
//
// The CLI is the one part of the program a user drives directly, and a parser that quietly accepts
// a typo produces a plausible-looking result for a configuration nobody asked for. These cases run
// here, with g++ and no GPU, because parse_cli is deliberately free of CUDA (cli_options.h).

std::vector<const char *> argv_of(const std::vector<std::string> &args)
{
    static std::vector<std::string> storage;  // the pointers must outlive the call
    storage = args;
    std::vector<const char *> out;
    out.reserve(storage.size());
    for (const std::string &arg : storage) {
        out.push_back(arg.c_str());
    }
    return out;
}

// The minimal accepted command line; individual cases add to it or replace parts of it.
std::vector<std::string> base_args()
{
    return {"kmeans",     "--data",      "X.kmd", "--centroids", "C0.kmd", "--k",
            "8",          "--max-iter",  "20",    "--tol",       "0.001",  "--strategy",
            "reduction"};
}

CliParse parse_args(const std::vector<std::string> &args, CliOptions *out)
{
    const std::vector<const char *> argv = argv_of(args);
    return parse_cli(static_cast<int>(argv.size()), argv.data(), out);
}

void test_cli_options()
{
    {
        TestScope scope("CLI", "BLD-003a: the documented command line is accepted in full");
        std::vector<std::string> args = base_args();
        for (const char *extra : {"--assign", "smem", "--block-size", "512", "--reduce-chunks",
                                  "64", "--fixed-iters", "--verify", "final", "--out-prefix",
                                  "out/run1", "--reps", "10", "--warmup", "3", "--csv",
                                  "bench_raw.csv"}) {
            args.push_back(extra);
        }
        CliOptions o;
        const CliParse p = parse_args(args, &o);
        check(p.ok, "accepted: " + p.rule + " " + p.message);
        check_eq_i64(o.K, 8, "--k");
        check_eq_i64(o.max_iter, 20, "--max-iter");
        check(o.tol == 0.001, "--tol: got " + double_str(o.tol));
        check(o.strategy == Strategy::kReduction, "--strategy reduction");
        check(o.assign_kernel == AssignKernel::kSharedMemory, "--assign smem");
        check_eq_i64(o.block_size, 512, "--block-size");
        check_eq_i64(o.reduce_chunks, 64, "--reduce-chunks");
        check(o.fixed_iters, "--fixed-iters");
        check(o.verify == VerifyMode::kFinal, "--verify final");
        check(o.out_prefix == "out/run1", "--out-prefix");
        check_eq_i64(o.reps, 10, "--reps");
        check_eq_i64(o.warmup, 3, "--warmup");
        check(o.csv_path == "bench_raw.csv", "--csv");
        check(!o.profile_phases, "--profile-phases defaults to off");
    }

    {
        TestScope scope("CLI", "defaults: block size 256, automatic G, no verification");
        CliOptions o;
        const CliParse p = parse_args(base_args(), &o);
        check(p.ok, "the minimal command line is accepted");
        check_eq_i64(o.block_size, 256, "CUDA-007 default block size");
        check_eq_i64(o.reduce_chunks, 0, "RED-006: 0 means choose G automatically");
        check(o.verify == VerifyMode::kNone, "--verify defaults to none");
        check(o.assign_kernel == AssignKernel::kBaseline, "--assign defaults to baseline");
        check_eq_i64(o.warmup, 3, "PERF-006 default warm-up count");
        check_eq_i64(o.reps, 0, "no repetition series without --reps");
        check(o.out_prefix.empty(), "no outputs are written without --out-prefix");
    }

    {
        TestScope scope("CLI", "every required option is required (BLD-003a)");
        const char *required[] = {"--data", "--centroids", "--k", "--max-iter", "--tol",
                                  "--strategy"};
        for (const char *name : required) {
            std::vector<std::string> args;
            const std::vector<std::string> full = base_args();
            for (size_t i = 0; i < full.size(); ++i) {
                if (full[i] == name) {
                    ++i;  // skip the value too
                    continue;
                }
                args.push_back(full[i]);
            }
            CliOptions o;
            const CliParse p = parse_args(args, &o);
            check(!p.ok, std::string("omitting ") + name + " is rejected");
            check_eq_i64(p.exit_code, kExitInvalidInput,
                         std::string("omitting ") + name + ": exit code");
        }
    }

    {
        TestScope scope("CLI", "malformed values are rejected with exit code 2 (BLD-005)");
        struct Case {
            const char *option;
            const char *value;
            const char *why;
        };
        const Case cases[] = {
            {"--k", "eight", "a non-numeric K"},
            {"--k", "8x", "trailing characters after K"},
            {"--max-iter", "", "an empty iteration limit"},
            {"--tol", "abc", "an unparsable tolerance (VAL-005, the unparsable half)"},
            {"--strategy", "atomics", "a misspelled strategy"},
            {"--assign", "shared", "an unknown assignment kernel"},
            {"--verify", "all", "an unknown verification mode"},
            {"--block-size", "-32", "a negative block size"},
            {"--reduce-chunks", "1.5", "a non-integer G"},
            {"--reps", "0", "zero repetitions"},
            {"--warmup", "-1", "a negative warm-up count"},
        };
        for (const Case &c : cases) {
            std::vector<std::string> args = base_args();
            args.push_back(c.option);
            args.push_back(c.value);
            CliOptions o;
            const CliParse p = parse_args(args, &o);
            check(!p.ok, std::string(c.why) + " is rejected");
            check_eq_i64(p.exit_code, kExitInvalidInput, std::string(c.why) + ": exit code");
            check(!p.rule.empty(), std::string(c.why) + ": the message names a rule");
        }

        // An unknown option, a bare argument, and an option with no value.
        for (const std::vector<std::string> &extra :
             std::vector<std::vector<std::string>>{{"--threads", "8"}, {"positional"}, {"--k"}}) {
            std::vector<std::string> args = base_args();
            for (const std::string &a : extra) {
                args.push_back(a);
            }
            CliOptions o;
            const CliParse p = parse_args(args, &o);
            check(!p.ok, "'" + extra[0] + "' is rejected");
            check_eq_i64(p.exit_code, kExitInvalidInput, "'" + extra[0] + "': exit code");
        }
    }

    {
        TestScope scope("CLI", "--tol nan and --tol -1 reach VAL-005, not the parser");
        // strtod parses both, so the parser must let them through and validate_parameters must be
        // the single place that applies VAL-005. Two rules in two places would eventually disagree.
        for (const char *value : {"nan", "-1", "inf"}) {
            std::vector<std::string> args = base_args();
            args[10] = value;  // the --tol value in base_args
            CliOptions o;
            const CliParse p = parse_args(args, &o);
            check(p.ok, std::string("--tol ") + value + " parses");
            if (!p.ok) {
                continue;
            }
            const Validation v = validate_parameters(100, 4, o.K, o.max_iter, o.tol);
            check(!v.ok, std::string("--tol ") + value + " is rejected by VAL-005");
            check(v.rule == "VAL-005", std::string("--tol ") + value + ": rule is VAL-005, got " +
                                           v.rule);
            check_eq_i64(v.exit_code, kExitInvalidInput,
                         std::string("--tol ") + value + ": exit code");
        }
    }

    {
        TestScope scope("CLI", "combinations that cannot mean anything are rejected");
        std::vector<std::string> args = base_args();
        args.push_back("--csv");
        args.push_back("bench.csv");
        CliOptions o;
        CliParse p = parse_args(args, &o);
        check(!p.ok, "--csv without --reps is rejected: a raw row is one measured repetition");

        args = base_args();
        for (const char *extra : {"--reps", "10", "--verify", "lockstep"}) {
            args.push_back(extra);
        }
        p = parse_args(args, &o);
        check(!p.ok, "--reps with --verify lockstep is rejected (FR-019: never a sample)");

        args = base_args();
        for (const char *extra : {"--profile-phases", "--verify", "lockstep"}) {
            args.push_back(extra);
        }
        p = parse_args(args, &o);
        check(!p.ok, "--profile-phases with --verify lockstep is rejected (PERF-007: the "
                     "per-iteration verification copies land between two phase marks)");

        args = base_args();
        args.push_back("--profile-phases");
        p = parse_args(args, &o);
        check(p.ok && o.profile_phases,
              "--profile-phases on its own is accepted: " + p.rule + " " + p.message);
    }

    {
        TestScope scope("CLI", "--help prints usage and is not an error");
        CliOptions o;
        const CliParse p = parse_args({"kmeans", "--help"}, &o);
        check(p.usage, "--help requests the usage text");
        check(p.ok, "--help is not a parse failure");
        check(cli_usage().find("--strategy {atomic|reduction}") != std::string::npos,
              "the usage text shows the BLD-003a command line");
    }
}

// --- TEST-I08: the FR-015 summary and the section 13.1 CSV row ---------------------------------

RunReport sample_report()
{
    RunReport r;
    r.env.gpu_name = "Tesla T4";
    r.env.compute_capability = "7.5";
    r.env.sm_count = 40;
    r.env.driver_version = 13000;
    r.env.cuda_runtime = 13000;
    r.env.nvcc_version = "13.0.88";
    r.env.host_compiler = "13.3.0";
    r.env.os = "Linux";
    r.env.build_flags = "-O3 -arch=sm_75";
    r.env.git_commit = "abc1234";
    r.timestamp_utc = "2026-10-18T07:00:00Z";
    r.dataset_id = "n1000_d4_k3";
    r.data_path = "data/n1000_d4_k3_X.kmd";
    r.centroids_path = "data/n1000_d4_k3_C0.kmd";
    r.config.N = 1000;
    r.config.D = 4;
    r.config.K = 3;
    r.config.max_iter = 20;
    r.config.tol = 0.001;
    r.config.strategy = Strategy::kReduction;
    r.config.assign_kernel = AssignKernel::kSharedMemory;
    r.config.block_size = 256;
    r.tile_k = 8;
    r.tile_d = 32;
    r.reduce_chunks_g = 512;
    r.verify = VerifyMode::kFinal;
    r.rep_index = 3;
    r.iterations = 7;
    r.status = Status::CONVERGED_ASSIGNMENTS;
    r.centroid_consistent = true;
    r.last_delta = 0.5;
    r.has_last_delta = true;
    r.t_alg_ms = 12.5;
    r.t_e2e_ms = 20.25;
    r.objective = 1234.5;
    r.correctness_status = "PASS";
    r.counts = {400, 350, 250};
    r.run_id = make_run_id(r);
    return r;
}

void test_i08_reporting()
{
    {
        TestScope scope("I08", "the FR-015 summary carries every field the requirement names");
        const RunReport r = sample_report();
        const std::string json = summary_json(r);
        const char *required[] = {
            "\"status\": \"CONVERGED_ASSIGNMENTS\"", "\"iterations\": 7",
            "\"centroid_consistent\": true",         "\"assignment_consistent\": true",
            "\"last_delta\": 0.5",                   "\"strategy\": \"reduction\"",
            "\"assign_kernel\": \"smem\"",           "\"configuration\"",
            "\"t_alg\": 12.5",                       "\"t_e2e\": 20.25",
            "\"environment\"",                       "\"gpu_name\": \"Tesla T4\"",
            "\"counts\": [400, 350, 250]",           "\"correctness_status\": \"PASS\"",
            "\"reduce_chunks_G\": 512",              "\"tile_k\": 8",
        };
        for (const char *field : required) {
            check(json.find(field) != std::string::npos,
                  std::string("the summary contains ") + field);
        }
        // Balanced braces and brackets are a cheap proxy for "numpy/json can parse it", which the
        // notebook cell checks for real.
        int braces = 0, brackets = 0;
        for (char c : json) {
            braces += (c == '{') - (c == '}');
            brackets += (c == '[') - (c == ']');
        }
        check_eq_i64(braces, 0, "braces balance");
        check_eq_i64(brackets, 0, "brackets balance");
    }

    {
        TestScope scope("I08", "FR-015: Delta is null, not zero, when no update was performed");
        RunReport r = sample_report();
        r.iterations = 0;
        r.has_last_delta = false;
        const std::string json = summary_json(r);
        check(json.find("\"last_delta\": null") != std::string::npos,
              "t = 0 reports null rather than 0.0, which would be a different claim");
    }

    {
        TestScope scope("I08", "section 13.1: the CSV row has one field per header column");
        const RunReport r = sample_report();
        const std::string header = csv_header();
        const std::string row = csv_row(r);
        const auto count_fields = [](const std::string &line) {
            int64_t fields = 1;
            bool quoted = false;
            for (char c : line) {
                if (c == '"') {
                    quoted = !quoted;
                } else if (c == ',' && !quoted) {
                    ++fields;
                } else if (c == '\n') {
                    break;
                }
            }
            return fields;
        };
        check_eq_i64(count_fields(row), count_fields(header),
                     "the row has as many fields as the header has columns");
        check(row.find("reduction,smem,fp64,") != std::string::npos,
              "the strategy, kernel and precision columns: " + row.substr(0, 200));
        check(row.find(",CONVERGED_ASSIGNMENTS,") != std::string::npos, "the status column");
        check(row.find(",NA,NA,NA,NA,") != std::string::npos,
              "the four PERF-007 phase columns are NA outside phase-timing mode");
        check(summary_json(r).find("\"t_assign\": null") != std::string::npos,
              "and null in the JSON summary, where 0 would claim a phase took no time");
    }

    {
        TestScope scope("I08", "PERF-007: a phase-timing row carries the four phase columns");
        RunReport r = sample_report();
        r.has_phases = true;
        r.t_assign_ms = 12.5;
        r.t_accum_ms = 3.25;
        r.t_update_ms = 0.5;
        r.t_conv_ms = 0.125;
        const std::string row = csv_row(r);
        check(row.find(",12.5,3.25,0.5,0.125,") != std::string::npos,
              "the phase columns hold the measured sums: " + row.substr(0, 400));
        const std::string json = summary_json(r);
        check(json.find("\"t_assign\": 12.5") != std::string::npos &&
                  json.find("\"t_conv\": 0.125") != std::string::npos,
              "and the JSON summary reports them too");
    }

    {
        TestScope scope("I08", "columns that describe nothing are NA, not zero");
        RunReport r = sample_report();
        r.config.strategy = Strategy::kAtomic;
        r.config.assign_kernel = AssignKernel::kBaseline;
        r.reduce_chunks_g = 0;  // Approach A allocates no partial buffers
        const std::string row = csv_row(r);
        check(row.find("atomic,baseline,fp64,256,NA,NA,NA,") != std::string::npos,
              "the tile and G columns are NA for Approach A with the baseline kernel: " +
                  row.substr(0, 220));
        const std::string json = summary_json(r);
        check(json.find("\"reduce_chunks_G\": null") != std::string::npos,
              "and null in the JSON summary, where 0 would be a wrong number");
    }

    {
        TestScope scope("I08", "a field containing a comma survives the round trip");
        RunReport r = sample_report();
        r.env.build_flags = "-O3,-lineinfo,\"quoted\"";
        const std::string row = csv_row(r);
        check(row.find("\"-O3,-lineinfo,\"\"quoted\"\"\"") != std::string::npos,
              "the field is quoted and embedded quotes are doubled: " + row);
    }

    {
        TestScope scope("I08", "PERF-006 statistics over the measured repetitions");
        const Statistics s = statistics({4.0, 1.0, 3.0, 2.0});
        check_eq_i64(s.count, 4, "count");
        check(s.min == 1.0 && s.max == 4.0, "min and max");
        check(s.mean == 2.5, "mean: got " + double_str(s.mean));
        check(s.median == 2.5, "median of an even count is the midpoint of the two middles");
        // Sample standard deviation of {1,2,3,4}: sqrt(5/3).
        check(std::fabs(s.stddev - std::sqrt(5.0 / 3.0)) < 1e-12,
              "sample standard deviation: got " + double_str(s.stddev));
        check(std::fabs(s.cv - s.stddev / s.mean) < 1e-15, "the coefficient of variation");

        const Statistics odd = statistics({10.0, 30.0, 20.0});
        check(odd.median == 20.0, "median of an odd count is the middle value");
        const Statistics one = statistics({7.0});
        check(one.stddev == 0.0, "a single sample has no spread, not a division by zero");
        const Statistics none = statistics({});
        check_eq_i64(none.count, 0, "no samples reports no samples");
    }

    {
        TestScope scope("I08", "the dataset id is recovered from the data file name (FR-003)");
        check(dataset_id_from_path("/content/kmeans/data/n1000_d4_k3_seed7-abcd_X.kmd") ==
                  "n1000_d4_k3_seed7-abcd",
              "directory, extension and the generator's _X suffix are stripped");
        check(dataset_id_from_path("X.kmd") == "X", "a bare file name");
        check(dataset_id_from_path("") == "unknown", "an empty path is reported as unknown");
    }
}

}  // namespace

int main()
{
    std::printf("M1-M5 CPU test suite (requirements.md §16.0, §16.3, TEST-U08/U09/U11/U12/U13,\n"
                "TEST-I08, the BLD-003a command line, and the host-side launch geometry of\n"
                "CUDA-007, RED-006 and DEC-001)\n\n");

    test_hand_cases();
    std::printf("\n");
    test_u08_movement();
    test_u09_empty_clusters();
    test_u11_size_calculator();
    test_launch_geometry();
    test_u12_file_io();
    test_u13_generator();
    test_distance_paths_agree();
    test_end_to_end_blobs();
    test_cli_options();
    test_i08_reporting();

    return tests::report_totals("all CPU-level tests passed");
}
