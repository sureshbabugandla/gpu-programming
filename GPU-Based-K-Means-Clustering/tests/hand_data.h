// hand_data.h — the hand-verifiable datasets H1-H6 and H2-a..H2-e of requirements.md §16.0,
// together with their expected outcomes.
//
// These are the oracle for the oracle: every value here is derived by hand in the specification,
// so a CPU reference that reproduces them is trustworthy enough to judge the device against
// (TEST-I01..I06 reuse exactly these cases on the GPU). All values except 5/3 are exactly
// representable in binary32 and are compared exactly; 5/3 is compared with NUM-006.

#ifndef KMEANS_TESTS_HAND_DATA_H
#define KMEANS_TESTS_HAND_DATA_H

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

namespace kmeans {
namespace tests {

struct HandCase {
    std::string id;
    int64_t N = 0;
    int64_t D = 0;
    int64_t K = 0;
    std::vector<float> X;
    std::vector<float> C0;
    int64_t max_iter = 10;
    double tol = 0.0;

    // Expected outcome (§16.0).
    std::vector<int32_t> expected_A;
    std::vector<float> expected_C;
    std::vector<uint32_t> expected_n;
    int64_t expected_iterations = 0;
    Status expected_status = Status::ERROR;
    bool expected_centroid_consistent = false;
    bool expect_warning_kgtn = false;  // VAL-003
    bool expect_warning_dupc = false;  // VAL-008
    bool centroids_exact = true;       // false where 5/3 appears, compared via NUM-006
    std::string note;
};

// The 1-D H2 point set, reused by H2 and its five variants.
inline std::vector<float> h2_points()
{
    return {0.0f, 2.0f, 3.0f, 10.0f};
}

inline std::vector<float> h2_initial_centroids()
{
    return {0.0f, 3.0f};
}

inline std::vector<HandCase> hand_cases()
{
    std::vector<HandCase> cases;

    // H1: two well-separated pairs. A(0) = {0,0,1,1}; C(1) = {0.5, 9.5}; A(1) = A(0).
    {
        HandCase c;
        c.id = "H1";
        c.N = 4; c.D = 1; c.K = 2;
        c.X = {0.0f, 1.0f, 9.0f, 10.0f};
        c.C0 = {0.0f, 10.0f};
        c.expected_A = {0, 0, 1, 1};
        c.expected_C = {0.5f, 9.5f};
        c.expected_n = {2, 2};
        c.expected_iterations = 1;
        c.expected_status = Status::CONVERGED_ASSIGNMENTS;
        c.expected_centroid_consistent = true;
        cases.push_back(c);
    }

    // H2: four updates of drift. A(0)={0,1,1,1}; C(1)={0,5} (delta 2); A(1)={0,0,1,1};
    // C(2)={1,6.5} (delta 1.5); A(2)={0,0,0,1}; C(3)={5/3,10} (delta 3.5); A(3)=A(2).
    {
        HandCase c;
        c.id = "H2";
        c.N = 4; c.D = 1; c.K = 2;
        c.X = h2_points();
        c.C0 = h2_initial_centroids();
        c.expected_A = {0, 0, 0, 1};
        c.expected_C = {5.0f / 3.0f, 10.0f};
        c.expected_n = {3, 1};
        c.expected_iterations = 3;
        c.expected_status = Status::CONVERGED_ASSIGNMENTS;
        c.expected_centroid_consistent = true;
        c.centroids_exact = false;  // 5/3 is not representable in binary32 (NUM-006)
        cases.push_back(c);
    }

    // H2-a: tolerance 2 equals the first delta exactly, so the inclusive comparison of DEC-002
    // stops after one update. P2 is false: cluster 0 now holds {0,2}, whose mean is 1, not 0.
    {
        HandCase c;
        c.id = "H2-a";
        c.N = 4; c.D = 1; c.K = 2;
        c.X = h2_points();
        c.C0 = h2_initial_centroids();
        c.tol = 2.0;
        c.expected_A = {0, 0, 1, 1};
        c.expected_C = {0.0f, 5.0f};
        c.expected_n = {2, 2};
        c.expected_iterations = 1;
        c.expected_status = Status::CONVERGED_MOVEMENT;
        c.expected_centroid_consistent = false;
        c.note = "delta == tol exactly: inclusive comparison (ENV-10)";
        cases.push_back(c);
    }

    // H2-b: tolerance 1.5 equals the second delta exactly.
    {
        HandCase c;
        c.id = "H2-b";
        c.N = 4; c.D = 1; c.K = 2;
        c.X = h2_points();
        c.C0 = h2_initial_centroids();
        c.tol = 1.5;
        c.expected_A = {0, 0, 0, 1};
        c.expected_C = {1.0f, 6.5f};
        c.expected_n = {3, 1};
        c.expected_iterations = 2;
        c.expected_status = Status::CONVERGED_MOVEMENT;
        c.expected_centroid_consistent = false;
        cases.push_back(c);
    }

    // H2-c: just below that delta, so the run continues to assignment stability as in H2.
    {
        HandCase c;
        c.id = "H2-c";
        c.N = 4; c.D = 1; c.K = 2;
        c.X = h2_points();
        c.C0 = h2_initial_centroids();
        c.tol = 1.4999;
        c.expected_A = {0, 0, 0, 1};
        c.expected_C = {5.0f / 3.0f, 10.0f};
        c.expected_n = {3, 1};
        c.expected_iterations = 3;
        c.expected_status = Status::CONVERGED_ASSIGNMENTS;
        c.expected_centroid_consistent = true;
        c.centroids_exact = false;
        cases.push_back(c);
    }

    // H2-d: the limit cuts the run short; termination is not reported as convergence.
    {
        HandCase c;
        c.id = "H2-d";
        c.N = 4; c.D = 1; c.K = 2;
        c.X = h2_points();
        c.C0 = h2_initial_centroids();
        c.max_iter = 2;
        c.expected_A = {0, 0, 0, 1};
        c.expected_C = {1.0f, 6.5f};
        c.expected_n = {3, 1};
        c.expected_iterations = 2;
        c.expected_status = Status::ITERATION_LIMIT;
        c.expected_centroid_consistent = false;
        cases.push_back(c);
    }

    // H2-e: max_iter = 0 runs the initial assignment pass only (§2.7 boundary behaviour).
    {
        HandCase c;
        c.id = "H2-e";
        c.N = 4; c.D = 1; c.K = 2;
        c.X = h2_points();
        c.C0 = h2_initial_centroids();
        c.max_iter = 0;
        c.expected_A = {0, 1, 1, 1};
        c.expected_C = {0.0f, 3.0f};
        c.expected_n = {1, 3};
        c.expected_iterations = 0;
        c.expected_status = Status::ITERATION_LIMIT;
        c.expected_centroid_consistent = false;  // cluster 1 holds {2,3,10}, mean 5, not 3
        cases.push_back(c);
    }

    // H3: one point exactly between two centroids. The tie goes to the lower index (DEC-005) and
    // cluster 1 is empty from the start, so it retains its initial centroid (DEC-003).
    {
        HandCase c;
        c.id = "H3";
        c.N = 1; c.D = 1; c.K = 2;
        c.X = {1.0f};
        c.C0 = {0.0f, 2.0f};
        c.expected_A = {0};
        c.expected_C = {1.0f, 2.0f};
        c.expected_n = {1, 0};
        c.expected_iterations = 1;
        c.expected_status = Status::CONVERGED_ASSIGNMENTS;
        c.expected_centroid_consistent = true;  // the only non-empty cluster holds its own mean
        // K = 2 > N = 1, so VAL-003 applies here as well. The §16.0 row for H3 does not mention
        // the warning, but the rule is normative and unconditional.
        c.expect_warning_kgtn = true;
        c.note = "exact tie -> lowest index; empty cluster retained and excluded from delta";
        cases.push_back(c);
    }

    // H4: duplicate initial centroids. Every point ties and goes to index 0 (W-DUPC).
    {
        HandCase c;
        c.id = "H4";
        c.N = 2; c.D = 1; c.K = 2;
        c.X = {0.0f, 1.0f};
        c.C0 = {5.0f, 5.0f};
        c.expected_A = {0, 0};
        c.expected_C = {0.5f, 5.0f};
        c.expected_n = {2, 0};
        c.expected_iterations = 1;
        c.expected_status = Status::CONVERGED_ASSIGNMENTS;
        c.expected_centroid_consistent = true;
        c.expect_warning_dupc = true;
        cases.push_back(c);
    }

    // H5: K > N, so at least K - N clusters stay empty for the whole run (VAL-003, W-KGTN).
    {
        HandCase c;
        c.id = "H5";
        c.N = 1; c.D = 1; c.K = 3;
        c.X = {4.0f};
        c.C0 = {0.0f, 1.0f, 2.0f};
        c.expected_A = {2};
        c.expected_C = {0.0f, 1.0f, 4.0f};
        c.expected_n = {0, 0, 1};
        c.expected_iterations = 1;
        c.expected_status = Status::CONVERGED_ASSIGNMENTS;
        c.expected_centroid_consistent = true;
        c.expect_warning_kgtn = true;
        cases.push_back(c);
    }

    // H6: the 2-D case, which catches a transposed or stride-confused layout that every 1-D case
    // above would pass.
    {
        HandCase c;
        c.id = "H6";
        c.N = 4; c.D = 2; c.K = 2;
        c.X = {0.0f, 0.0f, 0.0f, 1.0f, 10.0f, 0.0f, 10.0f, 1.0f};
        c.C0 = {0.0f, 0.0f, 10.0f, 0.0f};
        c.expected_A = {0, 0, 1, 1};
        c.expected_C = {0.0f, 0.5f, 10.0f, 0.5f};
        c.expected_n = {2, 2};
        c.expected_iterations = 1;
        c.expected_status = Status::CONVERGED_ASSIGNMENTS;
        c.expected_centroid_consistent = true;
        cases.push_back(c);
    }

    return cases;
}

}  // namespace tests
}  // namespace kmeans

#endif  // KMEANS_TESTS_HAND_DATA_H
