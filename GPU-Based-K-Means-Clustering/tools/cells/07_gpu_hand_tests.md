### 7b. Hand datasets on the GPU, both strategies (TEST-I01, TEST-I02)

The same §16.0 cases the CPU reference reproduced above, now run on the device with **Approach A
(atomic)** and **Approach B (reduction)**, which is TEST-I01 and TEST-I02. Because the H2 variants
encode the convergence semantics, this one cell also carries TEST-I03 (iteration counting: H1 at
`t=1`, H2 at `t=3`), TEST-I04 (the movement boundary: H2-a/b/c), TEST-I05 (the limit, including
`I_max = 0`: H2-d/e) and TEST-I06 (the P2 flag).

Expected output: every case `PASS`, for both strategies, with `0 failures`. The values checked are
exactly the ones derived by hand in §16.0 — assignments, centroids, counts, iteration count,
status and the P2 flag — compared exactly, except `5/3` in H2 and H2-c which goes through NUM-006.
Empty clusters are additionally checked to have retained their initial centroid **bit-exactly**
(DEC-003), and each case is re-judged by `gpu::verify_final`, so P1 is verified rather than
assumed (FR-019, §2.8).

A failure here means a kernel disagrees with arithmetic done by hand on four points, so fix it in
`src/` before reading anything into the lockstep or benchmark cells.
