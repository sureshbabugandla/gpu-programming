# TEST-S01..S04: compute-sanitizer over small, deliberately non-aligned configurations (§16.4).
import os
import re
import subprocess

if "QUICK" not in globals():
    raise RuntimeError("Run the configuration cell (section 3) first.")

if QUICK:
    print("QUICK = True: the sanitizer suite is skipped. Set QUICK = False in section 3 to run "
          "TEST-S01..S04.")
else:
    if not ENV.get("compute_sanitizer"):
        # ENV-08: the tools may be absent on a Colab runtime, and a sanitizer that was never run
        # must not be recorded as having passed (CLAUDE.md, PERF-011).
        raise RuntimeError(
            "compute-sanitizer is not available in this runtime (ENV-02/ENV-08), so TEST-S01..S04 "
            "cannot be run here. Do not record them as passed; re-run this notebook on a runtime "
            "that has it, or record the suite as not performed."
        )

    SAN_DIR = os.path.join(DATA_DIR, "sanitizer")
    os.makedirs(SAN_DIR, exist_ok=True)
    san_failures = []

    # Sizes chosen for shape, not for size: none of N, D or K is a multiple of the block (256),
    # the tile (T_K = 8, T_D = 32) or the warp (32), so every kernel runs a partial final block
    # and every tile load overhangs (EDGE-16, EDGE-17). Sanitizers cost one to two orders of
    # magnitude in time, so coverage has to come from shapes rather than from sizes.
    SAN_N, SAN_D, SAN_K = 1001, 33, 7
    SAN_PREFIX = os.path.join(SAN_DIR, "san")
    subprocess.run(["./kmeans-gen", "--n", str(SAN_N), "--d", str(SAN_D), "--k", str(SAN_K),
                    "--sep", "20", "--spread", "1", "--seed", "909", "--out-prefix", SAN_PREFIX],
                   capture_output=True, text=True, check=True)
    print(f"dataset: {SAN_N} x {SAN_D}, K = {SAN_K} "
          "(no dimension a multiple of the block, tile or warp size)\n")

    SAN_ARGS = ["--data", SAN_PREFIX + "_X.kmd", "--centroids", SAN_PREFIX + "_C0.kmd",
                "--k", str(SAN_K), "--max-iter", "4", "--tol", "0.0"]

    # Every (strategy, assign kernel) pair: a tool has nothing to say about a kernel that did not
    # run, so all four combinations are covered for each tool.
    VARIANTS = [
        (f"{strategy}/{assign}",
         ["./kmeans_dbg"] + SAN_ARGS + ["--strategy", strategy, "--assign", assign])
        for strategy in ("atomic", "reduction")
        for assign in ("baseline", "smem")
    ]


    def sanitize(test_id, tool, label, cmd, extra_tool_flags=(), env_extra=None):
        """Run one command under one tool and insist on a zero error count and no leak.

        The application's own exit code is deliberately not checked here: the error-path cases of
        EDGE-27/28 are *supposed* to exit 4, and what this function asserts about them is only
        that nothing was reported and nothing leaked on the way out. Their exit codes are checked
        in the test-suite cell above.
        """
        full = ["compute-sanitizer", "--tool", tool] + list(extra_tool_flags) + cmd
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        proc = subprocess.run(full, capture_output=True, text=True, env=env)
        output = proc.stdout + proc.stderr

        summary = re.search(r"ERROR SUMMARY: (\d+) error", output)
        leaked = re.search(r"LEAK SUMMARY: (\d+) bytes leaked", output)
        errors = int(summary.group(1)) if summary else None
        leak_bytes = int(leaked.group(1)) if leaked else None

        problems = []
        if errors is None:
            problems.append("no ERROR SUMMARY line in the output: the tool did not run to "
                            "completion")
        elif errors != 0:
            problems.append(f"{errors} error(s)")
        if leak_bytes not in (None, 0):
            problems.append(f"{leak_bytes} bytes leaked")

        detail = f"{errors if errors is not None else '?'} errors"
        if leak_bytes is not None:
            detail += f", {leak_bytes} B leaked"
        print(f"  {'ok  ' if not problems else 'FAIL'} {test_id:<4} {tool:<10} {label:<46} "
              f"{detail}")
        if problems:
            san_failures.append(f"{test_id} {tool} {label}: {'; '.join(problems)}\n"
                                f"{output.strip()[-3000:]}")


    print("TEST-S01 — memcheck --leak-check full (NFR-003, NFR-004, MEM-007)")
    for label, cmd in VARIANTS:
        sanitize("S01", "memcheck", label, cmd, ["--leak-check", "full"])
    # The error paths. This is the MEM-007 claim that actually matters: not that a successful run
    # frees its memory, but that a run which threw partway through the allocation sequence still
    # released what it had already taken.
    for label, env_extra in (("error path: KMEANS_FAIL_ALLOC=5", {"KMEANS_FAIL_ALLOC": "5"}),
                             ("error path: KMEANS_FAIL_LAUNCH=assign",
                              {"KMEANS_FAIL_LAUNCH": "assign"})):
        sanitize("S01", "memcheck", label,
                 ["./kmeans_dbg"] + SAN_ARGS + ["--strategy", "reduction"],
                 ["--leak-check", "full"], env_extra=env_extra)
    # The widest set of shapes the project has in one process: K > N, all-identical points, all-
    # identical centroids, N = 1, K = 1, the tie cases and the rejection paths.
    sanitize("S01", "memcheck", "test_edge --only edges",
             ["./test_edge_dbg", "--only", "edges", "--quick"], ["--leak-check", "full"])

    print("\nTEST-S02 — racecheck: shared-memory hazards in assign, assign_smem, reduce_stage2 "
          "and movement (NFR-005)")
    for label, cmd in VARIANTS:
        sanitize("S02", "racecheck", label, cmd)

    print("\nTEST-S03 — initcheck: no read of device memory that nothing wrote (MEM-006)")
    for label, cmd in VARIANTS:
        sanitize("S03", "initcheck", label, cmd)

    print("\nTEST-S04 — synccheck: every barrier reached by the whole block (CUDA-005, SMEM-002)")
    for label, cmd in VARIANTS:
        sanitize("S04", "synccheck", label, cmd)

    # N = 1 is the sharpest barrier case there is: the grid-stride loop body runs for one thread
    # of the block and for none of the others, which is exactly what the block-uniform trip count
    # of assign_smem exists to handle (SMEM-002).
    print("\nTEST-S04 — synccheck at N = 1, K = 1: the extreme partial-block shape")
    TINY = os.path.join(SAN_DIR, "tiny")
    subprocess.run(["./kmeans-gen", "--n", "1", "--d", "3", "--k", "1", "--sep", "10",
                    "--spread", "1", "--seed", "1", "--out-prefix", TINY],
                   capture_output=True, text=True, check=True)
    for assign in ("baseline", "smem"):
        sanitize("S04", "synccheck", f"N = 1, K = 1, {assign}",
                 ["./kmeans_dbg", "--data", TINY + "_X.kmd", "--centroids", TINY + "_C0.kmd",
                  "--k", "1", "--max-iter", "2", "--tol", "0.0", "--strategy", "reduction",
                  "--assign", assign])

    print("\n" + "=" * 100)
    if san_failures:
        print(f"{len(san_failures)} SANITIZER FAILURE(S):")
        for f in san_failures:
            print(f"\n--- {f}")
        raise RuntimeError(
            f"{len(san_failures)} sanitizer failure(s); see above. §16.4 requires zero errors, "
            "zero hazards, zero uninitialised reads and zero leaks. Fix the source in src/, "
            "regenerate the notebook, and re-run."
        )
    print("TEST-S01..S04: clean. 0 errors, 0 hazards, 0 uninitialised reads, 0 bytes leaked.")
    print("Note (§16.4): racecheck covers shared memory only. Approach B's freedom from global "
          "races rests on the RED-004 ownership argument and is supported by TEST-R01 and "
          "TEST-U05, not by this run.")
