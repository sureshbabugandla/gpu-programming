## 5. Source files

Each cell below writes one repository file verbatim into the working directory. They are generated
from `src/` and `tests/` by `tools/build_notebook.py`, which also checks that what the notebook
writes is byte-identical to what the repository holds (DEC-011) — so editing a cell here is always
wrong: change the file in the repository and regenerate.

The host side, in the ARC order of §18.2:

| File | Component | Responsibility |
|---|---|---|
| `common.h` | ARC-01 | `Status` (§2.7), exit codes (BLD-005), message format (BLD-006), the NUM-005/NUM-006 tolerances, and overflow-checked size arithmetic with the MEM-002 capacity model (CUDA-006) |
| `validate.h/.cpp` | ARC-01 | VAL-001…VAL-010, including the VAL-007 range bound $4DM^2 \le 2^{126}$ that guarantees no binary32 distance can overflow during the run |
| `io_kmd.h/.cpp` | ARC-02 | the `.kmd`/`.kma` binary formats of BLD-004, with the FR-002 exact-size check |
| `cpu_reference.h/.cpp` | ARC-03 | the oracle of NUM-003: the FP32 distance path of §2.2, a binary64 path for near-tie margins, binary64 sums, the tie rule, empty-cluster retention, the convergence loop and finalization. The step functions are exposed individually for the §10.3 lockstep verification added in M5 |
| `gen.h/.cpp`, `gen_main.cpp` | ARC-02 | the FR-003 generator and the `kmeans-gen` CLI of BLD-003a |
| `hand_data.h`, `test_cpu.cpp` | ARC-12 | H1–H6 and H2-a…e with their §16.0 expected outcomes, plus TEST-U08, U09, U11, U12 and U13 at CPU level |

The device side, added in M2, M3 and M4:

| File | Component | Responsibility |
|---|---|---|
| `launch_config.h` | ARC-04 | launch geometry and buffer sizing as plain integer arithmetic: the CUDA-007 grid rule, the RED-006 choice of `G`, the MEM-003 budget, the MEM-004/SMEM-003 shared-memory footprints and the DEC-001 chunk partition. Free of CUDA constructs on purpose, so all of it is unit tested with `g++` on a machine with no GPU |
| `cuda_util.cuh` | ARC-04 | CUDA-008 checked calls with file, line and `cudaGetErrorString`; `KMEANS_SYNC_CHECK`; and RAII owners for device memory, pinned memory, the stream and the events, so every error path releases what it allocated (MEM-007) |
| `device_context.cuh` | ARC-04 | CUDA-001 device query and the compute-capability 6.0 gate, the shared-memory limit checks, the `G` selection against real free memory, and VAL-009 — all before the first allocation, so EDGE-26 fails with exit code 3 and a byte count instead of a CUDA out-of-memory error |
| `kernels.cuh/.cu` | ARC-05/06/07/08 | the CUDA-002 kernel set: `assign` and `assign_smem` (§8), `accum_atomic` (§6), `reduce_stage1` and `reduce_stage2` (§7), `update`, `movement`, `max_reduce`. Thread mapping, ownership and synchronization are documented at each kernel and summarised in the three contract notes above |
| `kmeans_gpu.h/.cu` | ARC-09/ARC-10 | the §2.7 loop in the CUDA-003 launch order, §2.8 finalization, the PERF-005 timings, the §10.3 lockstep hook and the FR-019 `final` verification. Only the accumulation step differs between the two strategies (FR-018) |
| `test_gpu.cu` | ARC-12 | TEST-I01…I07, I10, U01, U03…U07, E16 and SMEM-004 on the device, with `--only` and `--quick` so the notebook can run the sections separately |

Still to come: the `kmeans` CLI executable and its `.kma`/`.kmd`/JSON outputs (FR-015, BLD-003a)
with the full test suite and the sanitizer suite in M5, and the benchmark harness in M6.
