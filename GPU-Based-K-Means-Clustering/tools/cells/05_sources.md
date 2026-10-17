## 5. Source files

Each cell below writes one repository file verbatim into the working directory. They are generated
from `src/` and `tests/` by `tools/build_notebook.py`, which also checks that what the notebook
writes is byte-identical to what the repository holds (DEC-011) — so editing a cell here is always
wrong: change the file in the repository and regenerate.

This milestone (M1) covers the host side only, in the ARC order of §18.2:

| File | Component | Responsibility |
|---|---|---|
| `common.h` | ARC-01 | `Status` (§2.7), exit codes (BLD-005), message format (BLD-006), the NUM-005/NUM-006 tolerances, and overflow-checked size arithmetic with the MEM-002 capacity model (CUDA-006) |
| `validate.h/.cpp` | ARC-01 | VAL-001…VAL-010, including the VAL-007 range bound $4DM^2 \le 2^{126}$ that guarantees no binary32 distance can overflow during the run |
| `io_kmd.h/.cpp` | ARC-02 | the `.kmd`/`.kma` binary formats of BLD-004, with the FR-002 exact-size check |
| `cpu_reference.h/.cpp` | ARC-03 | the oracle of NUM-003: the FP32 distance path of §2.2, a binary64 path for near-tie margins, binary64 sums, the tie rule, empty-cluster retention, the convergence loop and finalization. The step functions are exposed individually for the §10.3 lockstep verification added in M5 |
| `gen.h/.cpp`, `gen_main.cpp` | ARC-02 | the FR-003 generator and the `kmeans-gen` CLI of BLD-003a |
| `hand_data.h`, `test_cpu.cpp` | ARC-12 | H1–H6 and H2-a…e with their §16.0 expected outcomes, plus TEST-U08, U09, U11, U12 and U13 at CPU level |

The CUDA kernels, the device context and the `kmeans` driver are not here yet: they arrive in M2
once this reference passes, because every GPU result is judged against it.
