## 6. Compile the host code

Builds the two host executables with the BLD-002 host flags, extended with `-Wall -Wextra`:

```
-std=c++17 -O2 -ffp-contract=off -Wall -Wextra
```

`-ffp-contract=off` (NUM-003) is the one flag that changes results rather than diagnostics. The
CPU reference is the oracle the device is judged against, and §2.2 fixes the distance as a
binary32 multiply followed by a binary32 add. If the compiler were allowed to contract that pair
into an FMA, the oracle's rounding would shift, every near-tie margin computed from it would be
measured against the wrong baseline, and NUM-005 mismatch diagnosis would become meaningless. The
device is permitted to contract (`-fmad=true`), which is precisely why the reference must not.

The cell treats **any** compiler output as a failure, not only a non-zero exit status: NFR-014
requires the compile cells to produce no host-code warnings, and a warning that merely scrolls past
in a saved notebook is indistinguishable from one nobody read.
