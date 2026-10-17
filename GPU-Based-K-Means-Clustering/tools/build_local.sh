#!/usr/bin/env bash
# build_local.sh — build and run the CPU-only parts locally (CLAUDE.md: the validation, I/O,
# generator, CPU reference and H1-H6 tests are compiled and run with g++ before committing).
#
# The compiler flags are exactly those of BLD-002 and NUM-003 for the host code:
#   -std=c++17 -O2 -Wall -Wextra -ffp-contract=off
# -ffp-contract=off is not cosmetic: the CPU reference is the oracle, and a contracted
# multiply-add in distance2_fp32 would change its rounding and make NUM-005 diagnosis meaningless.
#
# Two additions apply only to this Windows development machine and not to the notebook build:
#   * the portable WinLibs GCC 13.3.0 toolchain is put first on PATH (this laptop has no system
#     g++, and GCC 13.3.0 is the same version Colab runs, so warnings and FP behaviour match);
#   * -static links libstdc++ and libgcc into the binaries. Git Bash ships its own
#     /mingw64/bin/libstdc++-6.dll from a different GCC build, and if it shadows the toolchain's
#     DLL the test binary runs correctly but crashes during process exit, which looks exactly
#     like a test failure. Static linking removes that trap.
#
# Nothing here is CUDA: the GPU build is the notebook's nvcc cells (BLD-002).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
TOOLCHAIN_BIN="${KMEANS_GXX_BIN:-/c/Users/sures/tools/mingw64/bin}"

CXXFLAGS=(-std=c++17 -O2 -Wall -Wextra -ffp-contract=off)
LINKFLAGS=()

case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*)
    if [ -d "$TOOLCHAIN_BIN" ]; then
        export PATH="$TOOLCHAIN_BIN:$PATH"
    fi
    LINKFLAGS+=(-static)
    ;;
esac

mkdir -p "$BUILD_DIR"
cd "$REPO_ROOT"

echo "g++: $(g++ --version | head -1)"
echo "flags: ${CXXFLAGS[*]} ${LINKFLAGS[*]:-}"
echo

LIB_SOURCES=(src/validate.cpp src/io_kmd.cpp src/cpu_reference.cpp src/gen.cpp)

echo "building kmeans-gen"
g++ "${CXXFLAGS[@]}" "${LINKFLAGS[@]:-}" -I src src/gen_main.cpp src/gen.cpp src/io_kmd.cpp \
    src/validate.cpp -o "$BUILD_DIR/kmeans-gen"

echo "building test_cpu"
g++ "${CXXFLAGS[@]}" "${LINKFLAGS[@]:-}" -I src -I tests tests/test_cpu.cpp "${LIB_SOURCES[@]}" \
    -o "$BUILD_DIR/test_cpu"

echo
echo "running the CPU test suite"
cd "$BUILD_DIR"
./test_cpu
