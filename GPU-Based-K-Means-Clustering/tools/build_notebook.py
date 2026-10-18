#!/usr/bin/env python3
"""Generate kmeans_cuda.ipynb from the repository sources (requirements.md DEC-011, BLD-003).

The notebook is the deliverable but never the source of truth: every source cell is a
``%%writefile`` cell holding one file from ``src/`` or ``tests/`` verbatim, and every markdown or
Python cell is one file from ``tools/cells/``. The cell order is the normative order of section
15.3, held in MANIFEST below.

    python tools/build_notebook.py            write the notebook, then verify what was written
    python tools/build_notebook.py --check    verify the committed notebook; non-zero on drift
    python tools/build_notebook.py --list     print the section 15.3 cell plan and part statuses

``--check`` is what makes the DEC-011 claim that source cells are byte-identical to the repository
files checkable: it regenerates the notebook in memory, compares it byte for byte with the
committed file, and then re-extracts every ``%%writefile`` cell and compares it with its repository
file so a mismatch names the file. Use it as a pre-commit gate.

Generation is deterministic -- no timestamps, no random cell ids, content-derived metadata only --
so regenerating an unchanged tree produces an identical file and the notebook diffs cleanly.
Dependencies: the Python standard library only (section 1.5, CLAUDE.md).
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CELL_DIR = REPO_ROOT / "tools" / "cells"
NOTEBOOK_PATH = REPO_ROOT / "kmeans_cuda.ipynb"
SPEC_PATH = REPO_ROOT / "requirements.md"


# --- cell kinds ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Md:
    """Markdown cell from tools/cells/<name>."""

    name: str


@dataclass(frozen=True)
class Code:
    """Python code cell from tools/cells/<name>."""

    name: str


@dataclass(frozen=True)
class Writefile:
    """Code cell that writes one repository file into the runtime working directory.

    ``path`` is relative to the repository root; ``dest`` is the flat file name the notebook
    creates, because the configuration cell chdirs into WORK_DIR.
    """

    path: str
    dest: str


@dataclass(frozen=True)
class Part:
    """One numbered part of the section 15.3 cell structure."""

    number: int
    title: str
    milestone: str
    cells: tuple = ()
    note: str = ""

    @property
    def active(self) -> bool:
        return bool(self.cells)


# --- the section 15.3 cell plan ----------------------------------------------------------------
# Parts with no cells are planned: they emit nothing and are activated by the milestone named
# here, which adds their content files and fills in `cells`. The order is normative; it is kept
# explicit rather than globbed so a reviewer can check it against section 15.3.

MANIFEST = (
    Part(1, "Title, problem statement and design decisions", "M0", (Md("01_title.md"),)),
    Part(2, "Environment record (ENV-01..ENV-03, ENV-08)", "M0",
         (Md("02_environment.md"), Code("02_environment.py"))),
    Part(3, "Configuration (SM_ARCH, paths, build flags)", "M0",
         (Md("03_configuration.md"), Code("03_configuration.py"))),
    Part(4, "Toolchain smoke test", "M0",
         (Md("04_smoke_test.md"), Writefile("tests/hello.cu", "hello.cu"),
          Code("04_smoke_test.py"))),
    Part(5, "Source cells (%%writefile)", "M1-M4",
         (Md("05_sources.md"),
          Writefile("src/common.h", "common.h"),
          Writefile("src/validate.h", "validate.h"),
          Writefile("src/validate.cpp", "validate.cpp"),
          Writefile("src/io_kmd.h", "io_kmd.h"),
          Writefile("src/io_kmd.cpp", "io_kmd.cpp"),
          Writefile("src/cpu_reference.h", "cpu_reference.h"),
          Writefile("src/cpu_reference.cpp", "cpu_reference.cpp"),
          Writefile("src/gen.h", "gen.h"),
          Writefile("src/gen.cpp", "gen.cpp"),
          Writefile("src/gen_main.cpp", "gen_main.cpp"),
          Writefile("tests/hand_data.h", "hand_data.h"),
          Writefile("tests/test_cpu.cpp", "test_cpu.cpp"),
          # M2-M3 device side. The two contract notes come first: they are the thread-mapping,
          # ownership and synchronisation arguments for the kernels that follow.
          Md("05_kernel_contracts.md"),
          Md("05_reduction_contracts.md"),
          Md("05_smem_contract.md"),
          Writefile("src/launch_config.h", "launch_config.h"),
          Writefile("src/cuda_util.cuh", "cuda_util.cuh"),
          Writefile("src/device_context.cuh", "device_context.cuh"),
          Writefile("src/kernels.cuh", "kernels.cuh"),
          Writefile("src/kernels.cu", "kernels.cu"),
          Writefile("src/kmeans_gpu.h", "kmeans_gpu.h"),
          Writefile("src/kmeans_gpu.cu", "kmeans_gpu.cu"),
          Writefile("tests/test_gpu.cu", "test_gpu.cu")),
         note="the kmeans CLI executable is added in M5"),
    Part(6, "Compile cells", "M1-M2",
         (Md("06_compile.md"), Code("06_compile.py"),
          Md("06_compile_cuda.md"), Code("06_compile_cuda.py")),
         note="the sanitizer build is added in M5 (BLD-002)"),
    Part(7, "Hand-dataset demonstration H1-H6", "M1-M2",
         (Md("07_hand_tests.md"), Code("07_hand_tests.py"),
          Md("07_gpu_hand_tests.md"), Code("07_gpu_hand_tests.py"))),
    Part(8, "Test suite cell", "M2-M5",
         (Md("08_kernel_checks.md"), Code("08_kernel_checks.py"),
          Md("08_smem_checks.md"), Code("08_smem_checks.py")),
         note="M2-M4: lockstep, buffer initialisation and SMEM-004; grows into TEST-B02 in M5"),
    Part(9, "Sanitizer cells", "M5", note="TEST-S01-S04"),
    Part(10, "Benchmark cells", "M6", note="PERF-003 matrix, CSVs per section 13.1"),
    Part(11, "Analysis cells", "M6", note="time vs. N, D, K; numerical comparison tables"),
    Part(12, "Written evaluation and limitations", "M6", note="section 1.5 deliverable 5"),
)


# --- reading -----------------------------------------------------------------------------------


def read_text_lf(path: Path) -> str:
    """Read a repository file as text with LF endings and a guaranteed trailing newline.

    Normalising here is what lets the byte-identity check survive a Windows checkout
    (core.autocrlf=true) and guarantees that the files recreated on the Colab Linux runtime have
    LF endings whatever the editing platform did.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"error: {path} is not valid UTF-8: {exc}")
    text = text.replace("\r\n", "\n")
    if "\r" in text:
        raise SystemExit(f"error: {path} contains a lone CR; use LF line endings")
    if not text.endswith("\n"):
        raise SystemExit(f"error: {path} has no trailing newline")
    return text


def spec_version() -> str:
    """Version of requirements.md, so the notebook records the specification it was built from."""
    match = re.search(r"^\| Version \| (.+?) \|", read_text_lf(SPEC_PATH), re.MULTILINE)
    return match.group(1).strip() if match else "unknown"


def source_lines(text: str) -> list:
    """Split cell text into the nbformat source list: newline-terminated, bar the last line."""
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


# --- emission ----------------------------------------------------------------------------------


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def writefile_text(path: str, dest: str) -> str:
    """Body of a source cell: the writefile magic, then the repository file verbatim."""
    return f"%%writefile {dest}\n" + read_text_lf(REPO_ROOT / path)


def writefile_cell(path: str, dest: str) -> dict:
    """Source cell whose body keeps its trailing newline.

    Markdown and Python cells follow the usual notebook convention of a last line without a
    newline, but a source cell is different: whatever the cell holds is what %%writefile puts on
    disk, so dropping the final newline would write a C++ file not ending in one and would break
    the byte-identity check against the repository file (DEC-011).
    """
    cell = code_cell("")
    cell["source"] = writefile_text(path, dest).splitlines(keepends=True)
    return cell


def build_cells() -> list:
    cells = []
    destinations = {}
    for part in MANIFEST:
        for cell in part.cells:
            if isinstance(cell, Md):
                cells.append(markdown_cell(read_text_lf(CELL_DIR / cell.name)))
            elif isinstance(cell, Code):
                text = read_text_lf(CELL_DIR / cell.name)
                try:
                    compile(text, str(CELL_DIR / cell.name), "exec")
                except SyntaxError as exc:
                    raise SystemExit(f"error: {cell.name} does not compile: {exc}")
                cells.append(code_cell(text))
            elif isinstance(cell, Writefile):
                previous = destinations.get(cell.dest)
                if previous is not None:
                    raise SystemExit(
                        f"error: two source cells write {cell.dest} ({previous} and {cell.path}); "
                        "the notebook working directory is flat, so destinations must be unique"
                    )
                destinations[cell.dest] = cell.path
                cells.append(writefile_cell(cell.path, cell.dest))
            else:
                raise SystemExit(f"error: unknown cell kind {cell!r} in part {part.number}")
    return cells


def build_notebook() -> dict:
    return {
        "cells": build_cells(),
        "metadata": {
            "accelerator": "GPU",  # ENV-11: this notebook needs a GPU runtime
            "colab": {"name": "kmeans_cuda.ipynb", "provenance": [], "toc_visible": True},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "kmeans": {
                "generated_by": "tools/build_notebook.py",
                "spec_version": spec_version(),
                "cell_plan": "requirements.md section 15.3",
                "active_parts": [part.number for part in MANIFEST if part.active],
                "edit_policy": "generated file; edit src/, tests/ and tools/cells/ and regenerate",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def serialise(notebook: dict) -> bytes:
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


# --- checking ----------------------------------------------------------------------------------


def writefile_cells_of(notebook: dict) -> dict:
    """Map destination name to embedded text for every writefile cell of a notebook."""
    embedded = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        match = re.match(r"%%writefile (\S+)\n", text)
        if match:
            embedded[match.group(1)] = text[match.end():]
    return embedded


def check(notebook_path: Path) -> int:
    """Verify the committed notebook against the repository files. Returns an exit code."""
    if not notebook_path.exists():
        print(f"FAIL: {notebook_path.name} does not exist; run the generator", file=sys.stderr)
        return 1

    expected = serialise(build_notebook())
    actual = notebook_path.read_bytes()
    problems = []

    if actual != expected:
        problems.append(
            f"{notebook_path.name} differs from a fresh generation "
            f"({len(actual)} bytes on disk, {len(expected)} bytes generated)"
        )

    # Report per-file drift too, so a mismatch names the file and not just the notebook.
    try:
        on_disk = json.loads(actual.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        problems.append(f"{notebook_path.name} is not valid notebook JSON: {exc}")
        on_disk = {"cells": []}

    embedded = writefile_cells_of(on_disk)
    wanted = {
        cell.dest: cell.path
        for part in MANIFEST
        for cell in part.cells
        if isinstance(cell, Writefile)
    }
    for dest, path in sorted(wanted.items()):
        if dest not in embedded:
            problems.append(f"source cell for {path} (as {dest}) is missing from the notebook")
        elif embedded[dest] != read_text_lf(REPO_ROOT / path):
            problems.append(f"source cell {dest} is not byte-identical to {path}")
    for dest in sorted(set(embedded) - set(wanted)):
        problems.append(f"notebook writes {dest}, which no manifest entry accounts for")

    if problems:
        print("FAIL: notebook is out of date with the repository (DEC-011)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("  run: python tools/build_notebook.py", file=sys.stderr)
        return 1

    print(
        f"OK: {notebook_path.name} matches the repository "
        f"(cells={len(on_disk.get('cells', []))}, sources={len(embedded)})"
    )
    return 0


# --- CLI ---------------------------------------------------------------------------------------


def print_plan() -> None:
    print(f"requirements.md section 15.3 cell plan (spec version {spec_version()}):")
    print()
    for part in MANIFEST:
        status = "active " if part.active else "planned"
        print(f"  {part.number:>2}. [{status}] ({part.milestone:<5}) {part.title}")
        for cell in part.cells:
            if isinstance(cell, Writefile):
                print(f"        code     %%writefile {cell.dest}  <- {cell.path}")
            else:
                kind = "markdown" if isinstance(cell, Md) else "code    "
                print(f"        {kind} {cell.name}")
        if part.note:
            print(f"        ({part.note})")
    cells = sum(len(part.cells) for part in MANIFEST)
    active = sum(1 for part in MANIFEST if part.active)
    print()
    print(f"  {cells} cells from {active} active parts, {len(MANIFEST) - active} parts planned")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the notebook matches the repository; do not write")
    parser.add_argument("--list", action="store_true", help="print the cell plan and exit")
    parser.add_argument("--output", type=Path, default=NOTEBOOK_PATH,
                        help="notebook path to write (default: kmeans_cuda.ipynb)")
    args = parser.parse_args(argv)

    if args.list:
        print_plan()
        return 0
    if args.check:
        return check(args.output)

    payload = serialise(build_notebook())
    args.output.write_bytes(payload)
    print(f"wrote {args.output.name} ({len(payload)} bytes)")
    return check(args.output)


if __name__ == "__main__":
    sys.exit(main())
