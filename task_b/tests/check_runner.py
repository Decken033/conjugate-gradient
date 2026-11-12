"""Bounded production-runner checks. Requires OpenMPI MPMD colon syntax."""
import argparse
import csv
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--preflag", action="append", default=[])
    parser.add_argument("--postflag", action="append", default=[])
    args = parser.parse_args()

    def run(command, success, marker):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=25)
        if ((result.returncode == 0) != success or marker not in result.stdout
                or (not success and "[rank " not in result.stdout)):
            raise RuntimeError(f"Unexpected exit/diagnostic: {command}\n{result.stdout}")

    def context(count, arguments):
        return [args.numproc_flag, str(count), *args.preflag, args.runner,
                *args.postflag, *arguments]

    def command(arguments):
        return [args.launcher, *context(2, arguments)]

    with tempfile.TemporaryDirectory(prefix="task_b_runner_") as temporary:
        root = Path(temporary)
        output = root / "result.csv"
        good = ["2d", "3", "1", str(output), "0"]
        run(command(good), True, "Converged: 1/1")
        with output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0]["converged"] != "1" or rows[0]["ranks"] != "2":
            raise RuntimeError("Successful launch did not produce the expected CSV")
        for arguments, marker in [
            (["2d"], "MPI runner: invalid argument count"),
            (["2d", "3", "0", str(output)], "MPI runner: repeats must be >= 1"),
            (["2d", "3", "1x", str(output)], "MPI runner: expected an unsigned integer"),
            (["2d", "50000", "1", str(output)], "MPI runner: grid exceeds supported MPI row count"),
            (["thermal2", str(root / "missing.mtx"), "1", str(output)], "Cannot open Matrix Market file:"),
            (["2d", "3", "1", str(root)], "MPI benchmark CSV: cannot open output file:"),
        ]:
            # Require a production rank diagnostic, not an unrelated launcher error.
            run(command(arguments), False, marker)
        version = subprocess.run([args.launcher, "--version"], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, timeout=10)
        if version.returncode or not any(name in version.stdout for name in ("Open MPI", "OpenRTE")):
            raise RuntimeError("MPMD regression requires OpenMPI; unsupported launcher is not a pass")
        def mpmd(first, second):
            return [args.launcher, *context(1, first), ":", *context(1, second)]
        output.unlink()
        run(mpmd(good, good), True, "Converged: 1/1")
        with output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0]["ranks"] != "2" or rows[0]["converged"] != "1":
            raise RuntimeError("MPMD preflight did not execute a two-rank production solve")
        for bad, marker in [(["2d"], "invalid argument count"),
                            (["2d", "3", "0", str(output), "0"], "repeats must be >= 1"),
                            (["2d", "3", "2", str(output), "0"], "arguments differ across ranks")]:
            run(mpmd(good, bad), False, "MPI runner: " + marker)
    print("Production runner safety checks passed")


if __name__ == "__main__":
    main()
