"""Bounded integration checks: MPI launcher failure is never a passing test."""
import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--launcher', required=True)
p.add_argument('--numproc-flag', default='-n')
p.add_argument('--runner', required=True)
p.add_argument('--ranks', type=int, default=4)
a = p.parse_args()
with tempfile.TemporaryDirectory(prefix='cg-mpi-regression-') as temp:
    root = Path(temp)
    cases = [
        ('missing-input', ['thermal2', str(root / 'missing.mtx'), '1', str(root / 'missing.csv'), '0'], False, 'Matrix preparation failed:'),
        ('invalid-grid', ['2d', 'not-a-number', '1', str(root / 'invalid.csv'), '0'], False, 'Matrix preparation failed:'),
        ('output-failure', ['2d', '3', '1', str(root), '0'], False, 'CSV output failed:'),
        ('normal', ['2d', '3', '1', str(root / 'normal.csv'), '0'], True, 'Converged: 1/1'),
    ]
    for name, args, success, marker in cases:
        cmd = [a.launcher, a.numproc_flag, str(a.ranks), a.runner, *args]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=35)
        print(f'=== {name}: exit={result.returncode} ===\n{result.stdout}', flush=True)
        if (result.returncode == 0) != success or marker not in result.stdout:
            raise AssertionError(f'{name}: unexpected exit or missing application marker')
        if not success and any(root.glob('missing.csv')):
            raise AssertionError('Failed input created output')
    with (root / 'normal.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0]['converged'] != '1' or float(rows[0]['true_relative_residual']) > 1e-8:
        raise AssertionError('Normal runner CSV invalid')
print('MPI runner normal/input/output regression checks passed.')
