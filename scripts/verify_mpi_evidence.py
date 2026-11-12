"""Read-only verification of the imported Task B evidence manifest."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('solver', 'matrix_name', 'n', 'nnz', 'ranks', 'threads', 'run_index',
          'iterations', 'converged', 'solve_seconds', 'true_relative_residual',
          'relative_solution_error')


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    manifest = json.loads((ROOT / 'results/mpi/evidence/manifest.json').read_text(encoding='utf-8'))
    files = manifest['files']
    check(len(files) == 76, 'Expected 76 evidence files')
    check(len({item['path'] for item in files}) == 76, 'Duplicate manifest path')
    check(Counter(item['role'] for item in files) == {
        'raw_csv': 60, 'historical_log': 7, 'recovered_script': 7, 'hostfile': 2}, 'Role inventory')
    ranks = (1, 2, 4, 8, 16, 32, 36, 72)
    expected = set()
    for size, job in ((512, '1482451'), (1024, '1482491')):
        expected.update(f'results/mpi/mpi_2d{size}_p{p}_{job}.stdct-mgmt-02.csv' for p in ranks)
    expected.update(f'results/mpi/thermal2/thermal2_p{p}_1495684.stdct-mgmt-02.csv' for p in (1, 2, 4, 8, 16))
    expected.update(f'results/mpi/3d/mpi_3d{s}_p{p}_1502451.stdct-mgmt-02.csv' for s in (40, 64, 102) for p in ranks)
    expected.update(f'results/mpi/extension/strong_1509066.stdct-mgmt-02/strong_2d1024_p{p}_1509066.stdct-mgmt-02.csv' for p in (*ranks, 144, 256))
    expected.update(f'results/mpi/extension/weak_1509067.stdct-mgmt-02/weak_2d{s}_p{p}_1509067.stdct-mgmt-02.csv' for p, s in ((1,128),(4,256),(16,512),(64,1024),(256,2048)))
    check({f['path'] for f in files if f['role'] == 'raw_csv'} == expected, 'Selected CSV inventory')
    records = 0
    for item in files:
        path = ROOT / item['path']
        check(path.resolve().is_relative_to(ROOT), f'Unsafe path: {path}')
        data = path.read_bytes()
        check(len(data) == item['bytes'], f'Byte length: {path}')
        check(hashlib.sha256(data).hexdigest() == item['sha256'], f'SHA-256: {path}')
        if item['role'] != 'raw_csv':
            continue
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            check(tuple(reader.fieldnames or ()) == FIELDS, f'Schema: {path}')
            rows = list(reader)
        check(len(rows) == 10 and {r['run_index'] for r in rows} == {str(i) for i in range(10)}, f'Run indices: {path}')
        for row in rows:
            check(set(row) == set(FIELDS) and all(row[k] not in ('', None) for k in FIELDS), f'Incomplete record: {path}')
            check(row['solver'] == 'mpi' and row['threads'] == '1' and row['converged'] == '1', f'Flags: {path}')
            check(int(row['iterations']) > 0 and int(row['n']) > 0 and int(row['nnz']) > 0 and int(row['ranks']) > 0, f'Metadata: {path}')
            for key in ('solve_seconds', 'true_relative_residual', 'relative_solution_error'):
                check(math.isfinite(float(row[key])), f'Nonfinite {key}: {path}')
            check(float(row['solve_seconds']) > 0 and 0 < float(row['true_relative_residual']) <= 1e-8 and 0 <= float(row['relative_solution_error']) < 1e-4, f'Numeric bounds: {path}')
        records += len(rows)
    check(records == 600 and manifest['timed_records'] == 600 and manifest['selected_csv_files'] == 60, 'Manifest totals')
    print('Verified 76 evidence hashes, exact 60-CSV inventory and 600 recorded numerical outcomes. No MPI solver executed.')


if __name__ == '__main__':
    main()
