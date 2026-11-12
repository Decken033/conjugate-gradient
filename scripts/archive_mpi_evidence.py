"""Import selected historical MPI evidence without overwriting differing files."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT.parent / 'conjugate-gradient-backup' / 'conjugate-gradient'
JOBS = ('1482451', '1482491', '1495684', '1502451', '1509066', '1509067')
LOGS = ('dsa5208_mpi_tests.o1482726', 'dsa5208_mpi_scaling_512.o1482451',
        'dsa5208_mpi_scaling_1024.o1482491', 'dsa5208_mpi_thermal2.o1495684',
        'dsa5208_mpi_scaling_3d.o1502451', 'dsa5208_mpi_strong_256.o1509066',
        'dsa5208_mpi_weak_256.o1509067')
SCRIPTS = ('test_mpi_correctness.pbs', 'benchmark_mpi_scaling.pbs',
           'benchmark_mpi_scaling_1024.pbs', 'benchmark_mpi_thermal2.pbs',
           'benchmark_mpi_scaling_3d.pbs', 'benchmark_mpi_strong_256.pbs',
           'benchmark_mpi_weak_256.pbs')


def main():
    pairs = []
    selected = sorted(p for p in (BACKUP / 'results/mpi').rglob('*.csv')
                      if any(job in p.name for job in JOBS))
    if len(selected) != 60:
        raise ValueError(f'Expected 60 selected CSVs, got {len(selected)}')
    for p in selected:
        with p.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 10 or {r['run_index'] for r in rows} != {str(i) for i in range(10)}:
            raise ValueError(f'Incomplete CSV: {p}')
        pairs.append((p, ROOT / p.relative_to(BACKUP), 'raw_csv'))
    for name in LOGS:
        pairs.append((BACKUP / name, ROOT / 'results/mpi/evidence/logs' / name, 'historical_log'))
    for name in SCRIPTS:
        pairs.append((BACKUP / 'scripts/vanda' / name,
                      ROOT / 'docs/task_b/evidence/scripts' / name, 'recovered_script'))
    for kind, job in (('strong', '1509066'), ('weak', '1509067')):
        rel = Path(f'results/mpi/extension/{kind}_{job}.stdct-mgmt-02/hosts.txt')
        pairs.append((BACKUP / rel, ROOT / rel, 'hostfile'))
    # Preflight everything before copying. Existing files must be byte-identical.
    for src, dst, _ in pairs:
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise ValueError(f'Refusing to overwrite differing file: {dst}')
    manifest = ROOT / 'results/mpi/evidence/manifest.json'
    if manifest.exists():
        raise FileExistsError(f'Read/review existing manifest before replacing: {manifest}')
    entries = []
    for src, dst, role in pairs:
        data = src.read_bytes()
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            print('IMPORTED', dst.relative_to(ROOT).as_posix())
        entries.append({'path': dst.relative_to(ROOT).as_posix(), 'role': role,
                        'source_path': src.relative_to(BACKUP).as_posix(),
                        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    payload = {'schema_version': 1, 'source': 'Vanda download backup: conjugate-gradient-backup/conjugate-gradient',
               'note': 'Hashes computed during local import, not historical submission-time hashes. No solver rerun.',
               'selected_csv_files': 60, 'timed_records': 600, 'files': entries}
    manifest.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'Archived {len(entries)} files; 60 CSVs / 600 timed records; existing matching files preserved.')


if __name__ == '__main__':
    main()
