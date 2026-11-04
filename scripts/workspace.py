"""Use the organized archive without restoring its old directory nesting."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / 'scripts/checkpoint_recovery'

def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def artifact_map():
    config = json.loads((ROOT / 'workspace.json').read_text())
    prefix = config['remote_root'].rstrip('/') + '/'
    mapping = {}
    for row in read_rows(ROOT / 'comparisons/files.csv'):
        stored = ROOT / row['storage_path']
        aliases = json.loads(row['remote_aliases_json'] or '[]')
        if row['source_remote_path']:
            aliases.append(row['source_remote_path'])
        if row['storage_path'].endswith('.run.json'):
            aliases += json.loads(row['consumer_records_json'] or '[]')
        for alias in aliases:
            relative = alias.removeprefix(prefix)
            if alias.startswith(prefix) or not Path(alias).is_absolute():
                if relative in mapping and mapping[relative] != stored:
                    raise ValueError(f'Ambiguous archive alias: {relative}')
                mapping[relative] = stored
    return mapping

def check():
    config = json.loads((ROOT / 'workspace.json').read_text())
    required = ['src/modeling/__main__.py', 'src/synthesize_data/main.py',
                'src/synthesize_data/tabddpm_core/modules.py',
                'scripts/run_corrected_matrix.py', 'scripts/build_corrected_results.py',
                'audit/sdv_api_smoke.py', 'simulated_eval/benchmark.py',
                'scripts/checkpoint_recovery/runtime.zip',
                config['ssh_identity'], config['ssh_known_hosts']]
    for name in required:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    for folder in ['src', 'scripts', 'simulated_eval']:
        for path in (ROOT / folder).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    scores = read_rows(ROOT / 'comparisons/scores_valbest.csv')
    if len(scores) != 602:
        raise ValueError('Expected 602 completed validation-best evaluations')
    for row in scores:
        for field in ['result_path', 'weights_path', 'predictions_path', 'original_record_path']:
            if not (ROOT / row[field]).is_file():
                raise FileNotFoundError(row[field])
    print(json.dumps({'working_root': str(ROOT), 'code_and_artifact_paths_valid': True,
                      'validation_best_records': len(scores),
                      'production_execution': config['remote_host'],
                      'python': sys.executable}, indent=2))

def reports(selection, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new output directory; existing reports are preserved')
    mapping = artifact_map()
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='report_build_', dir=cache) as temporary:
        view = Path(temporary)
        worker = view / 'recovery'
        worker.mkdir()
        lookup = {}
        for name in ['inventory.json', 'files_index.json', 'synthetic.csv', 'mix.csv', 'plot_recipe.py', 'build_reports.py']:
            shutil.copy2(RECOVERY / name, worker / name)
        for row in read_rows(ROOT / 'comparisons/scores_valbest.csv'):
            slug = row['record_relative_path'].replace('/', '__').removesuffix('.run.json')
            alias = worker / 'jobs' / slug / 'result.json'
            alias.parent.mkdir(parents=True)
            source = ROOT / row['result_path']
            os.link(source, alias)
            lookup[str(alias.resolve())] = source
        env = os.environ.copy()
        env['MPLCONFIGDIR'] = str(view / 'matplotlib')
        builder = worker / 'build_reports.py'
        builder.write_text(builder.read_text(encoding='utf-8').replace(
            'Sources: G:/SummerResearch_Retained_20261013_2221.', f'Sources: {ROOT.as_posix()}.'), encoding='utf-8')
        subprocess.run([sys.executable, str(builder)], check=True, env=env)
        output = worker / 'reports'
        if selection == 'delta':
            # The historical recipe is preserved; only its filesystem roots change.
            code = (RECOVERY / 'build_checkpoint_delta_plots.py').read_text(encoding='utf-8')
            code = code.replace("PROJECT = Path('D:/SummerResearch')", 'PROJECT = Path(os.environ["RESEARCH_WORKSPACE_ROOT"])')
            code = code.replace("BASE = PROJECT / 'output/comparison_devf1_20261015_v1'", 'BASE = Path(os.environ["RESEARCH_REPORT_BUILD_ROOT"])')
            code = code.replace("CACHE = PROJECT / '.cache/comparison_devf1_20261015'", 'CACHE = BASE')
            code = code.replace("RETAINED = Path('G:/SummerResearch_Retained_20261013_2221')", 'RETAINED = PROJECT')
            script = worker / 'build_checkpoint_delta_plots.py'
            script.write_text(code, encoding='utf-8')
            env.update(RESEARCH_WORKSPACE_ROOT=str(ROOT), RESEARCH_REPORT_BUILD_ROOT=str(worker))
            subprocess.run([sys.executable, str(script)], check=True, env=env)
            output = output / 'deltas'
        destination.mkdir(parents=True)
        for path in output.iterdir():
            if not path.is_file() or path.name == 'artifact_hashes.json':
                continue
            target = destination / path.name
            if path.suffix != '.csv':
                shutil.copy2(path, target)
                continue
            rows = read_rows(path)
            if not rows:
                shutil.copy2(path, target)
                continue
            for row in rows:
                for field, value in row.items():
                    if not value:
                        continue
                    resolved = None
                    if field == 'original_record':
                        resolved = mapping[value]
                    elif field == 'result':
                        resolved = lookup[str((worker / value).resolve())]
                    elif field != 'retained_record' and (field == 'alternative_result' or field.endswith('_record') or field.endswith('_d2_metadata')):
                        resolved = lookup[str((path.parent / value).resolve())]
                    if resolved is not None:
                        row[field] = Path(os.path.relpath(resolved, destination)).as_posix()
            with target.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in destination.iterdir() if p.is_file()}
        (destination / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    print(destination)

def original_reports(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new output directory')
    mapping = artifact_map()
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='original_report_', dir=cache) as temporary:
        view = Path(temporary)
        for relative, stored in mapping.items():
            if relative.startswith('output/') and relative.endswith(('.run.json', '.d2.json')):
                target = view / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(stored, target)
        (view / 'scripts').mkdir()
        shutil.copy2(ROOT / 'scripts/plot_recent_comparison.py', view / 'scripts/plot_recent_comparison.py')
        (view / 'audit').mkdir()
        for name in ['comparison_tabddpm_snapshot.json', 'comparison_rerun_verification_latest.json',
                     'comparison_news_log_fit_table_hashes.json', 'comparison_housing_fit_table_hashes.json',
                     'latest_comparison_snapshot.json', 'latest_comparison_mix_snapshot.json']:
            shutil.copy2(ROOT / 'audit' / name, view / 'audit' / name)
        env = os.environ.copy()
        env['MPLCONFIGDIR'] = str(view / 'matplotlib')
        for mode in ['synthetic', 'mix']:
            subprocess.run([sys.executable, str(view / 'scripts/plot_recent_comparison.py'),
                            '--train-option', mode], check=True, env=env)
        output = view / 'output/comparisons'
        destination.mkdir(parents=True)
        for source in output.iterdir():
            target = destination / source.name
            if source.suffix != '.csv':
                shutil.copy2(source, target)
                continue
            rows = read_rows(source)
            for row in rows:
                for field, value in row.items():
                    if value and (field.endswith('_record') or field.endswith('_d2_metadata')):
                        row[field] = Path(os.path.relpath(mapping[value], destination)).as_posix()
            with target.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        hashes = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in destination.iterdir() if x.is_file()}
        (destination / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    print(destination)


def remote_help():
    config = json.loads((ROOT / 'workspace.json').read_text())
    import shlex
    command = 'cd ' + shlex.quote(config['remote_root']) + ' && ' + shlex.join([
        config['remote_python'], 'scripts/run_corrected_matrix.py', '--help'])
    subprocess.run(['ssh', '-i', str(ROOT / config['ssh_identity']), '-o', 'IdentitiesOnly=yes',
                    '-o', 'UserKnownHostsFile=' + str(ROOT / config['ssh_known_hosts']),
                    '-o', 'BatchMode=yes', config['remote_host'], command], check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    sub.add_parser('remote-help')
    report = sub.add_parser('reports')
    report.add_argument('--selection', choices=['original', 'valbest', 'delta'], default='valbest')
    report.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'check':
        check()
    elif args.command == 'remote-help':
        remote_help()
    else:
        if args.selection == 'original':
            original_reports(args.out)
        else:
            reports(args.selection, args.out)
