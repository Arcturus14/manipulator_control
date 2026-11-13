"""Snapshot only local import dependencies, assets and one comparison input.

Never copies environments, caches, the complete repo, or experiment datasets.
Refuses to overwrite an existing snapshot or archive.
"""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ENTRYPOINTS=['kroc.benchmark_collision_fusion','kroc.validate_collision_solve',
    'kroc.search_circle_floor25','kroc.summarize_collision_fusion',
    'kroc.test_self_collision_cost','kroc.test_circle_reference',
    'tests.test_dob_acceleration_filter','tests.test_kroc_constraint_scenarios']
EXCLUDED={'controllers.sqp','controllers.ilqr'}


def module_path(name):
    path=ROOT.joinpath(*name.split('.'))
    for candidate in (path.with_suffix('.py'),path/'__init__.py'):
        if candidate.is_file():return candidate
    return None


def source_closure():
    pending=list(ENTRYPOINTS);seen=set();files=set()
    while pending:
        name=pending.pop()
        if name in seen or name in EXCLUDED:continue
        seen.add(name);path=module_path(name)
        if path is None:continue
        files.add(path)
        for parent in path.relative_to(ROOT).parents:
            init=ROOT/parent/'__init__.py'
            if init.is_file() and init!=path:
                pending.append('.'.join(parent.parts))
        package=name if path.name=='__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Import):pending.extend(n.name for n in node.names)
            elif isinstance(node,ast.ImportFrom):
                target=node.module or ''
                if node.level:
                    if not package:continue
                    target=importlib.util.resolve_name('.'*node.level+target,package)
                if target:
                    pending.append(target)
                    pending.extend(target+'.'+n.name for n in node.names if n.name!='*')
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-dir',type=Path,default=ROOT/'kroc'/'spark_bundle')
    p.add_argument('--archive',type=Path,default=ROOT/'kroc'/'spark_bundle_20261112.tar.gz')
    p.add_argument('--refresh-tools',action='store_true',
                   help='Rehash known bundle files after setup-script fixes; retain the source snapshot')
    a=p.parse_args();bundle=a.bundle_dir.resolve();source=bundle/'source'
    if a.refresh_tools:
        manifest=json.loads((bundle/'bundle_manifest.json').read_text())
        included=[bundle/name for name in manifest['files_sha256']]
        if not all(f.is_file() for f in included):raise SystemExit('Snapshot incomplete')
        if a.archive.exists():
            with tarfile.open(a.archive,'r:gz') as old:
                if not all(n.startswith(bundle.name+'/') for n in old.getnames()):
                    raise SystemExit('Archive is not this bundle; refusing replacement')
        manifest['files_sha256']={str(f.relative_to(bundle)):hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in included}
        (bundle/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
        with tarfile.open(a.archive,'w:gz') as archive:
            for path in sorted([*included,bundle/'bundle_manifest.json']):
                archive.add(path,arcname=str(Path(bundle.name)/path.relative_to(bundle)),recursive=False)
        print('Refreshed setup-file hashes and archive; source snapshot retained')
        return
    if source.exists() or (bundle/'bundle_manifest.json').exists() or a.archive.exists():
        raise SystemExit('Refusing to overwrite an existing snapshot/archive; choose a fresh bundle directory.')
    if not (bundle/'run.py').is_file():raise SystemExit('Bundle setup scripts missing')
    files=source_closure()
    files.update((ROOT/'kroc'/'assets'/'ur5_cb3_collision').glob('*'))
    files.update(ROOT/name for name in (
        'kroc/results/circle_collision_cost_boundary_pilot/s1_seed7_directional25_5.npz',
        'kroc/results/collision_free_circle_screen_outward/candidates.json'))
    for path in sorted(files):
        if not path.is_file():continue
        dest=source/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    reference=bundle/'reference';reference.mkdir(exist_ok=False)
    for name,original in {
        'gtx1650_micro.json':'collision_fusion_micro_final_20261112',
        'gtx1650_closedloop.json':'collision_fusion_comparison_20261112',
        'gtx1650_solve_validation.json':'collision_fusion_solve_validation_20261112',
    }.items():
        shutil.copy2(ROOT/'kroc'/'results'/original/'results.json',reference/name)
    included=[f for f in bundle.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'MPPI-only snapshot; no SQP/acados, venv, caches or complete datasets',
        'entrypoints':ENTRYPOINTS,'excluded_optional_modules':sorted(EXCLUDED),
        'target_test_status':'Spark ARM64/container build not yet tested; local snapshot checks required',
        'asset_source':'https://github.com/ros-industrial/universal_robot/tree/noetic-devel/ur_description',
        'files_sha256':{str(f.relative_to(bundle)):hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in sorted(included)}}
    (bundle/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(a.archive,'x:gz') as archive:
        for path in sorted([*included,bundle/'bundle_manifest.json']):
            archive.add(path,arcname=str(Path(bundle.name)/path.relative_to(bundle)),recursive=False)
    print(json.dumps({'bundle':str(bundle),'archive':str(a.archive),
        'source_files':len(files),'verified_files':len(included),
        'archive_bytes':a.archive.stat().st_size},indent=2))


if __name__=='__main__':main()
