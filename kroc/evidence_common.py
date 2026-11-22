"""Shared plumbing of the scripts that write tracked evidence to kroc/evidence."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import yaml

CONFIG=Path('configs/payload_circle_T2_J2_45_n0.yaml')
EVIDENCE=Path('kroc/evidence')
TEST_SEEDS=[7207+10*i for i in range(10)]
TUNING_SEEDS=[7307,7317,7327]
SOURCES=('controllers/rollout_kernels.py','controllers/payload_kernel_mppi.py',
         'controllers/payload_stabilized_mppi.py','controllers/payload_mppi.py',
         'dynamics/scalar_codegen.py','kroc/run_payload_cube_tcp.py','kroc/evidence_common.py',
         str(CONFIG))


class Simulated(Exception):
    """Carries the raw result out before the slow serial mesh audit of run_payload."""


def load_config():
    return yaml.safe_load(CONFIG.read_text())


def simulate(config,seed,case,modify=None):
    """Run the exact run_payload wiring; ``modify`` may edit the run_mppi options."""
    from kroc import run_payload_cube_tcp as runner
    original=runner.run_mppi

    def intercept(reference,**options):
        if modify:modify(options)
        raise Simulated(original(reference,**options))

    runner.run_mppi=intercept
    try:
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            runner.run_payload(config,seed,case,output=Path(folder)/'run')
    except Simulated as done:
        return done.args[0]
    finally:
        runner.run_mppi=original
    raise RuntimeError('run_payload returned without simulating')


def provenance(script):
    import importlib.metadata
    import torch

    def shell(*command):
        try:return subprocess.check_output(command,text=True).strip()
        except (OSError,subprocess.CalledProcessError):return None

    governor=Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')),None)
    return dict(
        git_parent_commit=shell('git','rev-parse','HEAD'),
        gpu=shell('nvidia-smi','--query-gpu=name,driver_version,power.limit','--format=csv,noheader'),
        cpu=cpu,cpu_governor=governor.read_text().strip() if governor.exists() else None,
        os=platform.platform(),python=platform.python_version(),torch=torch.__version__,
        triton=importlib.metadata.version('triton'),
        pytorch_mppi=importlib.metadata.version('pytorch-mppi'),
        source_sha256={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                       for path in (*SOURCES,str(script))})


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=1,allow_nan=False)+'\n')
