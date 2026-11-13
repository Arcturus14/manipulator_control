"""Install auxiliary packages, never replace the existing CUDA Torch/Triton."""
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    import torch
    import triton
    if torch.version.cuda is None or not hasattr(torch.library,'custom_op'):
        raise SystemExit('CUDA-enabled PyTorch with torch.library.custom_op is required. Use the supplied NVIDIA container.')
    protected={name:importlib.metadata.version(name) for name in ('torch','triton')}
    with tempfile.TemporaryDirectory(prefix='kroc-spark-pip-') as tmp:
        constraints=Path(tmp)/'protected.txt'
        constraints.write_text(''.join(f'{k}=={v}\n' for k,v in protected.items()))
        subprocess.run([sys.executable,'-m','pip','install','--constraint',str(constraints),
                        '--requirement',str(Path(__file__).with_name('requirements.txt'))],check=True)
    for name,version in protected.items():
        if importlib.metadata.version(name)!=version:
            raise RuntimeError(f'{name} was unexpectedly replaced')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    print('Protected CUDA runtime packages:',protected)


if __name__=='__main__':main()
