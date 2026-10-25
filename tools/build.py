"""Build the small, serial C99 row-sweep kernel from source."""
from pathlib import Path
import os
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def build():
    sources = [ROOT / 'analysis' / name for name in ('row_sweeps.c', 'affine_sweeps.c')]
    target = ROOT / 'analysis/row_sweeps.so'
    if target.exists() and target.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return target
    compiler = shlex.split(os.environ.get('CC', 'cc'))
    subprocess.run(compiler + ['-std=c99', '-O2', '-fPIC', '-shared'] +
                   [str(p) for p in sources] + ['-o', str(target)], check=True)
    return target

if __name__ == '__main__':
    print(build())
