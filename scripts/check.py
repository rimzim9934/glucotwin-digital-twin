"""Dependency-free runner for the assertion-based tests (also pytest-compatible)."""
from pathlib import Path
import runpy
root=Path(__file__).resolve().parents[1]
count=0
for file in sorted((root/'tests').glob('test_*.py')):
    namespace=runpy.run_path(str(file))
    for name,fn in namespace.items():
        if name.startswith('test_') and callable(fn):fn();count+=1;print('PASS',name)
print(f'{count} tests passed')
