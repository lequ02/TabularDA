"""Build the concise presentation of the frozen real-data analysis."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('build_concise_report.py')), run_name='__main__')
