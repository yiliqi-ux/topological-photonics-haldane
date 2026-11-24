"""Run M1–M5 and M7–M8; fail if a result or M6 smoke test fails."""

import subprocess
import sys
from pathlib import Path


def main():
    project = Path(__file__).resolve().parent
    scripts = ("haldane.py", "m2_chern.py", "m3_interactive.py", "m4_photonic.py", "m5_edge.py",
               "m7_defect.py", "m8_disorder.py", "test_smoke.py")
    for script in scripts:
        subprocess.run([sys.executable, str(project / script)], cwd=project, check=True)


if __name__ == "__main__":
    main()
