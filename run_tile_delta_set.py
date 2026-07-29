import subprocess as sp
from pathlib import Path

L = [8, 10, 12]
Delta_scale = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
out_dir = Path('output/')

for l in L:
    for i,delta_scale in enumerate(Delta_scale):
        prefix_job_name = f"tile_{l}_delta{i}"
        command = [
            "qsub",
            "-N", prefix_job_name,
            "-pe", "orte", "24",
            "submit_sweep_job.sh",
            "--out", out_dir.joinpath(prefix_job_name),
            "--memkm-sites", f"{l}",
            "--delta-scale", f"{delta_scale}",
        ]

        result = sp.run(
            command,
            check=True,
            text=True,
            capture_output=False,
        )

