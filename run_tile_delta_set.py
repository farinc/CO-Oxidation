import subprocess as sp
from pathlib import Path

L = [8, 10, 12]
out_dir = Path('output/')

n_cores = 2 * len(L)

job_name = "tile_delta_set"
command = [
    "qsub",
    "-N", job_name,
    "-pe", "orte", str(n_cores),
    "submit_sweep_job.sh",
    "--out", str(out_dir / job_name),
    "--sweep", "k_o_ads=0:10:0.5",
    "--sweep", "memkm_sites=" + ",".join(str(l) for l in L),
    "--coexistence",
    "--coexistence-axis", "k_o_ads",
]

sp.run(
    command,
    check=True,
    text=True,
    capture_output=False,
)
