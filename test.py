import numpy as np
import lc_model_opt
import timeit
from src.cuda_dip_modeller.cuda_dip_modeller import dip_modeller

# test size
ngrid = 1000
n_lc_points = 1000
n_runs = 1

# test system parameters
xx = np.linspace(-3, 3, n_lc_points)
yy = np.full(n_lc_points, 0.2, dtype=np.float64)
transparency = 0.0
rg, rmaj, rmin = 1.0, 2.0, 1.0
theta = 0.4
u = 0.68, 0.24

# initialise cpu code
xgrid, ygrid, mugrid, step = lc_model_opt.make_mugrid(ngrid)

# initialise gpu code
dm = dip_modeller(ngrid)

def run_cpu():
    return lc_model_opt.getlc(
        xx, yy, transparency, rg, rmaj, rmin, theta, *u, xgrid, ygrid, mugrid, step
    )


def run_gpu():
    pass


print(f" -- params --\n{ngrid = }\n{n_lc_points = }")
print("-- benchmarks --")
time_cpu = timeit.timeit(run_cpu, number=n_runs)
print(f"cpu: {time_cpu / n_runs * 1000 : .4f} ms per run")
time_gpu = timeit.timeit(run_gpu, number=n_runs)
print(f"gpu: {time_gpu / n_runs * 1000 : .4f} ms per run")
