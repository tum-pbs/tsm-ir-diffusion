import os
from phi.jax.flow import *
from tqdm import trange
import numpy as np

math.set_global_precision(64)

# SETTINGS
add_random_noise_to_forcing = False
stabilizing_force = True
cfl_max = 0.7

output_after_frames = 20
rk4_iterations = 25
timsteps = 50 + output_after_frames
resolution = 128

num_datasets = 1000
save_pressure = False
x_size = y_size = 2*PI

# DOMAIN AND FORCE DESCRIPTION
DOMAIN = dict(extrapolation=extrapolation.PERIODIC, bounds=Box(x=x_size, y=y_size), x=resolution, y=resolution)
FORCING = StaggeredGrid(lambda x, y: vec(x=math.sin(4 * y), y=0), **DOMAIN)

def cfl_check(vin, viscosity):
    eps = 1e-16

    dx = x_size / resolution
    dy = y_size / resolution

    v = vin.uniform_values().native('x,y,vector')

    u_max = jax.numpy.max(math.abs(v[:,:,0])) + eps
    v_max = jax.numpy.max(math.abs(v[:,:,1])) + eps

    dt = cfl_max/(u_max/dx + v_max/dy)

    dt_y = dy / v_max
    dt_x = dx / u_max
    dt_visc = (dx*dy)**2 / (2* viscosity * (dx**2 + dy**2))

    cond1 = dt <= dt_y
    cond2 = dt <= dt_x
    cond3 = dt <= dt_visc
    cfl = dt*(u_max/dx + v_max/dy)

    if not (cond1 and cond2 and cond3):
        raise ValueError(f"dt value ({dt}) is higher than the stable region ({dt_x}, {dt_y}, {dt_visc})!")

    return dt

def momentum_equation(v, viscosity):
    advection = advect.finite_difference(v, v, order=6, implicit=Solve('CG', 1e-5, 1e-5))
    diffusion = diffuse.finite_difference(v, viscosity, order=6, implicit=Solve('CG', 1e-5, 1e-5))

    if stabilizing_force:
        return advection + diffusion + FORCING - 0.1*v
    else:
        return advection + diffusion + FORCING

@jit_compile
def compiled_incomp (v,p,dt,viscosity):
  return fluid.incompressible_rk4(momentum_equation, v, p, dt, pressure_order=4, pressure_solve=Solve('CG', 1e-5, 1e-5), viscosity=viscosity)


def rk4_step(v, p, viscosity):
  dt = cfl_check(v, viscosity)
  return compiled_incomp(v,p,dt,viscosity), dt


def kolmogrov_flow(output_folder_name, viscosity):
    total_time = 0

    v = StaggeredGrid(Noise(), **DOMAIN)
    p = CenteredGrid(0, **DOMAIN)

    v_trj = [v]
    p_trj = [p]

    for i in trange(timsteps):
      for _ in range(rk4_iterations):
        vp, dt = rk4_step(v, p, viscosity)
        v, p = vp
        total_time += dt

      v_trj.append(v)
      p_trj.append(p)

    print(f'{total_time = } seconds, dt_avg = {total_time/(timsteps * rk4_iterations)} seconds')

    v_trj = stack(v_trj, batch('time'))
    p_trj = stack(p_trj, batch('time'))

    scene = Scene.create(output_folder_name)

    for i, v_frame in enumerate(v_trj.time):  # write each frame into one file
        if i >= output_after_frames:
            scene.write(velocity=v_frame, frame=i-output_after_frames)

    if save_pressure:
        for i, p_frame in enumerate(p_trj.time):  # write each frame into one file
            if i >= output_after_frames:
                scene.write(pressure=p_frame, frame=i-output_after_frames)


# GENERATING DATASETS
if __name__ == '__main__':

    REs = [100, 200, 1000, 1750, 2500, 4000, 5000]
    seeds = [[0, 99], [0, 239], [0, 239], [100, 199], [0, 239], [0, 239], [200, 299] ]

    for RE in enumerate(REs):
      viscosity = 1/RE
      start_seed = seeds[i][0]
      end_seed = seeds[i][1]

      print(f'\n\n{RE = }\n\n')

      # for i in remaining:
      for i in range(start_seed,end_seed):
          seed = int(i)
          print(f'{seed = }')

          output_folder_name = os.path.join('/datasets/kolmogorov', f"kolmogorov_res{resolution}", f"cfl{cfl_max}", f"re{int(RE)}", f"seed{seed}")
          os.makedirs(output_folder_name, exist_ok=True)
          print('Saving simulation to:', output_folder_name)
          if add_random_noise_to_forcing:
              FORCING += StaggeredGrid(Noise(), **DOMAIN) * (0.01, 0)

          math.seed(seed)
          kolmogrov_flow(output_folder_name, viscosity)
