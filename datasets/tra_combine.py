from phi.torch.flow import *
import os
import json
import numpy as np
from tqdm import trange, tqdm


path = "128_tra"

train_sims = 33
sims = 41
sim_type = 'test'

if sim_type == 'train':
    consider_sims = train_sims
else:
    consider_sims = sims - train_sims

frames = 1001
fields = ["velocity", "pressure", "density"]
params_names = ['Drag Coefficient', 'Lift Coefficient', 'Reynolds Number', 'Mach Number']

velocity = np.zeros((consider_sims, frames, 2, 128, 64))
pressure = np.zeros((consider_sims, frames, 1, 128, 64))
density = np.zeros_like(pressure)

mask = np.zeros((consider_sims, 128, 64))
mach_number = np.zeros((consider_sims))

avoid_machs = list(np.arange(start = 0.53, stop = 0.64, step = 0.01))
avoid_machs += list(np.arange(start = 0.69, stop = 0.9, step = 0.01))

avoid_machs = [round(value, 2) for value in avoid_machs]

assert len(avoid_machs) == train_sims

considered_machs = []

sim_num = -1

print(sim_type)

for sim in trange(sims):
    sim_name = f'sim_{sim:06d}'
    obsMask = np.load(os.path.join(path, sim_name, "obstacle_mask.npz"))['arr_0']
    
    # load sim info (Ma, Re, ...)
    f = open(os.path.join(path, sim_name, "src", "description.json"))            
    params = json.load(f)
    f.close()
    
    condition = not (params[params_names[-1]] in avoid_machs) if sim_type == 'train' else (params[params_names[-1]] in avoid_machs)

    if condition:
        print(sim, params[params_names[-1]])
        continue
    
    sim_num += 1

    mach_number[sim_num] = params[params_names[-1]]
    mask[sim_num] = obsMask

    considered_machs.append(mach_number[sim_num])

    for field in fields:
        for frame in range(frames):
            filename = f"{field}_{frame:06d}.npz"
            filepath = os.path.join(path, sim_name, filename)

            data = np.load(filepath)['arr_0']

            if field == 'velocity':
                velocity[sim_num, frame] = data
            elif field == 'pressure':
                pressure[sim_num, frame] = data
            else:
                density[sim_num, frame] = data


print(len(considered_machs), considered_machs)

np.save(path+f'_velocity_{sim_type}.npy', velocity)
np.save(path+f'_pressure_{sim_type}.npy', pressure)
np.save(path+f'_density_{sim_type}.npy', density)
np.save(path+f'_mask_{sim_type}.npy', mask)
np.save(path+f'_mach_number_{sim_type}.npy', mach_number)