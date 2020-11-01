import pickle
import h5py
import numpy as np

with open('b_statelist.pickle', 'rb') as f:
    b_statelist = pickle.load(f)

with open('nb_statelist.pickle', 'rb') as f:
    nb_statelist = pickle.load(f)

with open('no_bind_trajectories.pickle', 'rb') as f:
    nb_traj = pickle.load(f)

with open('binding_trajectories.pickle', 'rb') as f:
    b_traj = pickle.load(f)

west = h5py.File("west.h5")
weights = west['iterations']['iter_00000652']['seg_index']['weight']

b_weights = list(weights[x] for x in b_traj)

b_arr = np.c_(b_traj, b_weights)

print(b_arr)
