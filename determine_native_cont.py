import MDAnalysis as MDA
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import sys, h5py

#u = MDA.Universe("/home/boltzmann/ESATA/bnbs_pruned/bound_paths/bnbs_centered_crystal.gro")
u = MDA.Universe("./1brs.pdb")
print(u.select_atoms('all').positions.shape)
bn = u.select_atoms("(bynum 1-1727) and not (name H*)")
bs = u.select_atoms("(bynum 1728-3159) and not (name H*)")
bn_rnums = u.select_atoms("(bynum 1-1727) and not (name H*)").resnums
bs_rnums = u.select_atoms("(bynum 1728-3159) and not (name H*)").resnums
bn_rnames = u.select_atoms("(bynum 1-1727) and not (name H*)").resnames
bs_rnames = u.select_atoms("(bynum 1728-3159) and not (name H*)").resnames

bn_coords = bn.positions
bs_coords = bs.positions

print("bn coords shape: ", bn_coords.shape)
print("bs coords shape: ", bs_coords.shape)
distMat = cdist(bn_coords, bs_coords)

print("bs_rnames: ", len(bs_rnames))
print("bn_rnames: ", len(bn_rnames))
print("distmat shape: ", distMat.shape)

size_i, size_j = 878, 718
contact_dist = 5.5

conts = np.zeros((size_i, size_j), dtype=np.uint8)
inds = (distMat <= contact_dist)
conts += np.array(inds, dtype=np.uint8)
print("conts shape: ", conts.shape)
"""
np.save("native_contacts.npy", conts)
"""

bn_arg = bn.select_atoms("resid 59")
bs_asp = bs.select_atoms("resid 35")
