# This is my script hopefully recreating results of determine_native_contacts.py
# to make sure we have right metadata about contacts h5 files
import MDAnalysis as MDA
u = MDA.Universe("./1brs.pdb") # https://www.rcsb.org/structure/1BRS

# From PDB file:
"""
COMPND    MOL_ID: 1;
COMPND   2 MOLECULE: BARNASE;
COMPND   3 CHAIN: A, B, C;
"""
# <AtomGroup with 2581 atoms>
bn_nowat  = u.select_atoms('segid A and not (resname HOH)') +
            u.select_atoms('segid B and not (resname HOH)') +
            u.select_atoms(' segid C and not (resname HOH)')


# From PDB file:
"""
COMPND   6 MOL_ID: 2;
COMPND   7 MOLECULE: BARSTAR;
COMPND   8 CHAIN: D, E, F;
"""
# <AtomGroup with 2059 atoms>
bs_nowat  = u.select_atoms('segid D and not (resname HOH)') +
            u.select_atoms('segid E and not (resname HOH)') +
            u.select_atoms('segid F and not (resname HOH)')

bn_coords = bn_nowat.positions
bs_coords = bs_nowat.positions

distMat = cdist(bn_coords, bs_coords) # calculate euclidean distance

bn_rnums = bn_nowat.resnums
bn_rnames = bn_nowat.resnames
unique_bn_rnums = np.unique(bn_rnums) # shape: (110,)
bs_rnums = bs_nowat.resnums
bn_rnames = bs_nowat.resnames
unique_bs_rnums = np.unique(bs_rnums) # shape: (89,)


distMat = cdist(bn_coords, bs_coords)
