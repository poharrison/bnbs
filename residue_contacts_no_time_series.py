import h5py
import pickle
import pandas as pd
import numpy as np
import MDAnalysis as MDA
import os
import time
#from intertools import product

def open_files():
    trajs_f = h5py.File('./trajs.h5', 'r')
    trajectories = trajs_f['trajectories']
    assign_f = h5py.File('./assign.h5', 'r')
    assignments = assign_f['trajlabels']
    contacts_f = h5py.File("/home/poh8/bnbs_storage/data/contacts.h5", "r")

    return(trajectories, assignments, contacts_f, assign_f)

def load_pickles():
    with open('b_statelist.pickle', 'rb') as f:
        b_statelist = pickle.load(f)
    with open('nb_statelist.pickle', 'rb') as f:
        nb_statelist = pickle.load(f)
    with open('no_bind_trajectories.pickle', 'rb') as f:
        nb_traj = pickle.load(f)
    with open('binding_trajectories.pickle', 'rb') as f:
        b_traj = pickle.load(f)
    with open('unique_nb_seg_ids_endstate_0.pickle', "rb") as f:
        nb_set_ids = pickle.load(f)
    with open('unique_b_seg_ids_endstate_0.pickle', "rb") as f:
        b_set_ids = pickle.load(f)
    with open('unique_nb_df_endstate_0.pickle', "rb") as f:
        nb_df = pickle.load(f)
    with open('unique_b_df_endstate_0.pickle', "rb") as f:
        b_df = pickle.load(f)
        return(nb_traj, b_traj, nb_statelist, b_statelist, nb_set_ids, b_set_ids, nb_df, b_df)

class Trajectory():
    def __init__(self, traj_num, traj_label, trajectories, nres1, nres2):
        self.traj_num = traj_num
        self.traj_label = traj_label
        self.traj_data = trajectories[f'traj_652_{traj_num}']['segments'][:]
        self.seg_list = self.traj_data['seg_id']
        self.weights = self.traj_data['weight'] # get weight at each iteration in simulation
        self.iter_list = np.arange(start=1,stop=653,dtype=int)
        self.labels = np.full(652,-1)
        print("trajectory info:", len(self.seg_list), len(self.weights), len(self.iter_list), len(self.labels))
        self.atom_contacts = np.full((652,878,718), 0, dtype=np.uint16)
        self.contacts = np.full((652, nres1, nres2), 0, dtype=np.uint16)
        self.summed_contacts = np.empty(shape=(nres1, nres2), dtype=np.uint16)
        self.flattened_contacts = np.empty(shape=(nres1*nres2)) # flattened version of summed contacts

    def set_labels(self, assign_f):
        assignments = assign_f['trajlabels'] # indexed as iter, seg, tmpt. (use tmpt 20 [last tmpt in iter])
        for iter in range(651):
            iter_assignments = assignments[iter]
            seg_id = self.seg_list[iter]
            self.labels[iter] = iter_assignments[seg_id, 20] # use tmpt 20 (last tmpt in iter)
        # Determine label for last iteration/seg bc. it was not calculated in assign
        iter_652_pcoord = self.traj_data['final_pcoord'][651]
        if iter_652_pcoord[1] > 3:
            state = 0
        elif iter_652_pcoord[0] < 3.5:
            state = 2
        else:
            state = 1
        self.labels[651] = state

    def set_atom_contacts(self, contacts_f):
        for iter in range(1,652):
            try:
                iter_contacts = contacts_f[f'iter_00000{iter:03}']['contacts'] # contacts for given iter
                self.atom_contacts[iter-1] = iter_contacts[self.seg_list[iter-1]]
            except (ValueError, KeyError) as e:
                print("error! ", iter, e)
                # Value error from seg not existing for iter in contacts data (due to missing data)
                # Key error from iter missing from contacts data (only through iter 649 was calculated)
                pass
    def calculate_contacts(self, bn_indices, bs_indices, bn_i, bs_i):
        for iter in range(652):
            iter_contacts = self.atom_contacts[iter]
            residue_subset = iter_contacts[np.ix_(bn_indices, bs_indices)]
            num_of_residue_contacts = np.sum(residue_subset)
            if num_of_residue_contacts < 0:
                num_of_residue_contacts = 0
            self.contacts[iter, bn_i, bs_i] = num_of_residue_contacts

    def sum_contacts(self):
        # sum residue-residue contacts along the time axis
        self.summed_contacts = np.sum(self.contacts, axis=2)

    def flatten_contacts(self):
        self.flatten_contacts = self.summed_contacts.flatten()

    def pickle(self):
        with open(f"traj_{self.traj_num}.pkl") as f:
            pickle.dump(self, f)

    def save_to_csv(self, bn_resnames, bs_resnames):
        dir_path = f"/home/poh8/bnbs_storage/analysis/combined_analysis/inter_residue_contacts_traj_{self.traj_num}"
        weights_and_labels_path = f"{dir_path}/weights_and_labels.csv"
        weights_and_labels = pd.DataFrame(
            data = {"iteration":self.iter_list,
                    "seg_id":self.seg_list,
                    "weight":self.weights,
                    "label":self.labels})
        weights_and_labels.to_csv(weights_and_labels_path, index=False)

        for iter in range(1,653):
            iter_name = f"{dir_path}/iter_{iter:03}.csv"
            iter_df = pd.DataFrame(data = self.contacts[iter-1], index=bn_resnames, columns=bs_resnames)
            iter_df.to_csv(iter_name)

def save_all_traj_to_csv(all_traj, rownames, colnames):
    all_traj_df = pd.DataFrame(data=all_traj, index=rownames, columns=colnames)
    filename = f"/home/poh8/bnbs_storage/analysis/combined_analysis/residue_contacts_no_time_series.csv"
    all_traj_df.to_csv(filename)

def main():
    nb_traj, b_traj, nb_statelist, b_statelist, nb_set_ids, b_set_ids, nb_df, b_df = load_pickles()
    trajectories, assignments, contacts_f, assign_f = open_files()
    nsegs = assign_f['nsegs']
    # Get trajectory dicts (traj_dict[traj_id][<"iteration", "seg_id", "weight", "contacts">])

    u = MDA.Universe("./representative_bound_paths/bound_paths/track_I/trackI_topology.gro")
    bn = u.select_atoms("(bynum 1-1727) and not (name H*)")
    bs = u.select_atoms("(bynum 1728-3159) and not (name H*)")

    bn_map = {}
    for contact_index, atom_index in enumerate(bn.atoms.indices):
        bn_map[atom_index] = contact_index

    bs_map = {}
    for contact_index, atom_index in enumerate(bs.atoms.indices):
        bs_map[atom_index] = contact_index

    bn_reslabels = list(f"{residue.resname} {residue.resnum}" for residue in bn.residues)
    bs_reslabels = list(f"{residue.resname} {residue.resnum}" for residue in bs.residues)
    n_bn_res = len(bn_reslabels)
    n_bs_res = len(bs_reslabels)

    trajectory_objs = []
    for i, traj in enumerate(b_set_ids):
        t = Trajectory(traj, 1, trajectories, n_bn_res, n_bs_res) # traj_label = 1 --> binding trajectory
        if (i == 0 or i == 1):
            t.set_atom_contacts(contacts_f)
        #t.set_labels(assign_f)
        trajectory_objs.append(t)
    for i, traj in emumerate(nb_set_ids):
        t = Trajectory(traj, 0, trajectories, n_bn_res, n_bs_res)  # traj_label = 0 --> nonbinding trajectory
        #t.set_labels(assign_f)
        if (i == 0 or i == 1):
            t.set_atom_contacts(contacts_f)
        trajectory_objs.append(t)

    bn_i = 0
    bs_i = 0
    total_calculations = len(bn.residues) * len(bs.residues) * 651 * 29
    print("total calculations: ", total_calculations)
    calculations = 0
    tic = time.perf_counter()
    lasttime = tic

    contacts_for_iters = []
    print("BN res length: ", len(bn.residues))
    print("BS res length: ", len(bs.residues))

    for bn_res in bn.residues:
        bn_res_atom_indices = bn_res.atoms.select_atoms("not (name H*)").indices
        bn_res_contact_indices = list(bn_map[atom_index] for atom_index in bn_res_atom_indices)
        for bs_res in bs.residues:
            bs_res_atom_indices = bs_res.atoms.select_atoms("not (name H*)").indices
            bs_res_contact_indices = list(bs_map[atom_index] for atom_index in bs_res_atom_indices)
            for traj in traj_obs[0:1]:
                print("trajectory #:", traj.traj_num)
                traj.calculate_contacts(bn_res_contact_indices, bs_res_contact_indices, bn_i, bs_i)
            calculations += 650 * 29
            toc = time.perf_counter()
            currtime = toc-tic
            calc_time = currtime-lasttime
            print(f"time {((currtime)/60):.03}, {calculations}/{total_calculations} complete ({((calculations/total_calculations)*100):.03}%). Est. {(calc_time*(total_calculations-calculations)/(60*652))} remaining")
            lasttime = currtime
            bs_i += 1
        bs_i = 0
        bn_i += 1

    nrows = len(traj_obs)
    ncols = (n_bn_res*n_bs_res) + 1 # number of pairwise combinations, + 1 for label column
    all_traj = np.empty(shape=(nrows, ncols), dtype=np.int32)

    for i, traj in enumerate(traj_obs):
        traj.sum_contacts()
        traj.flatten_contacts()
        all_traj[i,0] = traj.traj_label # first column is trajectory labels
        all_traj[i,1:] = traj.flatten_contacts

    pairwise_names = []
    for bn_name in bn_reslabels:
        for bs_name in bs_reslabels:
            pairwise_names.append([bn_name, bs_name])
    traj_nums = b_set_ids + nb_set_ids
    save_all_traj_to_csv(all_traj, traj_nums, pairwise_names)

if __name__ == "__main__":
    main()
