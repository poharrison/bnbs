import h5py
import pickle
import pandas as pd
import numpy as np
import MDAnalysis as MDA
import os
import time
import sys
import memory_profiler
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
        self.atom_contacts = np.full((652,878,718), 0, dtype=np.uint16)
        self.contacts = np.full((652, nres1, nres2), 0, dtype=np.uint16)
        self.summed_contacts = np.empty(shape=(nres1, nres2), dtype=np.uint16)
        self.flattened_contacts = np.empty(nres1*nres2) # flattened version of summed contacts

    def set_labels(self, assign_f):
        assignments = assign_f['trajlabels'] # indexed as iter, seg, tmpt. (use tmpt 0 [last tmpt in iter])
        for iter in range(651):
            iter_assignments = assignments[iter]
            seg_id = self.seg_list[iter]
            self.labels[iter] = iter_assignments[seg_id, 0] # use tmpt 0 (first tmpt in iter)
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
        print(f"TRAJ {self.traj_num} SUM: ", np.sum(self.contacts))
        self.summed_contacts = np.sum(self.contacts, axis=0)
        print(f"TRAJ {self.traj_num} AXIS SUM: ", np.sum(self.summed_contacts))
        return self.summed_contacts

    def flatten_contacts(self):
        self.flattened_contacts = self.summed_contacts.flatten()
        print(f"flattened contacts sum : {np.sum(self.flattened_contacts)}")
        return self.flattened_contacts

    def pickle(self):
        with open(f"/home/poh8/bnbs_storage/analysis/combined_analysis/selected_trajectory_pickles/traj_{self.traj_num}.pkl", 'wb') as f:
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
    print("all traj shape:", all_traj.shape)
    print(all_traj)
    print("len(rownames): ", len(rownames))
    print("len(colnames): ", len(colnames))
    all_traj_df = pd.DataFrame(data=all_traj, index=rownames, columns=colnames)
    filename = f"/home/poh8/bnbs_storage/analysis/combined_analysis/residue_contacts_no_time_series.csv"
    all_traj_df.to_csv(filename)

@profile
def calculate_residue_contacts(bn_residues, bs_residues, bn_map, bs_map, traj_objs):
    bn_i = 0
    bs_i = 0
    for bn_res in bn_residues:
        bn_res_atom_indices = bn_res.atoms.select_atoms("not (name H*)").indices
        bn_res_contact_indices = list(bn_map[atom_index] for atom_index in bn_res_atom_indices)
        for bs_res in bs_residues:
            bs_res_atom_indices = bs_res.atoms.select_atoms("not (name H*)").indices
            bs_res_contact_indices = list(bs_map[atom_index] for atom_index in bs_res_atom_indices)
            for traj in traj_objs:
                traj.calculate_contacts(bn_res_contact_indices, bs_res_contact_indices, bn_i, bs_i)
            bs_i += 1
        bs_i = 0
        bn_i += 1

@profile
def sum_and_flatten_trajs(all_traj, traj_objs, indices):
    for i, traj in enumerate(traj_objs):
        print(traj.traj_num)
        all_traj_index = indices[i]
        print(i, all_traj_index)
        traj.sum_contacts()
        all_traj[all_traj_index,1:] = traj.flatten_contacts()
        print(f"all traj flattened contacts sum : {np.sum(all_traj[i,1:])}")
        all_traj[all_traj_index,0] = traj.traj_label # first column is trajectory labels

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

    print("BN res length: ", len(bn.residues))
    print("BS res length: ", len(bs.residues))

    total_calculations = len(b_set_ids) + len(nb_set_ids)
    calculations = 0
    tic = time.perf_counter()
    lasttime = tic

    nrows = len(b_set_ids) + len(nb_set_ids)
    ncols = (n_bn_res*n_bs_res) + 1 # number of pairwise combinations, + 1 for label column
    all_traj = np.empty(shape=(nrows, ncols), dtype=np.int32)

    selected_trajectories = b_set_ids + nb_set_ids
    ntraj = len(selected_trajectories)

    trajectory_objs = []
    all_traj_indices = [0]
    traj_labels = list(1 for i in range(len(b_set_ids))) + list(0 for i in range(len(nb_set_ids)))
    all_traj[:,0] = traj_labels

    for i, traj in enumerate(selected_trajectories):
        t = Trajectory(traj, traj_labels[i], trajectories, n_bn_res, n_bs_res) # traj_label = 1 --> binding trajectory
        print(f"TRAJECTORY SIZE: {sys.getsizeof(t)}")
        print(f"traj_data size: {sys.getsizeof(t.traj_data)}")
        print(f"atom_contacts size: {sys.getsizeof(t.atom_contacts)}")
        print(f"contacts size: {sys.getsizeof(t.contacts)}")
        print(f"summed_contacts size: {sys.getsizeof(t.summed_contacts)}")
        print(f"flattened_contacts size: {sys.getsizeof(t.flattened_contacts)}")
        total_size_bytes = sys.getsizeof(t) +sys.getsizeof(t.traj_data) + sys.getsizeof(t.atom_contacts) + sys.getsizeof(t.contacts) + sys.getsizeof(t.summed_contacts) + sys.getsizeof(t.flattened_contacts)
        total_size_GB = total_size_bytes/(1e9)
        print(f"total size: {total_size_bytes} B, {total_size_GB} GB")
        #print(f"Memory usage: {memory_profiler.memory_usage()}")
        t.set_atom_contacts(contacts_f)
        t.set_labels(assign_f)
        trajectory_objs.append(t)
        if (i % 8 == 0 or i == ntraj-1):
            if (i != 0 and i != ntraj-1):
                all_traj_indices = list(range(i-7, i+1))
            elif (i == ntraj-1):
                first_index = all_traj_indices[-1] + 1
                all_traj_indices = list(range(first_index, i+1))
            print("Calculating residues!")
            calculate_residue_contacts(bn.residues, bs.residues, bn_map, bs_map, trajectory_objs)
            sum_and_flatten_trajs(all_traj, trajectory_objs, all_traj_indices)
            #for trajectory in trajectory_objs:
                #trajectory.pickle()
                #print(f"Trajectory {trajectory.traj_num} pickled")
            trajectory_objs = []
            calculations += len(all_traj_indices)
            toc = time.perf_counter()
            currtime = toc-tic
            calc_time = currtime-lasttime
            print(f"time {((currtime)/60):.03}, {calculations}/{total_calculations} complete ({((calculations/total_calculations)*100):.03}%). Est. {(calc_time*(total_calculations-calculations)/(60))} remaining")
            lasttime = currtime

    pairwise_names = ["label"]
    for bn_name in bn_reslabels:
        for bs_name in bs_reslabels:
            pairwise_names.append(f"{bn_name}-{bs_name}")

    traj_nums = b_set_ids + nb_set_ids

    #save_all_traj_to_csv(all_traj, traj_nums, pairwise_names)

if __name__ == "__main__":
    main()
