import h5py
import pickle
import pandas
import numpy as np
import MDAnalysis as MDA
import os
import time

class Trajectory():
    def __init__(self, traj_num, iter_list, seg_list, weights):
        self.traj_num = traj_num
        self.weights_and_labels = pandas.DataFrame(
            data = {
                "iteration":iter_list,
                "seg_id":seg_list,
                "weight":weights,
                "label":""})
        self.contacts = list()

    def add_contacts(self, iter_contacts):
        self.contacts.append(iter_contacts)

def open_files():
    trajs_f = h5py.File('./trajs.h5', 'r')
    trajectories = trajs_f['trajectories']
    assign_f = h5py.File('./assign.h5', 'r')
    assignments = assign_f['trajlabels']
    #contacts_f = None
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

def get_full_trajectories(trajectories, traj_list):
    trajectory_dict = {}
    iter_list = np.arange(start=1,stop=653,dtype=int)
    time_list = list((iter - 1)*5 for iter in iter_list)
    traj_data = trajectories[f'traj_652_31']['segments'][:]
    # get seg ids for trajectory for each iteration in simulation
    seg_list = traj_data['seg_id']
    # get list of weights for each iteration in simulation
    weights = traj_data['weight']
    trajectory_dict[31] = Trajectory(31, iter_list, seg_list, weights)
    return trajectory_dict

def get_trajectory_contacts_and_labels(trajectories, set_ids, contacts_f, nsegs, assign_f, iter, bn_res_indices, bs_res_indices, bn_i, bs_i):
    assignments = assign_f['trajlabels'] # indexed as iter, seg, tmpt. (use tmpt 0 [first tmpt in iter])
    iter_assignments = assignments[iter-1]
    seg_id = trajectories[31].weights_and_labels.at[iter-1,"seg_id"]
    label = iter_assignments[seg_id, 0] # use tmpt 0 (first tmpt in iter)

    """ calculate inter-residue contacts for that iter,seg and add to df for corresponding traj """
    try:
        contacts = contacts_f[f'iter_00000{iter:03}']['contacts'] # contacts for given iter
        seg_contacts = contacts[seg_id]
        residue_subset = seg_contacts[np.ix_(bn_res_indices, bs_res_indices)]
        num_of_residue_contacts = np.sum(residue_subset)
        trajectories[31].contacts[iter-1].at[bn_i,bs_i] = num_of_residue_contacts
    except (ValueError, KeyError) as error:
        trajectories[31].contacts[iter-1].at[bn_i,bs_i] = -1
        trajectories[31].weights_and_labels.at[iter-1,"label"] = label
    return trajectories

def write_trajs_to_csv(statelist, indices, trajectories, df, type):
    print(f"Saving data to csv's")
    dir_path = "~/bnbs_storage/analysis/combined_analysis/inter_residue_contacts_traj_31/"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    traj = trajectories.get(31)
    weights_and_labels_name = f"{dir_path}/weights_and_labels.csv"
    traj.weights_and_labels.to_csv(weights_and_labels, index=False)
    for i in range(651):
        iter_name = f"{dir_path}/iter_{i:03}.csv"
        traj.contacts[i].to_csv(iter_name)

def main():
    nb_traj, b_traj, nb_statelist, b_statelist, nb_set_ids, b_set_ids, nb_df, b_df = load_pickles()
    trajectories, assignments, contacts_f, assign_f = open_files()

    unique_b_statelist_indices = [b_df[b_df.traj_id == 31].index[0]]
    nsegs = assign_f['nsegs']
    # Get trajectory dicts (traj_dict[traj_id][<"iteration", "seg_id", "weight", "contacts">])
    b_trajectories = get_full_trajectories(trajectories, b_set_ids) # b_set_ids = b traj list

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
    for iter in range(1,652):
        for traj_num in b_trajectories:
            contacts = pandas.DataFrame(index = bn_reslabels, columns = bs_reslabels)
            b_trajectories[traj_num].add_contacts(contacts)
    bn_i = 0
    bs_i = 0
    total_calculations = len(bn.residues) * len(bs.residues) * 651
    print(total_calculations)
    calculations = 0
    tic = time.perf_counter()
    lasttime = tic
    for bn_res in bn.residues:
        bn_res_atom_indices = bn_res.atoms.select_atoms("not (name H*)").indices
        bn_res_contact_indices = list(bn_map[atom_index] for atom_index in bn_res_atom_indices)
        for bs_res in bs.residues:
            bs_res_atom_indices = bs_res.atoms.select_atoms("not (name H*)").indices
            bs_res_contact_indices = list(bs_map[atom_index] for atom_index in bs_res_atom_indices)
            for iter in range (1,652):
                get_trajectory_contacts_and_labels(b_trajectories,
                                                b_set_ids,
                                                contacts_f,
                                                nsegs,
                                                assign_f,
                                                iter,
                                                bn_res_contact_indices,
                                                bs_res_contact_indices,
                                                bn_i,
                                                bs_i)
                calculations += 1
            toc = time.perf_counter()
            currtime = toc-tic
            calc_time = currtime-lasttime
            print(f"time {((currtime)/60):.03}, {calculations}/{total_calculations} complete ({(calculations/total_calculations)*100}%). Est. {calc_time*(total_calculations-calculations)/60:.03} remaining")
            lasttime = currtime
            bs_i += 1
        bn_i += 1

    print("saving trajectories to csv")
    write_trajs_to_csv(b_statelist, unique_b_statelist_indices, b_trajectories, b_df, "binding_trajectories")

if __name__ == "__main__":
    main()
