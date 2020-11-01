import h5py
import pickle
import pandas

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
    with open('unique_nb_seg_ids_endstate_0.pickle', "rb") as f:
        b_set_ids = pickle.load(f)

    return(nb_traj, b_traj, nb_statelist, b_statelist, nb_set_ids, b_set_ids)

def get_trajectory_contacts(trajectories):
    contacts_file = File("../contacts.h5", "r")
    contacts = open(contacts_file)
    print(trajectories)

def main():
    nb_traj, b_traj, nb_statelist, b_statelist, nb_set_ids, b_set_ids = load_pickles()
    nb_trajectories = get_full_trajectories(nb_traj)

if __name__ == "__main__":
    main()
