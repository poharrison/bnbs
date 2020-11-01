"""
Script to check that bins and states are being properly assigned.

Notes:
trajlabels is accessed as [iteration][segment][timepoint]
state_labels (pre combination):
    unbound_to_complex:
        0 - unbound
        1 - complex
        2 - unknown
    complex_to_bound:
        0 - bound
        1 - complex
        2 - unknown
desired state labels (post combination):
    0 - unbound
    1 - complex
    3 - bound
"""

import h5py

utc_file = h5py.File("unbound_to_complex/assign.h5", 'r') # shape (651, 2190, 21)
ctb_file = h5py.File("complex_to_bound_analysis/assign.h5", 'r') # shape (651, 2190, 21)
comb_file = h5py.File("combined_analysis/assign.h5", 'r')
west = h5py.File('west.h5', 'r')

utc_tlabels = utc_file['trajlabels'] # shape (651, 2190, 21)
ctb_tlabels = ctb_file['trajlabels'] # shape (651, 2190, 21)
comb_tlabels = comb_file['trajlabels']

test_points = [[312,650,3], [16,0,9], [609,1000,15], [200,70,4]] # iter, seg, timepoint

for set in test_points:
    print("set: ", set)
    print('\tpcoord: ', west['iterations']['iter_00000%s' % format(set[0]+1, '03d')]['pcoord'][set[1],set[2]])
    print('\tutc tlabels: ', utc_tlabels[set[0],set[1],set[2]])
    print('\tctb tlabels: ', ctb_tlabels[set[0],set[1],set[2]])
    print('\tcombined tlabels: ', comb_tlabels[set[0],set[1],set[2]])

# Correct for set 0 (unbound)
# Correct for set 1 (complex)
# Correct for set 2 (complex)
# Correct for set 3 (bound)
