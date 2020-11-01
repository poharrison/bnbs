"""
Script to combine the *trajlabels* dataset from the unbound_to_complex and
complex_to_bound assign.h5 files.

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

utc_file = h5py.File("unbound_to_complex/assign.h5")
ctb_file = h5py.File("complex_to_bound_analysis/assign.h5")

utc_tlabels = utc_file['trajlabels']
ctb_tlabels = ctb_file['trajlabels']

print(shape(utc_tlabels))
print(shape(ctb_tlabels))
