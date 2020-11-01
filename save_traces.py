# script to save list of traces created inside w_ipa to h5 file

trace_keys = ['auxdata','bins','iteration','pcoord','seg_id','state_labels',
    'states', 'weights']
f = open("trace_%04d.pickle" % i , "wb")
pickle.dump(traces[i], f)
traces_file = h5py.File("tracesdata.h5", "a")
for i in range(len(traces)):
    trace_i = traces_file.create_group("trace_%04d" % i)
    traces_file.create_group('bins')
    traces_file.create_group()
    traces_file.cr
