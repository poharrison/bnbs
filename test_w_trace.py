import os, shutil, tempfile

import unittest

import h5py

import H5Diff

class Test_W_Trace(unittest.TestCase, CommonToolTest):

    test_name = 'W_TRACE'
    ref_traj_file = ''

    def test_run_w_assign(self):
        '''Testing if w_tracen runs as expected and the assign.h5 file looks good.'''

        temp = tempfile.TemporaryDirectory(dir = "./")
        os.chdir(temp.name)
        shutil.copy2('refs/west.cfg', './')
        shutil.copy2('refs/west.h5', './')
        os.system("w_trace --config-from-file --scheme TEST")
        assert os.path.isfile('./ANALYSIS/TEST/assign.h5'), "The assign.h5 file was not generated."

        with h5py.File('./ANALYSIS/TEST/assign.h5', 'r') as f:
            assert 'assignments' in list(f.keys()), "'assignments' group not in output file"

        os.chdir("../")
        temp.cleanup()

    def test_h5_file_content(self):
        os.system("w_trace ")
        diff = H5Diff()
