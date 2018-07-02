# -*- coding: iso-8859-1 -*-
# Maintainer: Andrew White

import hoomd, hoomd.md
hoomd.context.initialize()
import hoomd.tensorflow_plugin
import unittest
import os
import numpy as np, math

#TODO: write test for changing particle number dynamically

class test_simple(unittest.TestCase):
    def test_ipc_to_tensor_import(self):
        import tensorflow as tf
        #TODO: pick a better path
        ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor

class test_compute(unittest.TestCase):
    def test_compute_loop(self):
        hoomd.context.initialize()
        hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nl_c = hoomd.md.nlist.cell(check_period=1)
        lj = hoomd.md.pair.lj(r_cut=2.5, nlist=nl_c)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1.2, tau=0.5)
        tfcompute = hoomd.tensorflow_plugin.tfcompute.tensorflow(1)
        #read out buffer containing hoomd positions
        pos0 = tfcompute.get_positions_array()
        hoomd.run(5)
        pos1 = tfcompute.get_positions_array()
        force1 = tfcompute.get_forces_array()

        assert(pos1.shape == force1.shape)
        for a,b in zip(pos1, force1):
            if(np.sum(a**2) + np.sum(b**2) > 0):
                #stupid generous fp comparison
                assert np.max(np.abs(b - a)) / np.max(np.concatenate((np.abs(a),np.abs(b)))) < 10**-6, '{} and {} are not close'.format(a,b)
        assert(np.sum(pos1**2) != 0)


if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])
