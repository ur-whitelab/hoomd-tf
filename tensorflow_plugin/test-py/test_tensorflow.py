# -*- coding: iso-8859-1 -*-
# Maintainer: Andrew White

import hoomd, hoomd.md
hoomd.context.initialize()
import hoomd.tensorflow_plugin
import unittest
import os
import numpy as np

class test_simple(unittest.TestCase):
    def test_constructor(self):
        sysdef = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[1,2])

        updater = hoomd.tensorflow_plugin.update.tensorflow(4)

    def test_ipc_to_tensor_import(self):
        import tensorflow as tf
        #TODO: pick a better path
        ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor

class test_updater(unittest.TestCase):
    def test_updater_loop(self):
        hoomd.context.initialize()
        hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nl_c = hoomd.md.nlist.cell(check_period=1)
        lj = hoomd.md.pair.lj(r_cut=2.5, nlist=nl_c)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1.2, tau=0.5)
        updater = hoomd.tensorflow_plugin.update.tensorflow(1)
        hoomd.run(1)

if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])
