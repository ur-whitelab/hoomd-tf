# -*- coding: iso-8859-1 -*-
# Maintainer: Andrew White

import hoomd
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
        for i in range(3):
            print('------------------------------')
        sysdef = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[1,2])

        updater = hoomd.tensorflow_plugin.update.tensorflow(1)
        hoomd.run(3)

if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])
