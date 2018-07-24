# -*- coding: iso-8859-1 -*-
# Maintainer: Andrew White

import hoomd, hoomd.md
hoomd.context.initialize()
import hoomd.tensorflow_plugin
import unittest
import os, tempfile, shutil
import numpy as np, math, scipy

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
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        lj = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all())

        #This assumes you have run the code in models/test-model/build.py
        save_loc = '/tmp'
        #
        def compute_forces(system):
            snapshot = system.take_snapshot()
            position = snapshot.particles.position
            forces = np.zeros((N, 3))
            for i in range(N):
                for j in range(i + 1, N):
                    r = position[j] - position[i]
                    rd = np.sqrt(np.sum(r**2))
                    if rd <= rcut:
                        f = -r / rd
                        forces[i, :] += f
                        forces[j, :] -= f
            return forces

        tfcompute = hoomd.tensorflow_plugin.tfcompute.tensorflow(save_loc, nlist, nneighbor_cutoff=NN)
        for i in range(1):
            hoomd.run(1)
            py_forces = compute_forces(system)
            for j in range(N):
                print(system.particles[j].net_force, py_forces[j, :])


if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])
