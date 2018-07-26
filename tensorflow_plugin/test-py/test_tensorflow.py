# -*- coding: iso-8859-1 -*-
# Maintainer: Andrew White

import hoomd, hoomd.md
hoomd.context.initialize()
import hoomd.tensorflow_plugin
import unittest
import os, tempfile, shutil
import numpy as np, math, scipy
import tensorflow as tf

#TODO: write test for changing particle number dynamically

class test_simple(unittest.TestCase):
    def test_ipc_to_tensor(self):
        ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
        shape = [4, 2, 3, 12]
        data = np.array(np.random.random_sample(shape), dtype=np.float32)
        pointer, _ = data.__array_interface__['data']
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor(address=pointer, T=np.float32, shape=shape)
        diff = tf.convert_to_tensor(data) - ipc_to_tensor
        sqe = tf.reduce_sum(diff**2)
        with tf.Session() as sess:
            result = sess.run(sqe)
        assert result < 10**-10

    def test_tensor_to_ipc(self):
        tensor_to_ipc_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/tensor2ipc/lib_tensor2ipc_op.so')
        shape = [8]
        data = np.ones(shape, dtype=np.float32)
        pointer, _ = data.__array_interface__['data']
        tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc(tf.zeros(shape, dtype=tf.float32), address=pointer, maxsize=np.prod(shape))
        with tf.Session() as sess:
            result = sess.run(tensor_to_ipc)
        assert np.sum(data) < 10**-10

class test_compute(unittest.TestCase):
#class test_compute:
    def test_compute_loop(self):
        hoomd.context.initialize()
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
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
                    r = np.array(snapshot.box.min_image(r))
                    rd = np.sqrt(np.sum(r**2))
                    if rd <= rcut:
                        f = -r / rd
                        forces[i, :] += f
                        forces[j, :] -= f
            return forces

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, nneighbor_cutoff=NN, r_cut=rcut, debug_mode=False)
        for i in range(1):
            hoomd.run(1)
            py_forces = compute_forces(system)
            for j in range(N):
                np.testing.assert_allclose(system.particles[j].net_force, py_forces[j, :], rtol=1e-5)


if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])
