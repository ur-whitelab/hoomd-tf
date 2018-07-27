# -*- coding: iso-8859-1 -*-
# Maintainer: Andrew White

import hoomd, hoomd.md
hoomd.context.initialize()
import hoomd.tensorflow_plugin
import unittest
import os, tempfile, shutil, pickle
import numpy as np, math, scipy
import tensorflow as tf

#TODO: write test for changing particle number dynamically

def compute_forces(system, rcut):
    '''1 / r^2 force'''
    snapshot = system.take_snapshot()
    position = snapshot.particles.position
    N = len(position)
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

class test_ipc(unittest.TestCase):
    def test_ipc_to_tensor(self):
        ipc_to_tensor_module = hoomd.tensorflow_plugin.tfmanager.load_op_library('ipc2tensor')
        shape = [9, 4, 8]
        data = np.array(np.random.random_sample(shape), dtype=np.float32)
        pointer, _ = data.__array_interface__['data']
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor(address=pointer, T=np.float32, shape=shape)
        diff = tf.convert_to_tensor(data) - ipc_to_tensor
        sqe = tf.reduce_sum(diff**2)
        with tf.Session() as sess:
            result = sess.run(sqe)
        assert result < 10**-10

    def test_tensor_to_ipc(self):
        tensor_to_ipc_module = hoomd.tensorflow_plugin.tfmanager.load_op_library('tensor2ipc')
        shape = [8, 3, 2]
        data = np.ones(shape, dtype=np.float32)
        pointer, _ = data.__array_interface__['data']
        tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc(tf.zeros(shape, dtype=tf.float32), address=pointer, maxsize=np.prod(shape))
        with tf.Session() as sess:
            result = sess.run(tensor_to_ipc)
        assert np.sum(data) < 10**-10

class test_builder(unittest.TestCase):
    def disabled_test_simple_potential(self):
        graph = hoomd.tensorflow_plugin.GraphBuilder(9, 9 - 1)
        with tf.name_scope('force-calc') as scope:
            nlist = graph.nlist[:, :, :3]
            neighs_rs = tf.norm(nlist, axis=1, keepdims=True)
            #no need to use netwon's law because nlist should be double counted
            fr = tf.multiply(-1.0, tf.multiply(tf.reciprocal(neighs_rs), nlist), name='nan-pairwise-forces')
            with tf.name_scope('remove-nans') as scope:
                zeros = tf.zeros(tf.shape(nlist))
                real_fr = tf.where(tf.is_nan(fr), zeros, fr, name='pairwise-forces')
            forces = tf.reduce_sum(real_fr, axis=1, name='forces')
        graph.save(force_tensor=forces, model_directory='/tmp/test-simple-potential-model')

        #check graph info
        with open('/tmp/model/graph_info.p', 'rb') as f:
            gi = pickle.load(f)
            assert gi['forces'] != 'forces:0'
            assert tf.get_default_graph().get_tensor_by_name(gi['forces']).shape[1] == 4

    def disabled_test_gradient_potential(self):
        graph = hoomd.tensorflow_plugin.GraphBuilder(9, 9 - 1)
        with tf.name_scope('force-calc') as scope:
            nlist = graph.nlist[:, :, :3]
            neighs_rs = tf.norm(nlist, axis=1)
            energy = graph.safe_div(numerator=tf.ones(neighs_rs.shape, dtype=neighs_rs.dtype), denominator=neighs_rs, name='energy')
        forces = graph.compute_forces(energy)
        graph.save(force_tensor=forces, model_directory='/tmp/test-gradient-potential-model')

    def disabled_test_noforce_graph(self):
        graph = hoomd.tensorflow_plugin.GraphBuilder(9, 9 - 1, output_forces=False)
        nlist = graph.nlist[:, :, :3]
        neighs_rs = tf.norm(nlist, axis=1)
        energy = graph.safe_div(numerator=tf.ones(neighs_rs.shape, dtype=neighs_rs.dtype), denominator=neighs_rs, name='energy')
        graph.save('/tmp/test-noforce-model', out_node=energy)


class test_compute(unittest.TestCase):
    def disabled_test_compute_force_overwrite(self):
        hoomd.context.initialize()
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all())

        #This assumes you have succeeded in the above test_builder suite
        save_loc = '/tmp/test-simple-potential-model'

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, nneighbor_cutoff=NN, r_cut=rcut, debug_mode=False)
        for i in range(3):
            hoomd.run(1)
            py_forces = compute_forces(system, rcut)
            for j in range(N):
                np.testing.assert_allclose(system.particles[j].net_force, py_forces[j, :], rtol=1e-5)

    def disabled_test_gradient_potential_forces(self):
        hoomd.context.initialize()
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all())

        #This assumes you have succeeded in the above test_builder suite
        save_loc = '/tmp/test-gradient-potential-model'

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, nneighbor_cutoff=NN, r_cut=rcut, debug_mode=False)
        for i in range(3):
            hoomd.run(1)
            py_forces = compute_forces(system, rcut)
            for j in range(N):
                np.testing.assert_allclose(system.particles[j].net_force, py_forces[j, :], rtol=1e-5)

    def disabled_test_compute_force_ignore(self):
        hoomd.context.initialize()
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all())

        #This assumes you have succeeded in the above test_builder suite
        save_loc = '/tmp/test-simple-potential-model'

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, nneighbor_cutoff=NN, r_cut=rcut, debug_mode=False, force_mode='ignore')
        for i in range(3):
            hoomd.run(1)
            for j in range(N):
                np.testing.assert_allclose(system.particles[j].net_force, [0,0,0], rtol=1e-5)

    def test_compute_noforce_graph(self):
        hoomd.context.initialize()
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all())

        #This assumes you have succeeded in the above test_builder suite
        save_loc = '/tmp/test-noforce-model'

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, nneighbor_cutoff=NN, r_cut=rcut, debug_mode=False, force_mode='output')
        for i in range(3):
            hoomd.run(1)
            for j in range(N):
                np.testing.assert_allclose(system.particles[j].net_force, [0,0,0], rtol=1e-5)


if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])
