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
                f = -r / rd * 1 / rd**2
                forces[i, :] += f
                forces[j, :] -= f
    return forces

class test_ipc(unittest.TestCase):
    def dtest_ipc_to_tensor(self):
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

    def dtest_tensor_to_ipc(self):
        tensor_to_ipc_module = hoomd.tensorflow_plugin.tfmanager.load_op_library('tensor2ipc')
        shape = [8, 3, 2]
        data = np.ones(shape, dtype=np.float32)
        pointer, _ = data.__array_interface__['data']
        tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc(tf.zeros(shape, dtype=tf.float32), address=pointer, maxsize=np.prod(shape))
        with tf.Session() as sess:
            result = sess.run(tensor_to_ipc)
        assert np.sum(data) < 10**-10

class test_compute(unittest.TestCase):
    def test_compute_force_overwrite(self):
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

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, r_cut=rcut, debug_mode=False)
        for i in range(3):
            hoomd.run(1)
            py_forces = compute_forces(system, rcut)
            for j in range(N):
                np.testing.assert_allclose(system.particles[j].net_force, py_forces[j, :], rtol=1e-5)

    def test_gradient_potential_forces(self):
        hoomd.context.initialize()
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=4)

        #This assumes you have succeeded in the above test_builder suite
        save_loc = '/tmp/test-gradient-potential-model'

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, r_cut=rcut, debug_mode=False)
        for i in range(2):
            hoomd.run(100)
            py_forces = compute_forces(system, rcut)
            for j in range(N):
                np.testing.assert_allclose(system.particles[j].net_force, py_forces[j, :], atol=1e-2)

    def test_compute_force_ignore(self):
        hoomd.context.initialize()
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=4, seed=1)

        #This assumes you have succeeded in the above test_builder suite
        save_loc = '/tmp/test-simple-potential-model'

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, r_cut=rcut, debug_mode=False, force_mode='ignore')
        for i in range(3):
            hoomd.run(100)
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

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, r_cut=rcut, debug_mode=False, force_mode='output')
        for i in range(3):
            hoomd.run(1)
            for j in range(N):
                np.testing.assert_allclose(system.particles[j].net_force, [0,0,0], rtol=1e-5)

    def test_lj_graph(self):
        hoomd.context.initialize()
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)

        save_loc = '/tmp/test-lj-potential-model'

        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, r_cut=rcut, debug_mode=False)
        hoomd.run(1)
        log = hoomd.analyze.log(filename=None, quantities=['potential_energy', 'pressure'], period=1)
        thermo_scalars = []
        for i in range(5):
            hoomd.run(3)
            snapshot = system.take_snapshot()
            thermo_scalars.append([log.query('potential_energy'), log.query('pressure')])

        #now run with stock lj
        hoomd.context.initialize()
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)
        lj = hoomd.md.pair.lj(r_cut=5.0, nlist=nlist)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        tfcompute = hoomd.tensorflow_plugin.tensorflow(save_loc, nlist, r_cut=rcut, debug_mode=False, force_mode='ignore')

        hoomd.run(1)
        log = hoomd.analyze.log(filename=None, quantities=['potential_energy', 'pressure'], period=1)
        for i in range(5):
            hoomd.run(3)
            snapshot = system.take_snapshot()
            v = snapshot.particles.velocity
            lj_virial = np.array([lj.forces[i].virial for i in range(N)])
            tf_virial = tfcompute.get_virial_array()[:,(0,1,2,4,5,8)]
            np.testing.assert_allclose(lj_virial, tf_virial, rtol=1e-2)
            np.testing.assert_allclose([log.query('potential_energy'), log.query('pressure')], thermo_scalars[i], rtol=1e-3)
if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])
