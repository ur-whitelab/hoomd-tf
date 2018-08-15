# -*- coding: iso-8859-1 -*-
# Maintainer: Andrew White

import hoomd, hoomd.md
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
    def test_ipc_array_comm(self):
        hoomd.context.initialize('--mode=cpu')
        shared = np.zeros(10)
        ipc = hoomd.tensorflow_plugin.IPCArrayComm(shared,  hoomd.context.exec_conf)

        np.testing.assert_allclose(shared, ipc.getArray())

        shared[4] = 10.0
        ipc.receive()
        np.testing.assert_allclose(shared, ipc.getArray())

        ref = shared[:]
        shared[:] = -1
        ipc.send()
        np.testing.assert_allclose(shared, ref)

class test_access(unittest.TestCase):
    def test_access(self):
        model_dir = '/tmp/test-simple-potential-model'
        tfcompute = hoomd.tensorflow_plugin.tensorflow(model_dir, _mock_mode=True)
        hoomd.context.initialize('--gpu_error_checking')
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all())

        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(1)

        tfcompute.get_virial_array()
        tfcompute.get_positions_array()                
        tfcompute.get_nlist_array()
        tfcompute.get_forces_array()

        print('Able to access all arrays!')

class test_compute(unittest.TestCase):
    def test_compute_force_overwrite(self):
        model_dir = '/tmp/test-simple-potential-model'
        tfcompute = hoomd.tensorflow_plugin.tensorflow(model_dir)
        hoomd.context.initialize()
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all())



        tfcompute.attach(nlist, r_cut=rcut)
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
        tf_virial = []
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

        hoomd.run(1)
        log = hoomd.analyze.log(filename=None, quantities=['potential_energy', 'pressure'], period=1)
        for i in range(5):
            hoomd.run(3)
            snapshot = system.take_snapshot()
            v = snapshot.particles.velocity
            lj_virial = np.array([lj.forces[j].virial for j in range(N)])
            np.testing.assert_allclose([log.query('potential_energy'), log.query('pressure')], thermo_scalars[i], rtol=1e-2)

if __name__ == '__main__':
    unittest.main()
