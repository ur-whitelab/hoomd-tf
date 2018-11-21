# -*- coding: iso-8859-1 -*-
# Maintainer: Andrew White

import hoomd, hoomd.md
import hoomd.tensorflow_plugin
import unittest
import os, tempfile, shutil, pickle, glob
import numpy as np, math
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
    def test_ipc_array_comm(self):
        hoomd.context.initialize('--mode=cpu')
        shared = np.zeros(10)
        ipc = hoomd.tensorflow_plugin.ipc_array_comm(shared,  hoomd.context.exec_conf)

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
        with hoomd.tensorflow_plugin.tfcompute(model_dir, _mock_mode=True) as tfcompute:
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


class test_compute(unittest.TestCase):
    def test_force_overwrite(self):
        model_dir = '/tmp/test-simple-potential-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)

            tfcompute.attach(nlist, r_cut=rcut)
            #use these to throw off timesteps
            hoomd.run(1)
            hoomd.run(1)
            for i in range(3):
                py_forces = compute_forces(system, rcut)
                for j in range(N):
                    np.testing.assert_allclose(system.particles[j].net_force, py_forces[j, :], atol=1e-5)
                hoomd.run(100)

    def test_throws_force_mode(self):
        model_dir ='/tmp/benchmark-nonlist-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[32,32])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)

            self.assertRaises(ValueError, tfcompute.attach(nlist, r_cut=rcut))

    def test_nonlist(self):
        model_dir ='/tmp/benchmark-nonlist-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[32,32])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)

            tfcompute.attach(nlist, r_cut=rcut)
            hoomd.run(10)


    def test_trainable(self):
        model_dir ='/tmp/test-trainable-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir, write_tensorboard=True) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)

            tfcompute.attach(nlist, r_cut=rcut, save_period=1)

            hoomd.run(5)

            checkpoints = glob.glob(os.path.join(model_dir, 'model-*.data*'))

            #6 because an extra is written at the end
            self.assertEqual(len(checkpoints), 6, 'Checkpoint files not being created.')

    def test_bootstrap(self):
        model_dir ='/tmp/test-trainable-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir,
            bootstrap=os.path.join(model_dir, 'bootstrap'),
            bootstrap_map={'epsilon':'lj-epsilon', 'sigma':'lj-sigma'}) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)

            tfcompute.attach(nlist, r_cut=rcut, save_period=1)

            hoomd.run(5)


    def test_print(self):
        model_dir = '/tmp/test-print-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=4, seed=1)

            tfcompute.attach(nlist, r_cut=rcut)
            for i in range(3):
                hoomd.run(2)

    def test_force_ignore(self):
        model_dir = '/tmp/test-simple-potential-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=4, seed=1)

            tfcompute.attach(nlist, r_cut=rcut, force_mode='ignore')
            for i in range(3):
                hoomd.run(100)
                for j in range(N):
                    np.testing.assert_allclose(system.particles[j].net_force, [0,0,0], rtol=1e-5)

    def test_noforce_graph(self):
        model_dir = '/tmp/test-noforce-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all())

            tfcompute.attach(nlist, r_cut=rcut, force_mode='hoomd2tf')
            for i in range(3):
                hoomd.run(1)
                for j in range(N):
                    np.testing.assert_allclose(system.particles[j].net_force, [0,0,0], rtol=1e-5)

    def test_output_graph(self):
        #TODO: Rewrite this to work
        model_dir = '/tmp/test-noforce-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            lj = hoomd.md.pair.lj(r_cut=5.0, nlist=nlist)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=4, seed=1)

            hoomd.run(100)

            tfcompute.attach(nlist, r_cut=rcut)


            #make sure positions are offset by one (so net force matches pos)
            last_force = np.copy(system.particles[1].net_force[:3])
            hoomd.run(0)
            tf_force = tfcompute.get_forces_array()[1,:3]
            np.testing.assert_allclose(last_force, tf_force, rtol=1e-5)

    def test_feeddict(self):
        model_dir = '/tmp/test-feeddict-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all())

            #multiple average force by particle 4 position
            #just for fun
            tfcompute.attach(nlist, r_cut=rcut, period=10, force_mode='hoomd2tf', feed_func=lambda tfc: {'test-tensor:0': tfc.get_positions_array()[4, :3]})
            hoomd.run(11)
            #tf_force = tfcompute.get_forces_array()[1,:3]
            print(tf_force)

    def test_lj_forces(self):
        model_dir = '/tmp/test-lj-potential-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            T = 100
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)


            tfcompute.attach(nlist, r_cut=rcut)
            hoomd.run(2)
            tf_forces = []
            for i in range(T):
                hoomd.run(1)
                snapshot = system.take_snapshot()
                tf_forces.append([system.particles[j].net_force for j in range(N)])

            tf_forces = np.array(tf_forces)
        #now run with stock lj
        hoomd.context.initialize()
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
        nlist = hoomd.md.nlist.cell(check_period = 1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)
        lj = hoomd.md.pair.lj(r_cut=5.0, nlist=nlist)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

        hoomd.run(2)
        lj_forces = []
        for i in range(T):
            hoomd.run(1)
            snapshot = system.take_snapshot()
            lj_forces.append([system.particles[j].net_force for j in range(N)])
        lj_forces = np.array(lj_forces)
        for i in range(T):
            for j in range(N):
                np.testing.assert_allclose(tf_forces[i,j], lj_forces[i,j], atol=1e-5)

    def test_lj_energy(self):
        model_dir = '/tmp/test-lj-potential-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            T = 10
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.001)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(seed=1, kT=0.8)
            log = hoomd.analyze.log(filename=None, quantities=['potential_energy', 'kinetic_energy'], period=1)
            tfcompute.attach(nlist, r_cut=rcut)
            energy = []
            for i in range(T):
                hoomd.run(250)
                energy.append(log.query('potential_energy') + log.query('kinetic_energy'))
                if i > 1:
                    np.testing.assert_allclose(energy[-1], energy[-2], atol=1e-3)

    def test_lj_pressure(self):
        model_dir = '/tmp/test-lj-potential-model'
        with hoomd.tensorflow_plugin.tfcompute(model_dir, _mock_mode=False) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)


            tfcompute.attach(nlist, r_cut=rcut)
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
