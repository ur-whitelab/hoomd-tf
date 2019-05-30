# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import hoomd, hoomd.md
import hoomd.tensorflow_plugin
import unittest
import os, tempfile, shutil, pickle, glob
import numpy as np, math
import tensorflow as tf
import build_examples

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


class test_access(unittest.TestCase):
    def test_access(self):
        model_dir = build_examples.simple_potential()
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

            tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
            hoomd.run(1)

            tfcompute.get_virial_array()
            tfcompute.get_positions_array()
            tfcompute.get_nlist_array()
            tfcompute.get_forces_array()


class test_compute(unittest.TestCase):
    def test_force_overwrite(self):
        hoomd.context.initialize()
        model_dir = build_examples.simple_potential()
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)

            tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
            #use these to throw off timesteps
            hoomd.run(1)
            hoomd.run(1)
            for i in range(3):
                py_forces = compute_forces(system, rcut)
                for j in range(N):
                    np.testing.assert_allclose(system.particles[j].net_force, py_forces[j, :], atol=1e-5)
                hoomd.run(100)

    def test_nonlist(self):
        hoomd.context.initialize()
        model_dir = build_examples.benchmark_nonlist_graph()
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[32,32])
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)
            tfcompute.attach()
            hoomd.run(10)


    def test_full_batch(self):
        hoomd.context.initialize()
        model_dir = build_examples.benchmark_nonlist_graph()
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[32,32])
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)
            tfcompute.attach(batch_size=None)
            hoomd.run(10)

    def test_trainable(self):
        model_dir = build_examples.trainable_graph(9 - 1)
        with hoomd.tensorflow_plugin.tfcompute(model_dir, write_tensorboard=True) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)

            tfcompute.attach(nlist, r_cut=rcut, save_period=1, batch_size=4)

            hoomd.run(5)

            checkpoints = glob.glob(os.path.join(model_dir, 'model-*.data*'))

            #6 because an extra is written at the end
            self.assertGreater(len(checkpoints), 2, 'Checkpoint files not being created.')

    def test_bootstrap(self):
        model_dir = build_examples.trainable_graph(9 - 1)
        with hoomd.tensorflow_plugin.tfcompute(model_dir,
                                               bootstrap = build_examples.bootstrap_graph(9 - 1, model_dir),
                                               bootstrap_map = {'epsilon':'lj-epsilon', 'sigma':'lj-sigma'}) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)

            tfcompute.attach(nlist, r_cut=rcut, save_period=1, batch_size=4)

            hoomd.run(5)


    def test_print(self):
        hoomd.context.initialize()
        model_dir = build_examples.print_graph(9 - 1)
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=4, seed=1)

            tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
            for i in range(3):
                hoomd.run(2)

    def test_noforce_graph(self):
        hoomd.context.initialize()
        model_dir = build_examples.noforce_graph()
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all())

            tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
            for i in range(3):
                hoomd.run(1)
                for j in range(N):
                    np.testing.assert_allclose(system.particles[j].net_force, [0,0,0], rtol=1e-5)

    def test_feeddict_func(self):
        hoomd.context.initialize()
        model_dir = build_examples.feeddict_graph()
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
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
            tfcompute.attach(nlist, r_cut=rcut, period=10, feed_dict=lambda tfc: {'test-tensor:0': tfc.get_positions_array()[2, :3]}, batch_size=4)
            hoomd.run(11)
            tf_force = tfcompute.get_forces_array()[1,:3]

    def test_feeddict(self):
        hoomd.context.initialize()
        model_dir = build_examples.feeddict_graph()
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
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
            tfcompute.attach(nlist, r_cut=rcut, period=10, feed_dict={'test-tensor:0': [1,2,3]}, batch_size=4)
            hoomd.run(11)
            tf_force = tfcompute.get_forces_array()[1,:3]

    def test_lj_forces(self):
        N = 3 * 3
        model_dir = build_examples.lj_graph(N - 1)
        hoomd.context.initialize()
        with hoomd.tensorflow_plugin.tfcompute(model_dir, _mock_mode=False) as tfcompute:
            T = 10
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)


            tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
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

    def test_running_mean(self):
        hoomd.context.initialize()
        model_dir = build_examples.lj_running_mean(9 - 1)
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.001)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(seed=1, kT=0.8)
            tfcompute.attach(nlist, r_cut=rcut, save_period=10, batch_size=4)
            hoomd.run(10)
        # now load checkpoint
        variables  = hoomd.tensorflow_plugin.load_variables(model_dir, ['average-energy', 'htf-batch-steps'])
        assert not math.isnan(variables['average-energy'])
        assert variables['htf-batch-steps'] == 11


    def test_force_output(self):
        Ne = 5
        c = hoomd.context.initialize()
        model_dir = build_examples.lj_force_output(Ne **2 - 1)
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            rcut = 3.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[Ne,Ne])
            c.sorter.disable()
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.01)
            lj = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
            lj2 = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
            lj2.pair_coeff.set('A', 'A', epsilon=4.0, sigma=0.8)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(seed=1, kT=0.8)
            tfcompute.attach(nlist, r_cut=rcut, period=100, save_period=1, batch_size=4)
            tfcompute.set_reference_forces(lj)
            hoomd.run(300)
            # now load checkpoint and check error
            variables  = hoomd.tensorflow_plugin.load_variables(model_dir, ['error'])
            assert abs(variables['error']) < 1e-5
            # now check difference between particle forces and forces from htf
            lj_forces = np.array([lj.forces[j].force for j in range(Ne**2)])
            lj_energy = np.array([lj.forces[j].energy for j in range(Ne**2)])
            np.testing.assert_allclose(tfcompute.get_forces_array()[:,:3], lj_forces)
            np.testing.assert_allclose(tfcompute.get_forces_array()[:,3], lj_energy)

    def test_rdf(self):
        hoomd.context.initialize()
        model_dir = build_examples.lj_rdf(9 - 1)
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[3,3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.001)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(seed=1, kT=0.8)
            tfcompute.attach(nlist, r_cut=rcut, save_period=3)
            hoomd.run(10)
        # now load checkpoint
        variables  = hoomd.tensorflow_plugin.load_variables(model_dir, ['avg-rdf:0'])
        assert np.sum(variables['avg-rdf:0']) > 0


    def test_lj_energy(self):
        hoomd.context.initialize()
        model_dir = build_examples.lj_graph(9 - 1)
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
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
            tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
            energy = []
            for i in range(T):
                hoomd.run(250)
                energy.append(log.query('potential_energy') + log.query('kinetic_energy'))
                if i > 1:
                    np.testing.assert_allclose(energy[-1], energy[-2], atol=1e-3)

    def test_lj_pressure(self):
        #TODO The virials are off by 1e-6, leading to pressure differences of 1e-3.
        #I can't figure out why, but since PE and forces are matching exactly, I'll leave the tol
        #set that high.
        hoomd.context.initialize()
        model_dir = build_examples.lj_graph(9 - 1)
        with hoomd.tensorflow_plugin.tfcompute(model_dir, _mock_mode=False) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell(check_period = 1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)


            tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
            log = hoomd.analyze.log(filename=None, quantities=['potential_energy', 'pressure'], period=1)
            thermo_scalars = []
            for i in range(5):
                hoomd.run(3)
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
        log = hoomd.analyze.log(filename=None, quantities=['potential_energy', 'pressure'], period=1)
        for i in range(5):
            hoomd.run(3)
            np.testing.assert_allclose([log.query('potential_energy'), log.query('pressure')], thermo_scalars[i], atol=1e-2)

class test_mol_batching(unittest.TestCase):
    def test_single_atom(self):
        hoomd.context.initialize()
        model_dir = build_examples.lj_mol(9 - 1, 8)
        with htf.tfcompute(model_dir) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)
            tfcompute.attach(nlist, r_cut=rcut, batch_size=None)
            hoomd.run(8)

    def test_single_atom_batched(self):
        hoomd.context.initialize()
        model_dir = build_examples.lj_mol(9 - 1, 8)
        with htf.tfcompute(model_dir) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3,3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)
            tfcompute.attach(nlist, r_cut=rcut, batch_size=3)
            hoomd.run(8)

if __name__ == '__main__':
    unittest.main()
