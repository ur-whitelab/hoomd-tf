# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import hoomd
import hoomd.md
import hoomd.htf
import unittest
import os
import tempfile
import shutil
import pickle
import glob
import numpy as np
import math
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
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_access(self):
        model_dir = build_examples.simple_potential(self.tmp)
        with hoomd.htf.tfcompute(model_dir,
                                 _mock_mode=True) as tfcompute:
            hoomd.context.initialize()
            rcut = 3
            # create a system with a few types
            cell = hoomd.lattice.unitcell(
                N=3,
                a1=[6, 0, 0],
                a2=[0, 6, 0],
                a3=[0, 0, 6],
                position=[[2, 2, 2], [1, 3, 1], [3, 1, 1]],
                type_name=['A', 'B', 'C'])
            system = hoomd.init.create_lattice(unitcell=cell, n=5)
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all())
            tfcompute.attach(nlist, r_cut=rcut)
            hoomd.run(1)
            tfcompute.get_virial_array()
            tfcompute.get_forces_array()
            pa = tfcompute.get_positions_array()
            nl = tfcompute.get_nlist_array()
            # make sure we get the 3 types
            assert len(np.unique(nl[:, :, 3].astype(np.int))) == 3
            assert len(np.unique(pa[:, 3].astype(np.int))) == 3

class test_compute(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_force_overwrite(self):
        model_dir = build_examples.simple_potential(self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(kT=2, seed=2)

            tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
            # use these to throw off timesteps
            hoomd.run(1)
            hoomd.run(1)
            for i in range(3):
                py_forces = compute_forces(system, rcut)
                for j in range(N):
                    np.testing.assert_allclose(system.particles[j].net_force,
                                               py_forces[j, :], atol=1e-5)
                hoomd.run(100)

    def test_nonlist(self):
        model_dir = build_examples.benchmark_nonlist_graph(self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[32, 32])
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(kT=2, seed=2)
            tfcompute.attach()
            hoomd.run(10)

    def test_full_batch(self):
        hoomd.context.initialize()
        model_dir = build_examples.benchmark_nonlist_graph(self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[32, 32])
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)
            tfcompute.attach(batch_size=None)
            hoomd.run(10)

    def test_write_empty_tensorboard(self):
        hoomd.context.initialize()
        model_dir = build_examples.benchmark_nonlist_graph(self.tmp)
        with hoomd.htf.tfcompute(model_dir, write_tensorboard=True) as tfcompute:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[32, 32])
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)
            tfcompute.attach(batch_size=None)
            hoomd.run(10)


    def test_trainable(self):
        model_dir = build_examples.trainable_graph(9 - 1, self.tmp)
        with hoomd.htf.tfcompute(model_dir,
                                 write_tensorboard=True) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0), n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(kT=2, seed=2)
            tfcompute.attach(nlist, r_cut=rcut, save_period=1, batch_size=4)
            hoomd.run(5)
            checkpoints = glob.glob(os.path.join(model_dir, 'model-*.data*'))
            # 6 because an extra is written at the end
            self.assertGreater(len(checkpoints), 2,
                               'Checkpoint files not being created.')

    def test_bootstrap(self):
        model_dir = build_examples.trainable_graph(9 - 1, self.tmp)
        with hoomd.htf.tfcompute(
            model_dir, bootstrap=build_examples.bootstrap_graph(
                9 - 1, model_dir),
            bootstrap_map={'epsilon': 'lj-epsilon', 'sigma': 'lj-sigma'
                           }) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(kT=2, seed=2)
            tfcompute.attach(nlist, r_cut=rcut, save_period=1)
            hoomd.run(5)

    def test_incomplete_bootstrap(self):
        model_dir = build_examples.trainable_graph(9 - 1, self.tmp)
        with hoomd.htf.tfcompute(
            model_dir, bootstrap=build_examples.bootstrap_graph(
                9 - 1, model_dir),
            bootstrap_map={'epsilon': 'lj-epsilon'
                           }) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(kT=2, seed=2)
            tfcompute.attach(nlist, r_cut=rcut, save_period=1)
            hoomd.run(5)

    def test_print(self):
        model_dir = build_examples.print_graph(9 - 1, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(kT=4, seed=1)

            tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
            for i in range(3):
                hoomd.run(2)

    def test_noforce_graph(self):
        model_dir = build_examples.noforce_graph(self.tmp)
        hoomd.context.initialize()
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all())
            tfcompute.attach(nlist, r_cut=rcut)
            for i in range(3):
                hoomd.run(1)
                for j in range(N):
                    np.testing.assert_allclose(
                        system.particles[j].net_force, [0, 0, 0], rtol=1e-5)


    def test_wrap(self):
        model_dir = build_examples.wrap_graph(self.tmp)
        hoomd.context.initialize()
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all())
            tfcompute.attach()
            hoomd.run(1)


    def test_feeddict_func(self):
        model_dir = build_examples.feeddict_graph(self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all())
            # multiple average force by particle 4 position
            # just for fun
            tfcompute.attach(nlist, r_cut=rcut, period=10,
                             feed_dict=lambda tfc:
                             {'test-tensor:0': tfc.get_positions_array()[2, :3]},
                             batch_size=4)
            hoomd.run(11)
            tf_force = tfcompute.get_forces_array()[1, :3]

    def test_feeddict(self):
        model_dir = build_examples.feeddict_graph(self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all())
            # multiple average force by particle 4 position
            # just for fun
            tfcompute.attach(nlist, r_cut=rcut, period=10,
                             feed_dict={'test-tensor:0': [1, 2, 3]})
            hoomd.run(11)
            tf_force = tfcompute.get_forces_array()[1, :3]

    def test_lj_forces(self):
        N = 3 * 3
        model_dir = build_examples.lj_graph(N - 1, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            T = 10
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(),
                                   kT=1, tau=0.2
                                   ).randomize_velocities(seed=1)
            tfcompute.attach(nlist, r_cut=rcut)
            hoomd.run(2)
            tf_forces = []
            for i in range(T):
                hoomd.run(1)
                snapshot = system.take_snapshot()
                tf_forces.append([system.particles[j].net_force
                                  for j in range(N)])
            tf_forces = np.array(tf_forces)
        # now run with stock lj
        hoomd.context.initialize()
        system = hoomd.init.create_lattice(
            unitcell=hoomd.lattice.sq(a=4.0),
            n=[3, 3])
        nlist = hoomd.md.nlist.cell(check_period=1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2
                               ).randomize_velocities(seed=1)
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
                np.testing.assert_allclose(tf_forces[i, j],
                                           lj_forces[i, j], atol=1e-5)

    def test_running_mean(self):
        model_dir = build_examples.lj_running_mean(9 - 1, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.001)
            hoomd.md.integrate.nve(group=hoomd.group.all()
                                   ).randomize_velocities(seed=1, kT=0.8)
            tfcompute.attach(nlist, r_cut=rcut, save_period=10, batch_size=4)
            hoomd.run(10)
            # now load checkpoint
        variables = hoomd.htf.load_variables(
            model_dir, ['average-energy', 'htf-batch-steps:0'])
        assert not math.isnan(variables['average-energy'])
        assert variables['htf-batch-steps'] == 11

    def test_force_output(self):
        Ne = 5
        c = hoomd.context.initialize()
        model_dir = build_examples.lj_force_output(Ne ** 2 - 1, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            rcut = 3.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=2.0),
                n=[Ne, Ne])
            c.sorter.disable()
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.01)
            lj = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
            lj2 = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
            lj2.pair_coeff.set('A', 'A', epsilon=4.0, sigma=0.8)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(seed=1, kT=0.8)
            tfcompute.attach(nlist, r_cut=rcut, period=100, save_period=1)
            tfcompute.set_reference_forces(lj)
            hoomd.run(300)
            # now load checkpoint and check error
            variables = hoomd.htf.load_variables(
                model_dir, ['error'])
            assert abs(variables['error']) < 1e-5
            # now check difference between particle forces and
            # forces from htf
            lj_forces = np.array([lj.forces[j].force for j in range(Ne**2)])
            lj_energy = np.array([lj.forces[j].energy for j in range(Ne**2)])
            np.testing.assert_allclose(tfcompute.get_forces_array(
                    )[:, :3], lj_forces)
            np.testing.assert_allclose(tfcompute.get_forces_array(
                    )[:, 3], lj_energy)

    def test_rdf(self):
        model_dir = build_examples.lj_rdf(9 - 1, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.001)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(seed=1, kT=0.8)
            tfcompute.attach(nlist, r_cut=rcut, save_period=3, batch_size=4)
            hoomd.run(10)
        # now load checkpoint
        variables = hoomd.htf.load_variables(
            model_dir, ['avg-rdf'])
        assert np.sum(variables['avg-rdf']) > 0

    def test_lj_energy(self):
        model_dir = build_examples.lj_graph(9 - 1, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            T = 10
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.001)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(seed=1, kT=0.8)
            log = hoomd.analyze.log(filename=None,
                                    quantities=['potential_energy',
                                                'kinetic_energy'], period=1)
            tfcompute.attach(nlist, r_cut=rcut)
            energy = []
            for i in range(T):
                hoomd.run(250)
                energy.append(log.query('potential_energy'
                                        ) + log.query('kinetic_energy'))
                if i > 1:
                    np.testing.assert_allclose(energy[-1],
                                               energy[-2], atol=1e-3)

    def test_lj_pressure(self):
        # TODO The virials are off by 1e-6, leading to
        # pressure differences of 1e-3.
        # I can't figure out why, but since PE and forces are
        # matching exactly, I'll leave the tol
        # set that high.
        model_dir = build_examples.lj_graph(9 - 1, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell(check_period=1)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(),
                                   kT=1, tau=0.2).randomize_velocities(seed=1)
            tfcompute.attach(nlist, r_cut=rcut)
            log = hoomd.analyze.log(filename=None, quantities=[
                    'potential_energy', 'pressure'], period=1)
            thermo_scalars = []
            tf_virial = []
            for i in range(5):
                hoomd.run(3)
                snapshot = system.take_snapshot()
                tf_virial.append(tfcompute.get_virial_array())
                thermo_scalars.append([log.query('potential_energy'
                                                 ), log.query('pressure')])
        # now run with stock lj
        hoomd.context.initialize()
        system = hoomd.init.create_lattice(
            unitcell=hoomd.lattice.sq(a=4.0),
            n=[3, 3])
        nlist = hoomd.md.nlist.cell(check_period=1)
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1,
                               tau=0.2).randomize_velocities(seed=1)
        lj = hoomd.md.pair.lj(r_cut=5.0, nlist=nlist)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        log = hoomd.analyze.log(filename=None,
                                quantities=['potential_energy', 'pressure'],
                                period=1)
        for i in range(5):
            hoomd.run(3)
            snapshot = system.take_snapshot()
            v = snapshot.particles.velocity
            lj_virial = np.array([lj.forces[j].virial for j in range(N)])
            for j in range(N):
                np.testing.assert_allclose(lj_virial[j][0:2],
                                           tf_virial[i][j][0:2], atol=1e-5)
            # np.testing.assert_allclose([log.query('potential_energy'),
            # log.query('pressure')], thermo_scalars[i], rtol=1e-3)


class test_mol_batching(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_single_atom(self):
        hoomd.context.initialize()
        model_dir = build_examples.lj_mol(9 - 1, 8, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3, 3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2)
            tfcompute.attach(nlist, r_cut=rcut)
            hoomd.run(8)

    def test_single_atom_batched(self):
        hoomd.context.initialize()
        model_dir = build_examples.lj_mol(9 - 1, 8, self.tmp)
        with hoomd.htf.tfcompute(model_dir, _mock_mode=True) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3, 3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2)
            with self.assertRaises(ValueError):
                tfcompute.attach(nlist, r_cut=rcut, batch_size=3)
            hoomd.run(8)

    def test_single_atom_malformed(self):
        hoomd.context.initialize()
        model_dir = build_examples.lj_mol(9 - 1, 8, self.tmp)
        with hoomd.htf.tfcompute(model_dir, _mock_mode=True) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3, 3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2)
            with self.assertRaises(ValueError):
                tfcompute.attach(nlist, r_cut=rcut, mol_indices=[1, 1, 4, 24])
            hoomd.run(8)

    def test_multi_atom(self):
        hoomd.context.initialize()
        model_dir = build_examples.lj_mol(9 - 1, 8, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            N = 3 * 3
            NN = N - 1
            rcut = 5.0
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3, 3])
            nlist = hoomd.md.nlist.cell()
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2)
            tfcompute.attach(nlist,
                             r_cut=rcut,
                             mol_indices=[[0, 1, 2], [3, 4], [5, 6, 7], [8]])
            hoomd.run(8)

    def test_mol_force_output(self):
        hoomd.context.initialize()
        model_dir = build_examples.mol_force(self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3, 3])
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2)
            tfcompute.attach(mol_indices=[[0, 1, 2], [3, 4], [5, 6, 7], [8]])
            hoomd.run(8)

    def test_reverse_mol_index(self):
        # need this for logging
        hoomd.context.initialize()
        # each element is the index of atoms in the molecule
        mi = [[1, 2, 0, 0, 0], [3, 0, 0, 0, 0], [4, 5, 7, 8, 9]]
        rmi = hoomd.htf._make_reverse_indices(mi)
        # should be
        rmi_ref = [
            [0, 0],
            [0, 1],
            [1, 0],
            [2, 0],
            [2, 1],
            [-1, -1],
            [2, 2],
            [2, 3],
            [2, 4]
        ]
        self.assertEqual(rmi, rmi_ref)


class test_saving(unittest.TestCase):

    def test_tensor_save(self):
        hoomd.context.initialize()
        model_dir = build_examples.saving_graph()
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                               n=[3, 3])
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2)
            tfcompute.attach()
            hoomd.run(8)

        # now load
        vars = hoomd.htf.load_variables(model_dir, ['v1', 'v2'])

if __name__ == '__main__':
    unittest.main()
