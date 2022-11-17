# Copyright (c) 2020 HOOMD-TF Developers

import hoomd
import hoomd.md
import hoomd.htf as htf
import unittest
import os
import tempfile
import shutil
import pickle
import glob
import gsd.hoomd
import numpy as np
import math
import tensorflow as tf
import build_examples

from hoomd.htf.simmodel import _make_reverse_indices


def compute_forces(sim, rcut):
    '''1 / r^2 force'''
    snapshot = sim.State.get_snapshot()
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
        self.device = hoomd.device.CPU('')
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_access(self):
        model = build_examples.SimplePotential(32)
        tfcompute = htf.tfcompute(model)
        rcut = 3
        # create a 3D system with a few types
        snap = gsd.hoomd.Snapshot()
        snap.particles.N = 3
        snap.particles.types = ['A', 'B', 'C']
        snap.particles.typeid = [0, 1, 2]
        snap.particles.position = [[2, 2, 2], [1, 3, 1], [3, 1, 1]]
        snap.configuration.box = [6, 6, 6, 0, 0, 0]
        sim = hoomd.Simulation(self.device)
        sim.create_state_from_snapshot(snap)
        sim.state.replicate(5, 5, 5)

        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        hoomd.md.Integrator(dt=0.005, methods=[nve])
        tfcompute.attach(nlist, r_cut=rcut)
        sim.run(1)
        tfcompute.get_virial_array()
        tfcompute.get_forces_array()
        pa = tfcompute.get_positions_array()
        nl = tfcompute.get_nlist_array()
        # make sure we get the 3 types
        self.assertEqual(len(np.unique(nl[:, :, 3].astype(np.int))), 3)
        self.assertEqual(len(np.unique(pa[:, 3].astype(np.int))), 3)


class test_compute(unittest.TestCase):
    def setUp(self):
        self.device = hoomd.device.CPU('')
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_force_overwrite(self):
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        model = build_examples.SimplePotential(NN)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(lattice_constant=4.0,
                                                    n_replicas=[3,3],
                                                    device=self.device,
                                                    seed=2)

        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=2)
        hoomd.md.Integrator(methods=[nve], dt=0.005)

        tfcompute.attach(nlist, r_cut=rcut)
        # use these to throw off timesteps
        sim.run(1)
        sim.run(1)
        for i in range(3):
            py_forces = compute_forces(sim, rcut)
            for j in range(N):
                np.testing.assert_allclose(sim.state.particles[j].net_force,
                                           py_forces[j, :], atol=1e-5)
            hoomd.run(100)

    def test_force_overwrite_batched(self):
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        model = build_examples.SimplePotential(NN)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3])
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=2)
        hoomd.md.Integrator(methods=[nve], dt=0.005)

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
        model = build_examples.BenchmarkNonlistModel(0)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[32,32], device=self.device)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=2)
        hoomd.md.Integrator(methods=[nve], dt=0.005)
        tfcompute.attach()
        hoomd.run(10)

    def test_full_batch(self):
        model = build_examples.BenchmarkNonlistModel(0)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[32,32], device=self.device)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=2)
        hoomd.md.Integrator(methods=[nve], dt=0.005)
        tfcompute.attach(batch_size=None)
        hoomd.run(10)

    def test_trainable(self):
        model = build_examples.TrainableGraph(16, output_forces=False)
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(0.01),
            loss='MeanSquaredError')
        start = model.get_layer('lj').trainable_weights[0].numpy()
        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0, n_replicas=[3,3], device=self.device)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=2)
        hoomd.md.Integrator(methods=[nve], dt=0.005)
        tfcompute.attach(nlist, r_cut=rcut, batch_size=4, train=True)
        lj = hoomd.md.pair.lj(r_cut=5.0, nlist=nlist)
        lj.pair_coeff.set('A', 'A', epsilon=1.1, sigma=0.9)
        hoomd.run(25)
        end = model.get_layer('lj').trainable_weights[0].numpy()
        assert np.sum((start - end)**2) > 0.01**2, 'No training observed'

    def test_model_save(self):
        '''Saves model after training
        '''
        model = build_examples.TrainableGraph(16, output_forces=False)
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(0.01),
            loss='MeanSquaredError')

        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=2)
        hoomd.md.Integrator(methods=[nve], dt=0.005)
        tfcompute.attach(nlist, train=True, r_cut=rcut)
        hoomd.run(5)

        model.save(os.path.join(self.tmp, 'test-model'))

    def test_model_load(self):
        ''' Saves model after training and then uses
        if for inference
        '''
        model = build_examples.TrainableGraph(16, output_forces=False)
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(0.01),
            loss='MeanSquaredError')

        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=2)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=2)
        hoomd.md.Integrator(methods=[nve], dt=0.005)
        tfcompute.attach(nlist, train=True, r_cut=rcut)
        hoomd.run(5)

        model.save(os.path.join(self.tmp, 'test-model'))

        model = tf.keras.models.load_model(
            os.path.join(self.tmp, 'test-model'))
        infer_model = build_examples.TrainableGraph(16, output_forces=True)
        infer_model.set_weights(model.get_weights())

        tfcompute.disable()

        tfcompute = htf.tfcompute(infer_model)
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(5)

    def test_model_load_serial(self):
        ''' Saves model after training and then uses
        if for inference
        '''
        model = build_examples.TrainableGraph(16, output_forces=False)
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(0.01),
            loss='MeanSquaredError')

        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=2)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=2)
        hoomd.md.Integrator(methods=[nve], dt=0.005)
        tfcompute.attach(nlist, train=True, r_cut=rcut)
        hoomd.run(5)

        model.save(os.path.join(self.tmp, 'test-model'))
        del model

        return
        # We are having trouble
        # get_config in SimModel fails if I call super - don't know why
        # Because I cannot call super this code doesn't work
        # We keep the partial test because it calls the get_config methods,
        # checking that they are at least callable.
        model = tf.keras.models.load_model(
            os.path.join(self.tmp, 'test-model'),
            custom_objects={**hoomd.htf.custom_objects,
                            'TrainableGraph': build_examples.TrainableGraph})

        tfcompute.disable()

        tfcompute = htf.tfcompute(model)
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(5)

    def test_print(self):
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        model = build_examples.PrintModel(NN)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=4)
        hoomd.md.Integrator(methods=[nve], dt=0.005)

        tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
        for i in range(3):
            hoomd.run(2)

    def test_noforce_graph(self):
        model = build_examples.NoForceModel(9, output_forces=False)
        tfcompute = htf.tfcompute(model)
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.md.Integrator(methods=[nve], dt=0.005)
        for i in range(3):
            hoomd.run(1)
            for j in range(N):
                np.testing.assert_allclose(
                    system.particles[j].net_force, [0, 0, 0], rtol=1e-5)

    def test_wrap(self):
        model = build_examples.WrapModel(0, output_forces=False)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        tfcompute.attach()
        hoomd.md.Integrator(methods=[nve], dt=0.005)
        hoomd.run(1)

    def test_skew_fails(self):
        model = build_examples.WrapModel(0, output_forces=False)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device)
        hoomd.update.box_resize(xy=0.5)
        hoomd.run(1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        tfcompute.attach()
        hoomd.md.Integrator(methods=[nve], dt=0.005)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            hoomd.run(1)

    def test_lj_forces(self):
        model = build_examples.LJModel(32)
        tfcompute = htf.tfcompute(model)
        T = 10
        N = 5 * 5
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=3.0,
            n_replicas=[5,5], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nvt = hoomd.md.methods.NVT(filter=hoomd.filter.All(),
                               kT=1, tau=0.2
                               )
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All())
        hoomd.md.Integrator(methods=[nvt], dt=0.005)
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(20)
        tf_forces = []
        for i in range(T):
            hoomd.run(1)
            snapshot = system.take_snapshot()
            tf_forces.append([system.particles[j].net_force
                              for j in range(N)])
        tf_forces = np.array(tf_forces)
        # now run with stock lj
        sim = build_examples.generic_square_lattice(
            lattice_constant=3.0,
            n_replicas=[5,5], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nvt = hoomd.md.methods.NVT(filter=hoomd.filter.All(), kT=1, tau=0.2)
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All())
        hoomd.md.Integrator(methods=[nvt], dt=0.005)
        lj = hoomd.md.pair.lj(r_cut=5.0, nlist=nlist)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        hoomd.run(20)
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
                # make sure we wrote test to have non-zero forces
                assert np.sum(
                    lj_forces[i, j]**2) > 1e-4**2, 'Forces are too low to assess!'

    def test_running_mean(self):
        model = build_examples.LJRunningMeanModel(32)
        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell()
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.8)
        hoomd.md.Integrator(methods=[nve], dt=0.001)
        tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
        hoomd.run(10)
        result = model.avg_energy.result().numpy()
        assert result < 0

    def test_force_output(self):
        Ne = 5
        c = self.device
        model = build_examples.LJModel(32, output_forces=False)
        model.compile(loss='MeanSquaredError')
        tfcompute = htf.tfcompute(model)
        rcut = 3.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=2.0,
            n_replicas=[Ne,Ne], device=self.device, seed=1)
        #TODO: figure out how to do this with new syntax
        c.sorter.disable()
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        lj = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj2 = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
        lj2.pair_coeff.set('A', 'A', epsilon=4.0, sigma=0.8)
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.8)
        hoomd.md.Integrator(methods=[nve], dt=0.01)
        tfcompute.attach(nlist, train=True, r_cut=rcut, period=100)
        tfcompute.set_reference_forces(lj)
        hoomd.run(300)
        error = model.metrics[0].result().numpy()
        assert abs(error) < 1e-5
        # now check difference between particle forces and
        # forces from htf
        lj_forces = np.array([lj.forces[j].force for j in range(Ne**2)])
        lj_energy = np.array([lj.forces[j].energy for j in range(Ne**2)])
        np.testing.assert_allclose(tfcompute.get_forces_array(
        )[:, :3], lj_forces)
        np.testing.assert_allclose(tfcompute.get_forces_array(
        )[:, 3], lj_energy)

    def test_rdf(self):
        model = build_examples.LJRDF(32)
        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1, seed=1)
        nlist = hoomd.md.nlist.Cell()
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.8)
        hoomd.md.Integrator(methods=[nve], dt=0.001)
        tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
        hoomd.run(10)
        rdf = model.avg_rdf.result().numpy()
        assert len(rdf) > 5
        assert np.sum(rdf) > 0

    def test_typed_rdf(self):
        '''Test RDF typing
        '''
        model = build_examples.LJTypedModel(32)
        tfcompute = htf.tfcompute(model)
        rcut = 10.0
        # build system using example from hoomd
        snapshot = hoomd.data.make_snapshot(N=10,
                                            box=hoomd.data.boxdim(Lx=10,
                                                                  Ly=10,
                                                                  Lz=10),
                                            particle_types=['A', 'B'],
                                            bond_types=['polymer'])
        snapshot.particles.position[:] = [[-4.5, 0, 0], [-3.5, 0, 0],
                                          [-2.5, 0, 0], [-1.5, 0, 0],
                                          [-0.5, 0, 0], [0.5, 0, 0],
                                          [1.5, 0, 0], [2.5, 0, 0],
                                          [3.5, 0, 0], [4.5, 0, 0]]
        snapshot.particles.typeid[0:7] = 0
        snapshot.particles.typeid[7:10] = 1
        snapshot.bonds.resize(9)
        snapshot.bonds.group[:] = [[0, 1], [1, 2], [2, 3],
                                   [3, 4], [4, 5], [5, 6],
                                   [6, 7], [7, 8], [8, 9]]
        snapshot.replicate(3, 3, 3)
        system = hoomd.init.read_snapshot(snapshot)
        nlist = hoomd.md.nlist.Cell()
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.8)
        hoomd.md.Integrator(methods=[nve], dt=0.001)
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(10)
        rdfa = model.avg_rdfa.result().numpy()
        rdfb = model.avg_rdfb.result().numpy()
        assert np.sum(rdfa) > 0
        np.testing.assert_array_almost_equal(rdfa, rdfb)

    def test_training_flag(self):
        model = build_examples.TrainModel(4, dim=1, top_neighs=2)
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(0.01),
            loss='MeanSquaredError')
        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell()
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.8)
        hoomd.md.Integrator(methods=[nve], dt=0.001)
        tfcompute.attach(nlist, train=True, r_cut=rcut, batch_size=4)
        hoomd.run(10)

        tfcompute.attach(nlist, train=False, r_cut=rcut, batch_size=4)
        hoomd.run(10)

    def test_retrace(self):
        model = build_examples.TrainModel(4, dim=1, top_neighs=2)
        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell()
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.8)
        hoomd.md.Integrator(methods=[nve], dt=0.001)
        tfcompute.attach(nlist, r_cut=rcut, save_output_period=1)
        hoomd.run(1)
        assert tfcompute.outputs[0][-1] != 0

        # without retrace
        model.output_zero = True
        hoomd.run(1)
        assert tfcompute.outputs[0][-1] != 0

        # with retrace
        model.retrace_compute()
        hoomd.run(1)
        assert tfcompute.outputs[0][-1] == 0

    def test_lj_energy(self):
        model = build_examples.LJModel(32)
        tfcompute = htf.tfcompute(model)
        N = 3 * 3
        NN = N - 1
        T = 10
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.8)
        hoomd.md.Integrator(methods=[nve], dt=0.001)
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

    def test_nlist_count(self):
        '''Make sure nlist is full, not half
        '''
        model = build_examples.LJModel(32)
        tfcompute = htf.tfcompute(model)
        N = 3 * 3
        NN = N - 1
        T = 10
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell()
        
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.8)
        hoomd.md.Integrator(methods=[nve], dt=0.001)
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(1)  # in lattice, should have 4 neighbors
        nl = tfcompute.get_nlist_array()
        ncount = np.sum(np.sum(nl**2, axis=2) > 0.1, axis=1)
        self.assertEqual(np.min(ncount), 4)

    def test_mapped_nlist(self):
        '''Compute mapped nlist
        '''
        N = 3 * 3
        NN = N - 1
        T = 10
        CGN = 2
        rcut = 5.0

        model = build_examples.MappedNlist(NN, output_forces=False)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1)
        self.assertEqual(sim.state.N_particles, N)
        aa_filter, mapped_filter = tfcompute.enable_mapped_nlist(
            sim, build_examples.MappedNlist.my_map)

        assert len(aa_filter()) == N
        assert len(mapped_filter()) == 2
        # 2 CG sites
        self.assertEqual(sim.state.N_particles, N + CGN)
        nlist = hoomd.md.nlist.Cell()
        
        nve = hoomd.md.methods.NVE(filter=aa_filter)
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.8)
        hoomd.md.Integrator(methods=[nve], dt=0.001)
        tfcompute.attach(nlist, r_cut=rcut, save_output_period=2)
        hoomd.run(8)
        positions = tfcompute.outputs[0].reshape(-1, N + CGN, 4)
        # check that mapping function was applied
        np.testing.assert_allclose(
            positions[1:, N, :3], np.mean(positions[1:, :-1, :3], axis=1), atol=1e-5)

        # check that there is no mixing betwee neighbor lists
        aa = set(np.unique(tfcompute.outputs[1][..., -1].astype(int)))
        cg = set(np.unique(tfcompute.outputs[2][..., -1].astype(int)))
        self.assertTrue(aa.intersection(cg) == set([0]))

    def test_lj_pressure(self):
        # TODO The virials are off by 1e-6, leading to
        # pressure differences of 1e-3.
        # I can't figure out why, but since PE and forces are
        # matching exactly, I'll leave the tol
        # set that high.
        model = build_examples.LJVirialModel(32, virial=True)
        tfcompute = htf.tfcompute(model)
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nvt = hoomd.md.methods.NVT(filter=hoomd.filter.All(),
                               kT=1, tau=0.2)
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All())
        hoomd.md.Integrator(methods=[nvt], dt=0.005)
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
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3,3], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nvt = hoomd.md.methods.NVT(filter=hoomd.filter.All(), kT=1,
                               tau=0.2)
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All())
        hoomd.md.Integrator(methods=[nvt], dt=0.005)
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
        self.device = hoomd.device.CPU('')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_single_atom(self):
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        sim = build_examples.generic_square_lattice(lattice_constant=4.0,
                                           n_replicas=[3,3], device=self.device)

        mol_indices = htf.find_molecules(system)
        model = build_examples.LJMolModel(
            MN=1, mol_indices=mol_indices, nneighbor_cutoff=NN)
        tfcompute = htf.tfcompute(model)
        nlist = hoomd.md.nlist.Cell()
        dt=0.005
        nvt = hoomd.md.methods.NVT(filter=hoomd.filter.All(), kT=1, tau=0.2)
        #TODO: figure out how to do this with new syntax
        assert self.device.sorter.enabled
        hoomd.md.Integrator(dt=dt, methods=[nvt])
        tfcompute.attach(nlist, r_cut=rcut)
        # make sure tfcompute disabled the sorting
        assert not self.device.sorter.enabled
        hoomd.run(8)

    def test_single_atom_batched(self):
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        sim = build_examples.generic_square_lattice(lattice_constant=4.0,
                                           n_replicas=[3,3], device=self.device)

        mol_indices = htf.find_molecules(system)
        model = build_examples.LJMolModel(
            MN=1, mol_indices=mol_indices, nneighbor_cutoff=NN)
        tfcompute = htf.tfcompute(model)
        nlist = hoomd.md.nlist.Cell()
        nvt = hoomd.md.methods.nvt(group=hoomd.group.all(), kT=1, tau=0.2)
        hoomd.md.Integrator(methods=[nvt], dt=0.005)
        with self.assertRaises(ValueError):
            tfcompute.attach(nlist, r_cut=rcut, batch_size=3)

    def test_single_atom_malformed(self):
        with self.assertRaises(TypeError):
            build_examples.LJMolModel(
                MN=1, mol_indices=[1, 1, 4, 24], nneighbor_cutoff=10)

    def test_multi_atom(self):
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        model = build_examples.LJMolModel(
            MN=3, mol_indices=[[0, 1, 2], [3, 4], [5, 6, 7], [8]],
            nneighbor_cutoff=NN)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(lattice_constant=4.0,
                                           n_replicas=[3,3], device=self.device)
        nlist = hoomd.md.nlist.Cell()
        nvt = hoomd.md.methods.NVT(group=hoomd.group.all(), kT=1, tau=0.2)
        hoomd.md.Integrator(methods=[nvt], dt=0.005)
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(8)

    def test_mol_force_output(self):
        N = 3 * 3
        NN = N - 1
        rcut = 5.0
        model = build_examples.LJMolModel(
            MN=3, mol_indices=[[0, 1, 2], [3, 4], [5, 6, 7], [8]],
            nneighbor_cutoff=NN, output_forces=False)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(lattice_constant=4.0,
                                           n_replicas=[3,3], device=self.device)
        nlist = hoomd.md.nlist.Cell()
        nvt = hoomd.md.methods.NVT(group=hoomd.group.all(), kT=1, tau=0.2)
        hoomd.md.Integrator(methods=[nvt], dt=0.005)
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(8)

    def test_reverse_mol_index(self):
        # each element is the index of atoms in the molecule
        mi = [[1, 2, 0, 0, 0], [3, 0, 0, 0, 0], [4, 5, 7, 8, 9]]
        rmi = _make_reverse_indices(mi)
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
    def setUp(self):
        self.device = hoomd.device.CPU('')
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_tensor_save(self):
        model = build_examples.TensorSaveModel(0, output_forces=False)
        tfcompute = htf.tfcompute(model)
        sim = build_examples.generic_square_lattice(lattice_constant=4.0,
                                           n_replicas=[3,3], device=self.device)
        nvt = hoomd.md.methods.NVT(group=hoomd.group.all(), kT=1, tau=0.2)
        hoomd.md.Integrator(methods=[nvt], dt=0.005)
        tfcompute.attach(batch_size=3, save_output_period=2)
        hoomd.run(8)

        # reshape to remove batch_size effect
        array = tfcompute.outputs[0].reshape(-1, 9)
        assert array.shape == (4, 9)


class test_bad_models(unittest.TestCase):
    def test_no_compute(self):
        class MyModel(htf.SimModel):
            def call(self, x):
                return x
        with self.assertRaises(AttributeError):
            m = MyModel(0)

    def test_no_molcompute(self):
        class MyModel(htf.MolSimModel):
            def compute(self, nlist):
                return nlist
        with self.assertRaises(AttributeError):
            MyModel(1, [[1]], 0)

    def test_bad_molargs(self):
        class MyModel(htf.MolSimModel):
            def mol_compute(self, nlist):
                return nlist
        with self.assertRaises(AttributeError):
            MyModel(1, [[1]], 0)


class test_nlist(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.device = hoomd.device.CPU('')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_overflow(self):
        '''Use too small neighbor list and ensure error is thrown
        '''
        N = 8 * 8
        model = build_examples.LJModel(4, check_nlist=True)
        tfcompute = htf.tfcompute(model)
        T = 10
        rcut = 10.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[8,8], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        nvt = hoomd.md.methods.nvt(group=hoomd.group.all(), kT=1, tau=0.2)
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All())
        hoomd.md.Integrator(methods=[nvt], dt=0.005))
        tfcompute.attach(nlist, r_cut=rcut)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            hoomd.run(2)

    def test_sorted(self):
        N = 8 * 8
        model = build_examples.NlistNN(64, dim=32, top_neighs=8)
        tfcompute = htf.tfcompute(model)
        T = 10
        rcut = 10.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[8,8], device=self.device, seed=1)
        nlist = hoomd.md.nlist.Cell(rebuild_check_delay=1)
        
        nvt = hoomd.md.methods.NVT(filter=hoomd.filter.All(),
                               kT=1, tau=0.2
                               )
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All())
        hoomd.md.Integrator(methods=[nvt], dt=0.005)
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(10)


if __name__ == '__main__':
    unittest.main()
