import hoomd
import hoomd.htf as htf
import unittest
import numpy as np
import tensorflow as tf
import build_examples
import tempfile
import shutil

class test_loading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_load_variables(self):
        self.tmp = tempfile.mkdtemp()
        model_dir = self.tmp
        # make model that does assignment
        g = htf.graph_builder(0, False)
        h = tf.ones([10], dtype=tf.float32)
        v = tf.get_variable('test', shape=[], trainable=False)
        as_op = v.assign(tf.reduce_sum(h))
        g.save(model_dir, out_nodes=[as_op])
        # run once
        hoomd.context.initialize()
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(kT=2, seed=2)
            tfcompute.attach(save_period=1)
            hoomd.run(1)
        # load
        vars = htf.load_variables(model_dir, ['test'])
        assert np.abs(vars['test'] - 10) < 10e-10

class test_mappings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # build system using example from hoomd
        hoomd.context.initialize()
        snapshot = hoomd.data.make_snapshot(N=10,
                                            box=hoomd.data.boxdim(Lx=10,
                                                                  Ly=0.5,
                                                                  Lz=0.5),
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
        snapshot.replicate(1, 3, 3)
        self.system = hoomd.init.read_snapshot(snapshot)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_find_molecules(self):
        # test out mapping
        mapping = htf.find_molecules(self.system)
        assert len(mapping) == 9
        assert len(mapping[0]) == 10

    def test_sparse_mapping(self):
        '''Checks the sparse mapping when used for
        summing forces, not center of mass
        '''
        # I write this as an N x M
        # However, we need to have them as M x N, hence the
        # transpose
        mapping_matrix = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]]).transpose()
        mapping = htf.find_molecules(self.system)
        s = htf.sparse_mapping([mapping_matrix for _ in mapping], mapping)
        # see if we can build it
        N = len(self.system.particles)
        p = tf.ones(shape=[N, 1])
        m = tf.sparse.matmul(s, p)
        dense_mapping = tf.sparse.to_dense(s)
        msum = tf.reduce_sum(m)
        with tf.Session() as sess:
            msum = sess.run(msum)
            # here we are doing sum, not center of mass.
            # So this is like com forces
            # number of nonzero mappeds = number of molecules
            # * number of particles in each molecule
            assert int(msum) == len(mapping) * mapping_matrix.shape[1]
            # now make sure we see the mapping matrix in first set
            dense_mapping = sess.run(dense_mapping)
        map_slice = dense_mapping[
            :mapping_matrix.shape[0], :mapping_matrix.shape[1]]
        # make mapping_matrix sum to 1
        ref_slice = mapping_matrix
        np.testing.assert_array_almost_equal(map_slice, ref_slice)
        # check off-diagnoal slice, which should be 0
        map_slice = dense_mapping[
            :mapping_matrix.shape[0], -mapping_matrix.shape[1]:]
        assert np.sum(map_slice) < 1e-10
        # make sure the rows sum to N
        assert (np.sum(np.abs(np.sum(dense_mapping, axis=1)))
                - dense_mapping.shape[1]) < 10e-10

    def test_com(self):
        # I write this as an N x M
        # However, we need to have them as M x N, hence the
        # transpose
        mapping_matrix = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 1, 0]]).transpose()
        mapping = htf.find_molecules(self.system)
        s = htf.sparse_mapping([mapping_matrix
                                for _ in mapping], mapping, self.system)
        # see if we can build it
        N = len(self.system.particles)
        p = tf.placeholder(tf.float32, shape=[N, 3])
        com = htf.center_of_mass(p, s, self.system)
        non_pbc_com = tf.sparse.matmul(s, p)
        with tf.Session() as sess:
            positions = self.system.take_snapshot().particles.position
            com, non_pbc_com = sess.run([com, non_pbc_com],
                                        feed_dict={p: positions})
        # TODO: Come up with a real test of this.
        assert True

    def test_force_matching(self):
        model_dir = build_examples.lj_force_matching(NN=15)
        # calculate lj forces with a leading coeff
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            N = 16
            NN = N-1
            rcut = 7.5
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[4, 4])
            nlist = hoomd.md.nlist.cell(check_period=1)
            lj = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(kT=2, seed=2)
            tfcompute.attach(nlist, r_cut=rcut, save_period=10)
            hoomd.run(1e3)
            input_nlist = tfcompute.get_nlist_array()
            variables = hoomd.htf.load_variables(
                model_dir, checkpoint=10,
                names=['loss', 'lj-epsilon', 'lj-sigma'],
                feed_dict=dict({'nlist-input:0': input_nlist}))
            new_variables = hoomd.htf.load_variables(
                model_dir, checkpoint=-1,
                names=['loss', 'lj-epsilon', 'lj-sigma'],
                feed_dict=dict({'nlist-input:0': input_nlist}))
            loss = variables['loss']
            new_loss = new_variables['loss']
        assert loss != new_loss
        assert new_variables['lj-epsilon'] != 0.9
        assert new_variables['lj-sigma'] != 1.1

    def test_compute_nlist(self):
        N = 10
        positions = tf.tile(tf.reshape(tf.range(N), [-1, 1]), [1, 3])
        system = type('',
                      (object, ),
                      {'box': type('', (object,),
                       {'Lx': 100., 'Ly': 100., 'Lz': 100.})})
        nlist = htf.compute_nlist(tf.cast(positions, tf.float32),
                                  100., 9, system, True)
        with tf.Session() as sess:
            nlist = sess.run(nlist)
            # particle 1 is closest to 0
            np.testing.assert_array_almost_equal(nlist[0, 0, :], [1, 1, 1, 1])
            # particle 0 is -9 away from 9
            np.testing.assert_array_almost_equal(nlist[-1, -1, :],
                                                 [-9, -9, -9, 0])

    def test_compute_nlist_cut(self):
        N = 10
        positions = tf.tile(tf.reshape(tf.range(N), [-1, 1]), [1, 3])
        system = type('',
                      (object, ),
                      {'box': type('', (object,),
                       {'Lx': 100., 'Ly': 100., 'Lz': 100.})})
        nlist = htf.compute_nlist(tf.cast(positions, tf.float32),
                                  5.5, 9, system, True)
        with tf.Session() as sess:
            nlist = sess.run(nlist)
            # particle 1 is closest to 0
            np.testing.assert_array_almost_equal(nlist[0, 0, :], [1, 1, 1, 1])
            # particle later particles on 0 are all 0s because
            # there were not enough neigbhors
            np.testing.assert_array_almost_equal(nlist[-1, -1, :],
                                                 [0, 0, 0, 0])

    def test_nlist_compare(self):
        rcut = 5.0
        c = hoomd.context.initialize()
        # disable sorting
        if c.sorter is not None:
            c.sorter.disable()
        # want to have a big enough system so that we actually have a cutoff
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.bcc(a=4.0),
                                           n=[4, 4, 4])
        model_dir = build_examples.custom_nlist(16, rcut, system, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            nlist = hoomd.md.nlist.cell()
            lj = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
            hoomd.md.integrate.mode_standard(dt=0.001)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                                   )).randomize_velocities(seed=1, kT=0.8)
            tfcompute.attach(nlist, r_cut=rcut,
                             save_period=10, batch_size=None)
            # add lj so we can hopefully get particles mixing
            hoomd.run(100)
        variables = hoomd.htf.load_variables(
            model_dir, ['hoomd-r', 'htf-r'])
        # the two nlists need to be sorted to be compared
        nlist = variables['hoomd-r']
        cnlist = variables['htf-r']
        for i in range(nlist.shape[0]):
            ni = np.sort(nlist[i, :])
            ci = np.sort(cnlist[i, :])
            np.testing.assert_array_almost_equal(ni, ci, decimal=5)

    def test_compute_pairwise_potential(self):
        model_dir = build_examples.lj_rdf(9 - 1, self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            rcut = 2.5
            system = hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            nlist = hoomd.md.nlist.cell()
            lj = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
            hoomd.md.integrate.mode_standard(dt=0.001)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                                   )).randomize_velocities(seed=1, kT=0.8)
            tfcompute.attach(nlist, r_cut=rcut,
                             save_period=10, batch_size=None)
            # add lj so we can hopefully get particles mixing
            hoomd.run(100)
            potentials = tfcompute.get_forces_array()[3]

        r = np.linspace(0.5, 1.5, 5)
        potential, forces = htf.compute_pairwise_potential(model_dir,
                                                           r, 'energy')
        np.testing.assert_equal(len(potential), len(r),
                                'Potentials not calculated correctly')


class test_bias(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_eds(self):
        T = 1000
        hoomd.context.initialize()
        model_dir = build_examples.eds_graph(self.tmp)
        with hoomd.htf.tfcompute(model_dir) as tfcompute:
            hoomd.init.create_lattice(
                unitcell=hoomd.lattice.sq(a=4.0),
                n=[3, 3])
            hoomd.md.integrate.mode_standard(dt=0.05)
            hoomd.md.integrate.nve(group=hoomd.group.all(
                    )).randomize_velocities(kT=0.2, seed=2)
            tfcompute.attach(save_period=10)
            hoomd.run(T)
        variables = hoomd.htf.load_variables(
                model_dir, ['cv-mean', 'alpha-mean', 'eds.mean', 'eds.ssd', 'eds.n', 'eds.a'])
        assert np.isfinite(variables['eds.a'])
        assert (variables['cv-mean'] - 4)**2 < 0.5


class test_trajectory(unittest.TestCase):
    def test_run_from_trajectory(self):
        import math
        import MDAnalysis as mda
        universe = mda.Universe('test_topol.pdb', 'test_traj.trr')
        # load example graph that calculates average energy
        model_directory = build_examples.run_traj_graph()
        htf.run_from_trajectory(model_directory, universe, period=1, r_cut=25.)
        # get evaluated outnodes
        variables = hoomd.htf.load_variables(model_directory, ['average-energy'])
        # assert they are calculated and valid?
        assert not math.isnan(variables['average-energy'])
        assert not variables['average-energy'] == 0

if __name__ == '__main__':
    unittest.main()
