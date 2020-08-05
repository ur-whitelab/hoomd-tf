import shutil
import tempfile
import build_examples
import hoomd
import hoomd.htf
import unittest
import numpy as np
import tensorflow as tf


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
        mapping = hoomd.htf.find_molecules(self.system)
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
        mapping = hoomd.htf.find_molecules(self.system)
        s = hoomd.htf.sparse_mapping(
            [mapping_matrix for _ in mapping], mapping)
        # see if we can build it
        N = len(self.system.particles)
        p = tf.ones(shape=[N, 1])
        m = tf.sparse.sparse_dense_matmul(s, p)
        dense_mapping = tf.sparse.to_dense(s)
        msum = tf.reduce_sum(input_tensor=m)
        # here we are doing sum, not center of mass.
        # So this is like com forces
        # number of nonzero mappeds = number of molecules
        # * number of particles in each molecule
        assert int(msum) == len(mapping) * mapping_matrix.shape[1]
        # now make sure we see the mapping matrix in first set
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
        mapping = hoomd.htf.find_molecules(self.system)
        s = hoomd.htf.sparse_mapping([mapping_matrix
                                      for _ in mapping], mapping, self.system)
        # see if we can build it
        N = len(self.system.particles)
        positions = tf.cast(
            self.system.take_snapshot().particles.position, tf.float32)
        box_size = [self.system.box.Lx, self.system.box.Ly, self.system.box.Lz]
        com = hoomd.htf.center_of_mass(positions, s, box_size)
        non_pbc_com = tf.sparse.sparse_dense_matmul(s, positions)
        # TODO: Come up with a real test of this.
        assert True

    def test_compute_nlist(self):
        N = 10
        positions = tf.tile(tf.reshape(tf.range(N), [-1, 1]), [1, 3])
        box_size = [100., 100., 100.]
        nlist = hoomd.htf.compute_nlist(
            tf.cast(
                positions,
                tf.float32),
            100.,
            9,
            box_size,
            True)
        nlist = nlist.numpy()
        # particle 1 is closest to 0
        np.testing.assert_array_almost_equal(nlist[0, 0, :], [1, 1, 1, 1])
        # particle 0 is -9 away from 9
        np.testing.assert_array_almost_equal(nlist[-1, -1, :],
                                             [-9, -9, -9, 0])

    def test_compute_nlist_cut(self):
        N = 10
        positions = tf.tile(tf.reshape(tf.range(N), [-1, 1]), [1, 3])
        box_size = [100., 100., 100.]
        nlist = hoomd.htf.compute_nlist(
            tf.cast(
                positions,
                tf.float32),
            5.5,
            9,
            box_size,
            True)
        nlist = nlist.numpy()
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

        model = build_examples.CustomNlist(32, output_forces=False)
        model.r_cut = rcut
        tfcompute = hoomd.htf.tfcompute(model)
        nlist = hoomd.md.nlist.cell()
        lj = hoomd.md.pair.lj(r_cut=rcut, nlist=nlist)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=0.001)
        hoomd.md.integrate.nve(group=hoomd.group.all(
        )).randomize_velocities(seed=1, kT=0.8)
        tfcompute.attach(nlist, r_cut=rcut,
                         save_output_period=100, train=False)
        # add lj so we can hopefully get particles mixing
        hoomd.run(101)
        # the two nlists need to be sorted to be compared
        nlist = tfcompute.outputs[0]
        cnlist = tfcompute.outputs[1]
        for i in range(nlist.shape[0]):
            ni = np.sort(nlist[i, :])
            ci = np.sort(cnlist[i, :])
            np.testing.assert_array_almost_equal(ni, ci, decimal=5)

    def test_compute_pairwise_potential(self):
        model_dir = build_examples.lj_rdf(9 - 1, self.tmp)
        with hoomd.htf.tensorflowcompute.tfcompute(model_dir) as tfcompute:
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
        potential, forces = hoomd.htf.compute_pairwise_potential(model_dir,
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
        with hoomd.htf.tensorflowcompute.tfcompute(model_dir) as tfcompute:
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


class test_mol_properties(unittest.TestCase):
    def disabled_test_mol_features(self):
        import hoomd.md
        import hoomd.group
        import gsd
        import gsd.hoomd
        import os
        test_gsd = os.path.join(os.path.dirname(__file__), 'meth.gsd')
        # g = gsd.hoomd.open(test_gsd)
        set_rcut = 6.0
        c = hoomd.context.initialize()
        system = hoomd.init.read_gsd(filename=test_gsd)
        c.sorter.disable()
        model_dir = build_examples.mol_features_graph()
        with hoomd.htf.tensorflowcompute.tfcompute(model_dir) as tfcompute:
            nlist = hoomd.md.nlist.cell()
            # set-up pppm
            charged = hoomd.group.all()
            pppm = hoomd.md.charge.pppm(nlist=nlist, group=charged)
            pppm.set_params(Nx=32, Ny=32, Nz=32, order=6, rcut=set_rcut)
            # set-up pair coefficients
            nlist.reset_exclusions(['1-2', '1-3', '1-4', 'body'])
            lj = hoomd.md.pair.force_shifted_lj(r_cut=set_rcut, nlist=nlist)
            forces = [lj]
            lj.pair_coeff.set(
                "opls_156",
                "opls_156",
                sigma=2.5,
                epsilon=0.0299)
            lj.pair_coeff.set(
                "opls_156",
                "opls_157",
                sigma=2.9580,
                epsilon=0.0445)
            lj.pair_coeff.set(
                "opls_156",
                "opls_154",
                sigma=2.7929,
                epsilon=0.0714)
            lj.pair_coeff.set("opls_156", "opls_155", sigma=5.0, epsilon=0.0)
            lj.pair_coeff.set("opls_157", "opls_157", sigma=3.5, epsilon=0.066)
            lj.pair_coeff.set(
                "opls_157",
                "opls_154",
                sigma=3.3045,
                epsilon=0.1059)
            lj.pair_coeff.set(
                "opls_157",
                "opls_155",
                sigma=5.9161,
                epsilon=0.0)
            lj.pair_coeff.set(
                "opls_154",
                "opls_154",
                sigma=3.12,
                epsilon=0.1699)
            lj.pair_coeff.set(
                "opls_154",
                "opls_155",
                sigma=5.5857,
                epsilon=0.0)
            lj.pair_coeff.set("opls_155", "opls_155", sigma=10.0, epsilon=0.0)
            # set-up special pairs
            hoomd_special_coul = hoomd.md.special_pair.coulomb()
            hoomd_special_lj = hoomd.md.special_pair.lj()
            hoomd_special_lj.pair_coeff.set(
                "opls_155-opls_156", epsilon=0.0, sigma=5.0, r_cut=10.0)
            hoomd_special_coul.pair_coeff.set(
                "opls_155-opls_156", alpha=0.5, r_cut=10.0)
            # set-up bonds
            harmonic = hoomd.md.bond.harmonic()
            harmonic.bond_coeff.set("opls_156-opls_157", k=339.9999, r0=1.09)
            harmonic.bond_coeff.set("opls_154-opls_157", k=319.9999, r0=1.41)
            harmonic.bond_coeff.set("opls_154-opls_155", k=552.9999, r0=0.945)
            # set-up angles
            harm_angle = hoomd.md.angle.harmonic()
            harm_angle.angle_coeff.set(
                "opls_154-opls_157-opls_156", k=70.0, t0=1.9111)
            harm_angle.angle_coeff.set(
                "opls_155-opls_154-opls_157", k=110.0, t0=1.8937)
            harm_angle.angle_coeff.set(
                "opls_156-opls_157-opls_156", k=66.0, t0=1.8815)
            # set-up dihedrals
            dihedral = hoomd.md.dihedral.opls()
            dihedral.dihedral_coeff.set(
                "opls_155-opls_154-opls_157-opls_156",
                k1=0.0,
                k2=0.0,
                k3=0.45,
                k4=0.0)
            group_all = hoomd.group.all()
            kT = 1.9872 / 1000
            # Now NVE
            im = hoomd.md.integrate.mode_standard(dt=0.0409)
            nvt = hoomd.md.integrate.nvt(
                group=group_all, kT=298.15 * kT, tau=350 / 48.9)
            nvt.randomize_velocities(1234)
            hoomd.run(500)
            tfcompute.attach(nlist, r_cut=set_rcut, period=1, save_period=10)
            hoomd.run(100)
            variables = hoomd.htf.load_variables(
                model_dir, ['avg_r', 'avg_a', 'avg_d'])
            assert np.isfinite(variables['avg_r'])
            assert np.isfinite(variables['avg_a'])
            assert np.isfinite(variables['avg_d'])


class test_trajectory(unittest.TestCase):
    def test_run_from_trajectory(self):
        try:
            import MDAnalysis as mda
        except ImportError:
            self.skipTest(
                "MDAnalysis not available; skipping test_run_from_trajectory")
        import math
        import os
        test_pdb = os.path.join(os.path.dirname(__file__), 'test_topol.pdb')
        test_traj = os.path.join(os.path.dirname(__file__), 'test_traj.trr')
        universe = mda.Universe(test_pdb, test_traj)
        # load example graph that calculates average energy
        model_directory = build_examples.run_traj_graph()
        hoomd.htf.run_from_trajectory(model_directory, universe,
                                      period=1, r_cut=25.)
        # get evaluated outnodes
        variables = hoomd.htf.load_variables(model_directory,
                                             ['average-energy'])
        # assert they are calculated and valid?
        assert not math.isnan(variables['average-energy'])
        assert not variables['average-energy'] == 0.0


if __name__ == '__main__':
    unittest.main()
