import hoomd
import hoomd.tensorflow_plugin as htf
import unittest
import numpy as np
import tensorflow as tf


class test_loading(unittest.TestCase):
    def test_load_variables(self):
        print('ONLY DISPLAY ONCE')
        model_dir = '/tmp/test-load'
        # make model that does assignment
        g = htf.graph_builder(0, False)
        h = tf.ones([10], dtype=tf.float32)
        v = tf.get_variable('test', shape=[], trainable=False)
        as_op = v.assign(tf.reduce_sum(h))
        g.save(model_dir, out_nodes=[as_op])
        # run once
        with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:
            hoomd.context.initialize()
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                n=[3,3])
            hoomd.md.integrate.mode_standard(dt=0.005)
            hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=2, seed=2)
            tfcompute.attach(save_period=1)
            hoomd.run(1)
        # load
        vars = htf.load_variables(model_dir, ['test'])
        print(vars)
        assert np.abs(vars['test'] - 10) < 10e-10




class test_mappings(unittest.TestCase):
    def setUp(self):
        # build system using example from hoomd
        hoomd.context.initialize()
        snapshot = hoomd.data.make_snapshot(N=10,
                                        box=hoomd.data.boxdim(Lx=10, Ly=0.5, Lz=0.5),
                                        particle_types=['A', 'B'],
                                        bond_types=['polymer'])
        snapshot.particles.position[:] = [[-4.5, 0, 0], [-3.5, 0, 0],
                                        [-2.5, 0, 0], [-1.5, 0, 0],
                                        [-0.5, 0, 0], [0.5, 0, 0],
                                        [1.5, 0, 0], [2.5, 0, 0],
                                        [3.5, 0, 0], [4.5, 0, 0]]


        snapshot.particles.typeid[0:7]=0
        snapshot.particles.typeid[7:10]=1


        snapshot.bonds.resize(9)
        snapshot.bonds.group[:] = [[0,1], [1, 2], [2,3],
                                [3,4], [4,5], [5,6],
                                [6,7], [7,8], [8,9]]
        snapshot.replicate(1,3,3)
        self.system = hoomd.init.read_snapshot(snapshot)
    def test_find_molecules(self):
        # test out mapping
        mapping = htf.find_molecules(self.system)
        assert len(mapping) == 9
        assert len(mapping[0]) == 10
    def test_sparse_mapping(self):
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
        s = htf.sparse_mapping([mapping_matrix for _ in mapping], mapping)
        # see if we can build it
        N = len(self.system.particles)
        p = tf.ones(shape=[N, 1])
        m = tf.sparse.matmul(s, p)
        dense_mapping = tf.sparse.to_dense(s)
        msum = tf.reduce_sum(m)
        with tf.Session() as sess:
            msum = sess.run(msum)
            assert int(msum) == len(mapping) *  mapping_matrix.shape[0]
            #now make sure we see the mapping matrix in first set
            dense_mapping = sess.run(dense_mapping)

        map_slice = dense_mapping[:mapping_matrix.shape[0], :mapping_matrix.shape[1]]
        #make mapping_matrix sum to 1
        ref_slice = mapping_matrix / np.sum(mapping_matrix, axis=1).reshape(-1,1)
        print(map_slice, ref_slice)
        np.testing.assert_array_almost_equal(map_slice, ref_slice)

        #check off-diagnoal slice, which should be 0
        map_slice = dense_mapping[:mapping_matrix.shape[0], -mapping_matrix.shape[1]:]
        assert np.sum(map_slice) < 1e-10

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
        s = htf.sparse_mapping([mapping_matrix for _ in mapping], mapping, self.system)
        # see if we can build it
        N = len(self.system.particles)
        p = tf.placeholder(tf.float32, shape=[N, 3])
        com = htf.center_of_mass(p, s, self.system)
        with tf.Session() as sess:
            positions = self.system.take_snapshot().particles.position
            com = sess.run(com, feed_dict={p:positions})
        print(com)
        assert False

if __name__ == '__main__':
    unittest.main()