import hoomd
import hoomd.tensorflow_plugin as htf
import unittest
import numpy as np
import tensorflow as tf


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
        msum = tf.reduce_sum(m)
        with tf.Session() as sess:
            msum = sess.run(msum)
        assert int(msum) == len(mapping) *  mapping_matrix.shape[0]