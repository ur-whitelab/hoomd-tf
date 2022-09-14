# Copyright (c) 2020 HOOMD-TF Developers
import build_examples
import hoomd
import hoomd.htf as htf
import unittest
import tensorflow as tf


class test_layers(unittest.TestCase):
    def test_wca(self):
        device = hoomd.device.CPU('')
        model = build_examples.WCA(32)
        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        sim = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[3, 3],
            device=device)
        nlist = hoomd.md.nlist.Cell()
        #TODO: update syntax
        hoomd.md.integrate.mode_standard(dt=0.001)
        hoomd.md.integrate.nve(group=hoomd.group.all(
        )).randomize_velocities(seed=1, kT=0.8)
        tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
        sim.run(10)

    def test_rbf(self):
        hoomd.device.CPU('')
        rbf = htf.RBFExpansion(0, 2, 10)
        nlist = tf.ones((10, 6, 3))
        r = htf.safe_norm(nlist, axis=2)
        out = rbf(r)
        self.assertEqual(out.shape, (10, 6, 10))


if __name__ == '__main__':
    unittest.main()
