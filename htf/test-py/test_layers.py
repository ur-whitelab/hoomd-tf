# Copyright (c) 2020 HOOMD-TF Developers
import build_examples
import hoomd
import hoomd.htf as htf
import unittest
import tensorflow as tf


class test_layers(unittest.TestCase):
    def test_wca(self):
        hoomd.device.CPU('')
        model = build_examples.WCA(32)
        tfcompute = htf.tfcompute(model)
        rcut = 5.0
        system = hoomd.init.create_lattice(
            unitcell=hoomd.lattice.sq(a=4.0),
            n=[3, 3])
        nlist = hoomd.md.nlist.Cell()
        hoomd.md.integrate.mode_standard(dt=0.001)
        hoomd.md.integrate.nve(group=hoomd.group.all(
        )).randomize_velocities(seed=1, kT=0.8)
        tfcompute.attach(nlist, r_cut=rcut, batch_size=4)
        hoomd.run(10)

    def test_rbf(self):
        hoomd.device.CPU('')
        rbf = htf.RBFExpansion(0, 2, 10)
        nlist = tf.ones((10, 6, 3))
        r = htf.safe_norm(nlist, axis=2)
        out = rbf(r)
        self.assertEqual(out.shape, (10, 6, 10))


if __name__ == '__main__':
    unittest.main()
