import unittest
import tensorflow as tf
import hoomd
import hoomd.md
import hoomd.htf
import build_examples


class test_force_gpu(unittest.TestCase):
    def test_force_overwrite(self):
        model = build_examples.BenchmarkNonlistModel(0)
        tfcompute = hoomd.htf.tfcompute(model)
        hoomd.context.initialize('--mode=gpu')
        system = hoomd.init.create_lattice(
            unitcell=hoomd.lattice.sq(a=4.0),
            n=[32, 32])
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all(
        )).randomize_velocities(kT=2, seed=2)
        tfcompute.attach()
        hoomd.run(10)


if __name__ == '__main__':
    unittest.main()
