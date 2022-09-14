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
        device = hoomd.device.GPU()
        system = build_examples.generic_square_lattice(
            lattice_constant=4.0,
            n_replicas=[32, 32],
            device=device)
        #TODO: update syntax
        hoomd.md.integrate.mode_standard(dt=0.005)
        hoomd.md.integrate.nve(group=hoomd.group.all(
        )).randomize_velocities(kT=2, seed=2)
        tfcompute.attach()
        sim.run(10)


if __name__ == '__main__':
    unittest.main()
