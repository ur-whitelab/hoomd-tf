# Copyright (c) 2020 HOOMD-TF Developers
import hoomd.htf as htf
import hoomd
from hoomd import md
from hoomd import comm
from hoomd import init
import unittest
import build_examples
import numpy as np
import tempfile
import shutil


def run_tf_lj(N, T, device):
    model = build_examples.LJModel(256)
    tfcompute = hoomd.htf.tfcompute(model)
    rcut = 5.0
    sim = build_exaples.generic_square_lattice(
        lattice_constant=4.0,
        n_replicas=np.ones(2)*np.sqrt(N).astype(np.int),
        device=device)
    nlist = md.nlist.Cell()
    md.integrate.mode_standard(dt=0.005)
    #TODO: syntax update
    md.integrate.nvt(group=hoomd.group.all(),
                     kT=1, tau=0.2).randomize_velocities(seed=1)
    tfcompute.attach(nlist, r_cut=rcut)
    sim.run(2)
    tf_forces = []
    for i in range(T):
        sim.run(1)
        snapshot = sim.state.get_snapshot()
        tf_forces.append([system.state.particles[j].net_force for j in range(N)])
    tf_forces = np.array(tf_forces)
    return tf_forces


def run_hoomd_lj(N, T, device):
    sim = build_exaples.generic_square_lattice(
        lattice_constant=4.0,
        n_replicas=np.ones(2)*np.sqrt(N).astype(np.int),
        device=device)
    nlist = md.nlist.Cell()
    #TODO: syntax update
    md.integrate.mode_standard(dt=0.005)
    md.integrate.nvt(group=hoomd.group.all(),
                     kT=1, tau=0.2).randomize_velocities(seed=1)
    lj = md.pair.lj(r_cut=5.0, nlist=nlist)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

    sim.run(2)
    lj_forces = []
    for i in range(T):
        sim.run(1)
        snapshot = sim.state.get_snapshot()
        lj_forces.append([sim.state.particles[j].net_force for j in range(N)])
    lj_forces = np.array(lj_forces)
    return lj_forces


class test_mpi(unittest.TestCase):

    def test_lj_forces(self):
        # need to be big enough for MPI testing
        # Needs to be perfect square
        N = 1024
        T = 32
        # try normal and then uneven fraction
        params = [{'nx': 2}, {'x': [0.33]}]
        for p in params:
            with self.subTest(decomp=p):
                device = hoomd.device.CPU('')
                comm.decomposition(**p)
                tf_forces = run_tf_lj(N, T, device)
                device = hoomd.device.CPU('')
                comm.decomposition(**p)
                hoomd_forces = run_hoomd_lj(N, T, device)
                print(N, T, tf_forces.shape)
                for i in range(T):
                    for j in range(N):
                        np.testing.assert_allclose(tf_forces[i, j],
                                                   hoomd_forces[i, j],
                                                   atol=1e-5)


if __name__ == '__main__':
    unittest.main()
