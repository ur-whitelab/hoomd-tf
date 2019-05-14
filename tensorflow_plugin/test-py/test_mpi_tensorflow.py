import hoomd.tensorflow_plugin as htf
import hoomd
from hoomd import md
from hoomd import comm
import unittest
import build_examples
import numpy as np


def run_tf_lj(N, T):
    model_dir = build_examples.lj_graph(N - 1)
    with hoomd.tensorflow_plugin.tfcompute(model_dir, _mock_mode=False) as tfcompute:
        rcut = 5.0
        system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                           n=[np.sqrt(N).astype(np.int),np.sqrt(N).astype(np.int)])
        nlist = md.nlist.cell(check_period = 1)
        md.integrate.mode_standard(dt=0.005)
        md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)               
        tfcompute.attach(nlist, r_cut=rcut)
        hoomd.run(2)
        tf_forces = []
        for i in range(T):
            hoomd.run(1)
            snapshot = system.take_snapshot()
            tf_forces.append([system.particles[j].net_force for j in range(N)])            
        tf_forces = np.array(tf_forces)
    return tf_forces


def run_hoomd_lj(N, T):
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                       n=[np.sqrt(N).astype(np.int),np.sqrt(N).astype(np.int)])
    nlist = md.nlist.cell(check_period = 1)
    md.integrate.mode_standard(dt=0.005)
    md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.2).randomize_velocities(seed=1)
    lj = md.pair.lj(r_cut=5.0, nlist=nlist)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    
    hoomd.run(2)
    lj_forces = []
    for i in range(T):
        hoomd.run(1)
        snapshot = system.take_snapshot()
        lj_forces.append([system.particles[j].net_force for j in range(N)])
    lj_forces = np.array(lj_forces)
    return lj_forces

class test_mpi(unittest.TestCase):
    def test_lj_forces(self):
        # need to be big enough for MPI testing
        # Needs to be perfect square
        N = 225
        T = 32        
        # try normal and then uneven fraction
        params = [{'nx': 2}, {'x': [0.33]}]
        for p in params:
            with self.subTest(decomp = p):
                hoomd.context.initialize()
                comm.decomposition(**p)
                tf_forces = run_tf_lj(N, T) 
                hoomd.context.initialize()
                comm.decomposition(**p)
                hoomd_forces = run_hoomd_lj(N, T) 
                print(N, T, tf_forces.shape)
                for i in range(T):
                    for j in range(N):
                        np.testing.assert_allclose(tf_forces[i,j], hoomd_forces[i,j], atol=1e-5)

if __name__ == '__main__':
    unittest.main()
