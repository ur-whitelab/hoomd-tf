import hoomd.htf as htf
from hoomd.htf import tfcompute
import hoomd
import hoomd.md
import hoomd.dump
import hoomd.group
import hoomd.benchmark
from math import sqrt
import numpy as np
from sys import argv as argv
import tensorflow as tf

'''This script runs a quick benchmark with a simple TensorFlow model
   and writes out the results to file.'''


if(len(argv) != 3):
    print('Usage: benchmark.py [N_PARTICLES (int)]\
                                   [EXECUTION_MODE (str, either "cpu" or "gpu")]')

    exit(0)

N = int(argv[1])
mode_string = argv[2].lower()

if mode_string != 'cpu' and mode_string != 'gpu':
    raise(ValueError('Execution mode argument must be either "cpu" or "gpu".'))


NN = 64
class LJModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        # get r
        rinv = htf.nlist_rinv(nlist)
        inv_r6 = rinv**6
        # pairwise energy. Double count -> divide by 2
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces

model = LJModel(NN)
tfcompute = htf.tfcompute(model)
hoomd.context.initialize('--mode={}'.format(mode_string))
rcut = 3.0
sqrt_N = int(sqrt(N))

system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                   n=[sqrt_N, sqrt_N])
nlist = hoomd.md.nlist.cell(check_period=1)
# basic LJ forces from HOOMD
lj = hoomd.md.pair.lj(rcut, nlist)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
hoomd.md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.0, seed=42)
# equilibrate for 4k steps first
hoomd.run(4000)
# now attach the LJ force model
tfcompute.attach(nlist,
                 r_cut=rcut)

# train on 5k timesteps
hoomd.run(5000)
# train on 5k timesteps and benchmark with 20 repeats
benchmark_results = hoomd.benchmark.series(warmup=6000,
                                           repeat=5,
                                           steps=50000,
                                           limit_hours=2)

# write results
with open('{}-particles_{}_time.txt'.format(N, mode_string), 'w+') as f:
    f.write('Elapsed time with {} particles: {}'.format(N, str(benchmark_results)))

print('Elapsed time with {} particles: {}'.format(N, str(benchmark_results)))
