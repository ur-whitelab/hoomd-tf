import hoomd
import hoomd.md
import hoomd.dump
import hoomd.group
import hoomd.benchmark
import numpy as np
from hoomd.tensorflow_plugin import tfcompute
import tensorflow as tf
from math import sqrt
from sys import argv as argv
import time

if(len(argv) != 2):
    print('Usage: basic_ann_ff.py [N_PARTICLES]')
    exit(0)


N = int(argv[1])
model_dir = '/scratch/rbarret8/ann-training'
np.random.seed(42)
# start_time = time.time()


with hoomd.tensorflow_plugin.tfcompute(model_dir,
                                       _mock_mode=False,
                                       write_tensorboard=True) as tfcompute:
    hoomd.context.initialize('--mode=gpu')
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
    # hoomd.md.integrate.nve(group=hoomd.group.all()).
    # randomize_velocities(kT=1.2, seed=42)
    # equilibrate for 4k steps first
    hoomd.run(4000)
    # now attach the trainable model
    tfcompute.attach(nlist,
                     r_cut=rcut,
                     save_period=10,
                     period=100,
                     feed_func=lambda x: {'keep_prob:0': 0.8})
    # train on 50k timesteps
    hoomd.run(50000)

