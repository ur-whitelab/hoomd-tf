import hoomd, hoomd.md, hoomd.dump, hoomd.group
import numpy as np
from hoomd.tensorflow_plugin import tfcompute
import tensorflow as tf
from math import sqrt
from sys import argv as argv

if(len(argv) != 2):
    print('Usage: basic_ann_ff.py [N_PARTICLES]')
    exit(0)

N = int(argv[1])


model_dir = '/tmp/ann-training'

np.random.seed(42)

with hoomd.tensorflow_plugin.tfcompute(model_dir, _mock_mode=False, write_tensorboard=True) as tfcompute:
    hoomd.context.initialize('--gpu_error_checking')
    rcut = 3.0
    sqrt_N = int(sqrt(N))#MAKE SURE THIS IS A WHOLE NUMBER???
    
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[sqrt_N, sqrt_N])
    nlist = hoomd.md.nlist.cell(check_period = 1)
    lj = hoomd.md.pair.lj(rcut, nlist)#basic LJ forces from HOOMD
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=0.2, seed=42)
    #hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=1.2, seed=42)
    #equilibrate for 4k steps first
    hoomd.run(4000)
    #now attach the trainable model
    tfcompute.attach(nlist, r_cut=rcut)
    hoomd.analyze.log(filename='TRAINING_log.log',
                      quantities = ['potential_energy','temperature'],
                      period=100,
                      overwrite=True)
    hoomd.dump.gsd(filename='TRAINING_trajectory.gsd', period=10, group=hoomd.group.all(), overwrite=True)
    #train on 5k timesteps
    hoomd.run(5000)
