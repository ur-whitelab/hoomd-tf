import hoomd, hoomd.md, hoomd.dump, hoomd.group, hoomd.benchmark
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

#start_time = time.time()

with hoomd.tensorflow_plugin.tfcompute(model_dir, _mock_mode=False, write_tensorboard=True) as tfcompute:#
    hoomd.context.initialize('--mode=gpu')#'--gpu_error_checking')
    rcut = 3.0
    sqrt_N = int(sqrt(N))#MAKE SURE THIS IS A WHOLE NUMBER???
    
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[sqrt_N, sqrt_N])
    nlist = hoomd.md.nlist.cell(check_period = 1)
    lj = hoomd.md.pair.lj(rcut, nlist)#basic LJ forces from HOOMD
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.0, seed=42)
    #hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(kT=1.2, seed=42)
    #equilibrate for 4k steps first
    hoomd.run(4000)
    #now attach the trainable model
    tfcompute.attach(nlist, r_cut=rcut, save_period=10, period=100, feed_func=lambda x: {'keep_prob:0': 0.8})
    #hoomd.analyze.log(filename='TRAINING_log.log',
    #                  quantities = ['potential_energy','temperature'],
    #                  period=100,
    #                  overwrite=True)
    #hoomd.dump.gsd(filename='TRAINING_trajectory.gsd', period=10, group=hoomd.group.all(), overwrite=True)
    #train on 5k timesteps
    hoomd.run(500000)#, profile=True)
    #tain on 5k timesteps and benchmark with 20 repeats
    #benchmark_results = hoomd.benchmark.series(warmup=6000, repeat=5,steps=5000, limit_hours=2)
    

#end_time = time.time()

#with open('{}-particles_time.txt'.format(N), 'w+') as f:
#    f.write('Elapsed time with {} particles: {}'.format(N,str(benchmark_results)))
