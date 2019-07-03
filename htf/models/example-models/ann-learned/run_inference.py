import hoomd
import hoomd.md
import hoomd.dump
import hoomd.group
from hoomd.htf import tfcompute
import tensorflow as tf
from sys import argv as argv
from math import sqrt
if(len(argv) != 3):
    print('Usage: basic_ann_ff.py [N_PARTICLES] [training_dir]')
    exit(0)
N = int(argv[1])
training_dir = argv[2]
with hoomd.htf.tfcompute(bootstrap=training_dir
                                       ) as tfcompute:
    hoomd.context.initialize('--gpu_error_checking')
    sqrt_N = int(sqrt(N))  # MAKE SURE THIS IS A WHOLE NUMBER???
    rcut = 3.0
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[sqrt_N, sqrt_N])
    nlist = hoomd.md.nlist.cell(check_period=1)
    # we're loading forces now, so no HOOMD calculation
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=0.2, seed=42)
    hoomd.md.constrain.oneD(group=hoomd.group.all(),
                            constraint_vector=[1, 1, 0])
    # equilibrate for 4k steps first
    hoomd.run(4000)
    tfcompute.attach(nlist, r_cut=rcut,
                     feed_func=lambda x: {'keep_prob:0': 0.5})
    hoomd.analyze.log(filename='INFERENCE_log.log',
                      quantities=['potential_energy', 'temperature'],
                      period=10,
                      overwrite=True)
    hoomd.dump.gsd(filename='INFERENCE_trajectory.gsd',
                   period=10, group=hoomd.group.all(), overwrite=True)
    # run for 5k steps with dumped trajectory and logged PE and T
    hoomd.run(5000)
