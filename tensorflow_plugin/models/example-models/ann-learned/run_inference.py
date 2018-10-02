import hoomd, hoomd.md, hoomd.dump, hoomd.group
import numpy as np
from hoomd.tensorflow_plugin import tfcompute
import tensorflow as tf
import sys

training_dir = '/tmp/ann-training'
inference_dir = '/tmp/ann-inference'

np.random.seed(42)

with hoomd.tensorflow_plugin.tfcompute(inference_dir, bootstrap = training_dir) as tfcompute:
    hoomd.context.initialize('--gpu_error_checking')
    N = 8 * 8
    NN = N - 1
    rcut = 3.0
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[8,8])
    nlist = hoomd.md.nlist.cell(check_period = 1)
    #we're loading forces now, so no HOOMD calculation
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=0.2, seed=42)
    hoomd.md.constrain.oneD(group=hoomd.group.all(), constraint_vector=[1,1,0])

    tfcompute.attach(nlist, r_cut=rcut)
    hoomd.analyze.log(filename='INFERENCE_log.log',
                      quantities = ['potential_energy','temperature'],
                      period=10,
                      overwrite=True)
    hoomd.dump.gsd(filename='INFERENCE_trajectory.gsd', period=10, group=hoomd.group.all(), overwrite=True)
    hoomd.run(5000)

