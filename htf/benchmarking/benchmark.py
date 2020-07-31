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


if(len(argv) != 4):
    print('Usage: benchmark_xla.py [N_PARTICLES (int)]\
                                   [EXECUTION_MODE (str, either "cpu" or "gpu")]\
                                   [SAVE_DIRECTORY]')
    exit(0)

N = int(argv[1])
mode_string = argv[2].lower()
save_directory = argv[3]

if mode_string != 'cpu' and mode_string != 'gpu':
    raise(ValueError('Execution mode argument must be either "cpu" or "gpu".'))
model_dir = '{}/{}_benchmarking'.format(save_directory, mode_string)


def lj_force_matching(NN=15, directory='/tmp/test-lj-force-matching'):
    graph = htf.graph_builder(NN, output_forces=False)
    # make trainable variables
    epsilon = tf.Variable(0.9, name='lj-epsilon', trainable=True)
    sigma = tf.Variable(1.1, name='lj-sigma', trainable=True)
    # get LJ potential using our variables
    # uses built in nlist_rinv which provides
    # r^-1 with each neighbor
    inv_r6 = sigma**6 * graph.nlist_rinv**6
    # use 2 * epsilon because nlist is double-counted
    p_energy = 2.0 * epsilon * (inv_r6**2 - inv_r6)
    # sum over pairs to get total energy
    energy = tf.reduce_sum(input_tensor=p_energy, axis=1, name='energy')
    # compute forces
    computed_forces = graph.compute_forces(energy)
    # compare hoomd-blue forces (graph.forces) with our
    # computed forces
    minimizer, loss = htf.force_matching(graph.forces[:, :3],
                                         computed_forces[:, :3],
                                         learning_rate=1e-2)
    # save loss so we can visualize later
    graph.save_tensor(loss, 'loss')
    # Make sure to have minimizer in out_nodes so that
    # the force matching occurs!
    graph.save(model_directory=directory,
               out_nodes=[minimizer])
    return directory


NN = 63

lj_force_matching(NN, model_dir)

with hoomd.htf.tfcompute(model_dir,
                         _mock_mode=False,
                         write_tensorboard=False,
                         ) as tfcompute:
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
                     r_cut=rcut,
                     save_period=100,
                     period=100)
    # train on 5k timesteps
    hoomd.run(50000)
    # train on 5k timesteps and benchmark with 20 repeats
    benchmark_results = hoomd.benchmark.series(warmup=6000,
                                               repeat=5,
                                               steps=50000,
                                               limit_hours=2)

# write results
with open('{}-particles_{}_time.txt'.format(N, mode_string), 'w+') as f:
    f.write('Elapsed time with {} particles: {}'.format(N, str(benchmark_results)))

print('Elapsed time with {} particles: {}'.format(N, str(benchmark_results)))
