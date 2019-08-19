import tensorflow as tf
from hoomd.htf import graph_builder
import pickle
import gsd
import gsd.hoomd
import numpy as np
import sys
if(len(sys.argv) != 2):
    print('Usage: build_scattering.py [model_dir]')
    exit(0)

model_dir = sys.argv[1]

struct_dir = '/scratch/hgandhi/run_scattering/'

# load topology information
with open(struct_dir + 'water.pickle', 'rb') as f:
    param_sys, _ = pickle.load(f)
g = gsd.hoomd.open(struct_dir + 'water.gsd')
frame = g[0]

NN = 256
# build scattering cross-section
data = [None for _ in range(NN)]
# Neutron Scattering lengths (NIST)
# Hydrogen
data[1] = -3.742
# Carbon
data[6] = 6.646
# Oxygen
data[8] = 5.805
N = len(param_sys.atoms)
cross = np.zeros(len(frame.particles.types), dtype=np.float32)

for a in param_sys.atoms:
    cross[frame.particles.types.index(a.type)] = data[a.element]

graph = graph_builder(NN, output_forces=False)
# get pairwise scattering length bi*bj
bj = tf.gather(cross, tf.cast(graph.nlist[:, :, 3], tf.int32))
bi = tf.gather(cross, tf.cast(graph.positions[:, 3], tf.int32))
bij = tf.einsum('ij,i -> ij', bj, bi)
# get neighbor list
nlist = graph.nlist[:, :, :3]
# get interatomic distances
r = tf.norm(nlist, axis=2)
q = tf.lin_space(0.01, 10., 100)
# q_rij must be QxNxNN - Outer product
# qr[i, j, k] = q[i] * r[j,k]
qr = tf.einsum('i, jk -> ijk', q, r)
intensities = tf.multiply(0.5, tf.reduce_sum(
        tf.reduce_sum(bij*tf.sin(qr)*graph_builder.safe_div(
                1., qr), axis=2), axis=1))
# print_node = tf.Print(intensities, [ intensities], summarize=1000)
# Keep a running average of intensity
avg = tf.Variable(tf.zeros(tf.shape(q)), name='intensity')
steps = tf.Variable(1., name='n')
steps_1 = tf.assign_add(steps, 1.)
avg_op = tf.assign_add(avg, (intensities - avg)/steps)

graph.save(model_directory=model_dir,
           out_nodes=[avg_op, steps_1])
