import tensorflow as tf
from hoomd.tensorflow_plugin import graph_builder
b = -3.739  # assuming bunch of Hydrogens
N = 64
graph = graph_builder(N, N-1, output_forces=False)
# get neighbor list
nlist = graph.nlist[:, :, :3]
# get interatomic distances
r = tf.norm(nlist, axis=2)
q = tf.lin_space(0., 10., 100)
# q_rij must be QxNxNN - Outer product
# qr[i, j, k] = q[i] * r[j,k]
qr = tf.einsum('i, jk -> ijk', q, r)
intensities = tf.multiply(0.5, tf.reduce_sum(
        tf.reduce_sum(b*b*tf.sin(qr)*graph_builder.safe_div(
                1., qr), axis=2), axis=1))
# print_node = tf.Print(intensities, [ intensities], summarize=1000)

avg = tf.Variable(tf.zeros(tf.shape(q)), name='intensity')
steps = tf.Variable(1., name='n')
steps_1 = tf.assign_add(steps, 1.)
avg_op = tf.assign_add(avg, (intensities - avg)/steps)

graph.save(model_directory='/scratch/hgandhi/scattering/',
           out_nodes=[avg_op, steps_1])
