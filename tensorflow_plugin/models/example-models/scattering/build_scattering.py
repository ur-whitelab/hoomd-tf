import tensorflow as tf
from hoomd.tensorflow_plugin import graph_builder
import numpy as np

b = -3.739 #assuming bunch of Hydrogens
N = 64
graph = graph_builder(N,N-1, output_forces=False)
#get neighbor list
nlist = graph.nlist[:,:,:3]
#get interatomic distances
r = tf.norm(nlist, axis=2)
q = np.linspace(0.1, 10., N)
intensities = []
for i in range(len(q)):
    q_rij = tf.scalar_mul(q[i], r)
    intensities.append( 0.5 * tf.reduce_sum(b*b* tf.sin(q_rij) * graph_builder.safe_div(1., q_rij)))
intensities = tf.convert_to_tensor(intensities, dtype=float)
print_node = tf.Print(intensities, [ intensities], summarize=1000)

graph.save(model_directory='/home/hgandhi/hoomd-tf/tensorflow_plugin/models/my_model/',  out_nodes = [print_node])

