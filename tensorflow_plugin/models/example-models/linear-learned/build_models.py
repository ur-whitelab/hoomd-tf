import hoomd.tensorflow_plugin
import tensorflow as tf

def make_train_graph(N, NN, directory):
    # build a model that fits the energy to a linear term
    graph = hoomd.tensorflow_plugin.graph_builder(N, NN, output_forces=False)
    # get r
    nlist = graph.nlist[:, :, :3]
    r = graph.safe_norm(nlist, axis=2)
    # build energy model
    m = tf.Variable(1.0, name='m')
    b = tf.Variable(0.0, name='b')
    predicted_particle_energy = tf.reduce_sum(m * r + b, axis=1)
    # get energy from hoomd
    particle_energy = graph.forces[:, 3]
    # make them match
    loss = tf.losses.mean_squared_error(particle_energy, predicted_particle_energy)
    optimize = tf.train.AdamOptimizer(1e-3).minimize(loss)
    graph.save(model_directory=directory, out_nodes=[optimize])

def make_force_graph(N, NN, directory):
    # this model applies the variables learned in the example above
    # to compute forces
    graph = hoomd.tensorflow_plugin.graph_builder(N, NN)
    # get r
    nlist = graph.nlist[:, :, :3]
    r = graph.safe_norm(nlist, axis=2)
    # build energy model
    m = tf.Variable(1.0, name='m')
    b = tf.Variable(0.0, name='b')
    predicted_particle_energy = tf.reduce_sum(m * r + b, axis=1)
    forces = graph.compute_forces(predicted_particle_energy)
    graph.save(force_tensor=forces, model_directory=directory)
make_train_graph(64, 16, '/tmp/training')
make_force_graph(64, 16, '/tmp/inference')