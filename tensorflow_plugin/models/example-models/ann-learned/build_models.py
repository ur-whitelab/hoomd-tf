import hoomd
import tensorflow as tf
import numpy as np
import hoomd.tensorflow_plugin
from sys import argv as argv


# make a simple ANN, based on: https://medium.com/@curiousily/
# tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b

def perceptron_layer(x, weights, biases, keep_prob=1.0,
                     final_layer=False, activation_function=lambda x: x):
    '''Creates a single neural network layer given weights and bias tensors.
        Final layers should not have activation functions, must specify.
        Defaults to no dropuout and linear activation.'''
    layer = tf.matmul(x, weights) + biases
    if not final_layer:
        layer = activation_function(layer)  # example: tf.nn.sigmoid(layer)
        layer = tf.nn.dropout(layer, keep_prob=keep_prob)
        # default to no dropout
    return(layer)


def make_weights_and_biases(N_nodes, N_layers):
    '''Returns dicts of weights and biases for use in the neural
        network with specified number of nodes per hidden layer
        and number of hidden layers.'''
    weights = {}
    biases = {}
    weights['w0'] = tf.Variable(tf.ones([1, N_nodes]),
                                name='weight_w0', trainable=True)
    # tf.Variable(1.0, name='weight_0')
    biases['b0'] = tf.Variable(tf.zeros([N_nodes]), trainable=True,
                               name='bias_b0')
    # tf.Variable(0.0,name='bias_0')
    for i in range(N_layers):
        weights['w{}'.format(i+1)] = tf.Variable(
            tf.ones([N_nodes, N_nodes]), name='weight_w{}'.format(i+1),
            trainable=True)  # tf.Variable(1.0, name='weight_0')
        biases['b{}'.format(i+1)] = tf.Variable(tf.zeros([N_nodes]),
                                                name='bias_b{}'.format(i+1),
                                                trainable=True)
    # tf.Variable(0.0,name='bias_0')
    weights['out'] = tf.Variable(tf.ones([N_nodes, 1]), name='weight_out',
                                 trainable=True)  # 1.0, name='weight_out')
    biases['out'] = tf.Variable(tf.zeros([1]), name='bias_out', trainable=True)
    # 0.0,name='bias_out')
    return(weights, biases)


def build_neural_network(x, weights, biases, keep_prob, N_nodes,
                         N_layers, activation_func=lambda x: x):
    '''Builds the TensorFlow graph with chosen weights and biases of the
        chosen width (N_nodes) and depth (N_layers)'''
    input_layer = perceptron_layer(x, weights['w0'], biases['b0'], keep_prob)
    hidden_layers = []
    for i in range(N_layers):
        hidden_layers.append(perceptron_layer(
            (input_layer if i == 0 else hidden_layers[i-1]),
            weights['w{}'.format(i+1)],
            biases['b{}'.format(i+1)], keep_prob))
    output_layer = perceptron_layer(
        (input_layer if N_hidden_layers == 0 else hidden_layers[-1]),
        weights=weights['out'],
        biases=biases['out'],
        final_layer=True)
    return(output_layer)


minval = -1.
maxval = 1.
NN = 63
N_hidden_nodes = 2
N_hidden_layers = 1


def make_train_graph(NN, N_hidden_nodes, N_hidden_layers):
    graph = hoomd.tensorflow_plugin.graph_builder(NN, output_forces=False)
    # get neighbor list
    nlist = graph.nlist[:, :, :3]
    # get the interatomic radii
    r = hoomd.tensorflow_plugin.graph_builder.safe_norm(nlist, axis=2)
    nn_r = tf.Variable(tf.zeros(shape=[64, NN]), trainable=False)
    nn_r.assign(r)
    histo3 = tf.summary.histogram('neighbor radius', nn_r)
    r_inv = hoomd.tensorflow_plugin.graph_builder.safe_div(1., r)
    print('r_inv shape: {}'.format(r_inv.shape))
    # make weights tensors, using our number of hidden nodes
    # NxNN out because we want pairwise forces
    r_inv = tf.reshape(r_inv, shape=[-1, 1], name='r_inv')

    weights, biases = make_weights_and_biases(N_hidden_nodes, N_hidden_layers)

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob', shape=[])
    # specify the network structure
    output_layer = build_neural_network(r_inv, weights, biases, keep_prob,
                                        N_hidden_nodes, N_hidden_layers,
                                        activation_func=tf.nn.relu)
    nn_energies = tf.reshape(output_layer, shape=[-1, NN])  # recover structure
    print('nn_energies shape: {}'.format(nn_energies.shape))
    # calculate the forces
    calculated_energies = tf.reduce_sum(nn_energies, axis=1,
                                        name='calculated_energies')
    print('calculated_energies shape: {}'.format(calculated_energies.shape))
    calculated_forces = graph.compute_forces(calculated_energies)
    # printer = tf.Print(calculated_forces, [calculated_forces],
    # summarize=10, message = 'calculated_forces is: ')
    # compare calculated forces to HOOMD's forces
    cost = tf.losses.mean_squared_error(graph.forces, calculated_forces)
    # need to minimize the cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    # print summaries for tensorboard
    tf.summary.scalar('cost', cost)
    histo = tf.summary.histogram('calculated forces', calculated_forces)
    histo2 = tf.summary.histogram('hoomd forces', graph.forces)
    # print cost for a more granular plot
    printer2 = tf.Print(cost, [cost], summarize=100, message="cost is: ")
    # check = tf.add_check_numerics_ops()
    graph.save(model_directory='/scratch/rbarret8/ann-training',
               out_nodes=[optimizer, histo, histo2, histo3, printer2])
    # check, printer,


def make_force_graph(NN, N_hidden_nodes, N_hidden_layers):
    graph = hoomd.tensorflow_plugin.graph_builder(NN)
    # get neighbor list
    nlist = graph.nlist[:, :, :3]
    # get the interatomic radii
    r = hoomd.tensorflow_plugin.graph_builder.safe_norm(nlist, axis=2)
    r_inv = hoomd.tensorflow_plugin.graph_builder.safe_div(1., r)
    r_inv = tf.reshape(r_inv, shape=[-1, 1])
    # make weights tensors, using our number of hidden nodes
    weights, biases = make_weights_and_biases(N_hidden_nodes, N_hidden_layers)
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob', shape=[])
    output_layer = build_neural_network(r_inv, weights, biases, keep_prob,
                                        N_hidden_nodes, N_hidden_layers,
                                        activation_func=tf.nn.relu)
    nn_energies = tf.reshape(output_layer, shape=[-1, NN])  # recover structure
    calculated_energies = tf.reduce_sum(nn_energies, axis=1,
                                        name='calculated_energies')
    # same forces as before
    calculated_forces = graph.compute_forces(calculated_energies)
    printer = tf.Print(calculated_forces, [calculated_forces], summarize=10,
                       message='calculated_forces is: ')
    # no cost nor minimizer this time
    graph.save(model_directory='/scratch/rbarret8/ann-inference',
               force_tensor=calculated_forces, out_nodes=[printer])


make_train_graph(NN, N_hidden_nodes, N_hidden_layers)


make_force_graph(NN, N_hidden_nodes, N_hidden_layers)
