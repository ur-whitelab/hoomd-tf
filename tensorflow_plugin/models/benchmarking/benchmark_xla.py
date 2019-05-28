import hoomd.tensorflow_plugin
from hoomd.tensorflow_plugin import tfcompute
import hoomd
import hoomd.md
import hoomd.dump
import hoomd.group
import hoomd.benchmark
from math import sqrt
import numpy as np
from sys import argv as argv
import tensorflow as tf


if(len(argv) != 3):
    print('Usage: benchmark_xla.py [N_PARTICLES] [USE_XLA (as int)]')
    exit(0)

N = int(argv[1])
use_xla=bool(int(argv[2]))
model_dir = '/scratch/rbarret8/benchmarking_{}_xla'.format(use_xla)

# make a simple ANN
# based on: https://medium.com/@curiousily/
# tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b

def perceptron_layer(x,
                     weights,
                     biases,
                     keep_prob=1.0,
                     final_layer=False,
                     activation_function=lambda x: x):
    '''Creates a single neural network layer given weights and bias tensors.
        Final layers should not have activation functions, must specify.
        Defaults to no dropuout and linear activation.'''
    layer = tf.matmul(x, weights) + biases
    if not final_layer:
        layer = activation_function(layer)                   #example: tf.nn.sigmoid(layer)
        layer = tf.nn.dropout(layer, keep_prob=keep_prob)  #default to no dropout
    return(layer)


def make_weights_and_biases(N_nodes, N_layers):
    '''Returns dicts of weights and biases for use in the neural network with specified 
        number of nodes per hidden layer and number of hidden layers.'''
    weights = {}
    biases = {}
    weights['w0']= tf.Variable(tf.ones([1, N_nodes]), name='weight_w0', trainable=True)
    biases['b0']= tf.Variable(tf.zeros([N_nodes]), name='bias_b0', trainable=True)
    for i in range(N_layers):
        weights['w{}'.format(i+1)]= tf.Variable(tf.ones([N_nodes, N_nodes]),
                                                name='weight_w{}'.format(i+1),
                                                trainable=True)
        biases['b{}'.format(i+1)]= tf.Variable(tf.zeros([N_nodes]),
                                               name='bias_b{}'.format(i+1),
                                               trainable=True)
    weights['out']= tf.Variable(tf.ones([N_nodes, 1]),
                                name='weight_out',
                                trainable=True)
    biases['out']= tf.Variable(tf.zeros([1]),
                               name='bias_out',
                               trainable=True)
    return(weights, biases)

def build_neural_network(x,
                         weights,
                         biases,
                         keep_prob,
                         N_nodes,
                         N_layers,
                         activation_func=lambda x: x):
    '''Builds the TensorFlow graph with chosen weights and biases of the
        chosen width (N_nodes) and depth (N_layers)'''
    input_layer = perceptron_layer(x, weights['w0'], biases['b0'], keep_prob)
    hidden_layers = []
    for i in range(N_layers):
        hidden_layers.append(
            perceptron_layer(
                (input_layer if i==0 else hidden_layers[i-1]),
                weights['w{}'.format(i+1)],
                biases['b{}'.format(i+1)],
                keep_prob))
    output_layer = perceptron_layer(
        (input_layer if N_hidden_layers==0 else hidden_layers[-1]),
        weights['out'],
        biases['out'],
        final_layer=True)
    return(output_layer)

minval = -1.
maxval = 1.

NN = 63
N_hidden_nodes = 50
N_hidden_layers = 5

def make_train_graph(NN, N_hidden_nodes, N_hidden_layers):
    graph = hoomd.tensorflow_plugin.graph_builder(NN, output_forces=False)
    # get neighbor list
    nlist = graph.nlist[:,:,:3]
    # get the interatomic radii
    r = hoomd.tensorflow_plugin.graph_builder.safe_norm(nlist, axis=2)
    nn_r = tf.Variable(tf.zeros(shape=[64, NN]), trainable=False)
    nn_r.assign(r)
    r_inv = hoomd.tensorflow_plugin.graph_builder.safe_div(1.,r)
    # make weights tensors, using our number of hidden nodes
    # NxNN out because we want pairwise forces
    r_inv = tf.reshape(r_inv, shape=[-1,1], name='r_inv')
    weights, biases = make_weights_and_biases(N_hidden_nodes, N_hidden_layers)

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob', shape=[])
    # specify the network structure
    output_layer = build_neural_network(r_inv,
                                        weights,
                                        biases,
                                        keep_prob,
                                        N_hidden_nodes,
                                        N_hidden_layers,
                                        activation_func = tf.nn.relu)
    # recover structure
    nn_energies = tf.reshape(output_layer, shape=[-1, NN])
    # calculate the forces
    calculated_energies = tf.reduce_sum(nn_energies, axis=1, name='calculated_energies')
    calculated_forces = graph.compute_forces(calculated_energies)
    # compare calculated forces to HOOMD's forces
    cost = tf.losses.mean_squared_error(graph.forces, calculated_forces)
    # need to minimize the cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    graph.save(model_directory=model_dir, out_nodes=[optimizer]), 

make_train_graph(NN, N_hidden_nodes, N_hidden_layers)

np.random.seed(42)

with hoomd.tensorflow_plugin.tfcompute(model_dir,
                                       _mock_mode=False,
                                       write_tensorboard=False,
                                       use_xla=use_xla) as tfcompute:
    hoomd.context.initialize('--mode=gpu')
    rcut = 3.0
    sqrt_N = int(sqrt(N))
    
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[sqrt_N, sqrt_N])
    nlist = hoomd.md.nlist.cell(check_period = 1)
    lj = hoomd.md.pair.lj(rcut, nlist)# basic LJ forces from HOOMD
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.0, seed=42)
    # equilibrate for 4k steps first
    hoomd.run(4000)
    # now attach the trainable model
    tfcompute.attach(nlist,
                     r_cut=rcut,
                     save_period=100,
                     period=100,
                     feed_dict=dict({'keep_prob:0': 0.8}))
    # train on 5k timesteps
    hoomd.run(50000)#, profile=True)
    # train on 5k timesteps and benchmark with 20 repeats
    benchmark_results = hoomd.benchmark.series(warmup=6000,
                                               repeat=5,
                                               steps=50000,
                                               limit_hours=2)
    
with open('{}-particles_{}_xla_time.txt'.format(N, use_xla), 'w+') as f:
    f.write('Elapsed time with {} particles: {}'.format(N,str(benchmark_results)))
