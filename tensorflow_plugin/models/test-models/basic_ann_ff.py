import hoomd
import tensorflow as tf
import numpy as np
import hoomd.tensorflow_plugin
from sys import argv as argv



#make a simple ANN, c/o: https://medium.com/@curiousily/tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b
def multilayer_perceptron_layer(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights), biases)
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    return layer_1

def multilayer_perceptron_end_layer(x, weights, biases):
    out_layer = tf.matmul(x, weights) + biases
    return out_layer


NN = 63
N_hidden = NN
def make_train_graph(NN, N_hidden):
    graph = hoomd.tensorflow_plugin.graph_builder(NN, output_forces=False)
    
    #get neighbor list
    nlist = graph.nlist[:,:,:3]
    #get the interatomic radii
    r = hoomd.tensorflow_plugin.graph_builder.safe_norm(nlist, axis=2)
    r_inv = hoomd.tensorflow_plugin.graph_builder.safe_div(1.,r)
    #make weights tensors, using our number of hidden nodes
    #NxNN out because we want pairwise forces
    weights = {}
    weights['h1']= tf.Variable(tf.random_normal([NN, N_hidden]), name='weight_h1')
    weights['h2']= tf.Variable(tf.random_normal([NN, N_hidden]), name='weight_h2')
    weights['out']= tf.Variable(tf.random_normal([N_hidden, NN]), name='weight_out')
    
    #biases must be same dimensionality as we are applying them to
    biases = {}
    biases['b1']= tf.Variable(tf.random_normal([N_hidden]), name='bias_b1')
    biases['b2']= tf.Variable(tf.random_normal([N_hidden]), name='bias_b2')
    biases['out']= tf.Variable(tf.random_normal([NN]), name='bias_out')
    keep_prob = tf.Variable(0.9, name='keep_prob')
    #specify the network structure
    input_layer = multilayer_perceptron_layer(r_inv, weights['h1'], biases['b1'], keep_prob)
    first_hidden_layer = multilayer_perceptron_layer(input_layer, weights['h2'], biases['b2'], keep_prob)
    output_layer = multilayer_perceptron_end_layer(first_hidden_layer, weights['out'], biases['out'])
    #calculate the forces
    calculated_energies = tf.reduce_sum(output_layer, axis=1)
    calculated_forces = graph.compute_forces(calculated_energies)
    #compare calculated forces to HOOMD's forces
    cost = tf.losses.mean_squared_error(graph.forces, calculated_forces)
    #need to minimize the cost
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    #print summaries for tensorboard
    tf.summary.scalar('cost', cost)
    histo = tf.summary.histogram('calculated forces', calculated_forces)
    histo2 = tf.summary.histogram('hoomd forces', graph.forces)
    #print cost for a more granular plot
    #printer = tf.Print(cost, [cost], summarize=100, message = "cost is: ")
    check = tf.add_check_numerics_ops()
    
    graph.save(model_directory='/tmp/ann-training', out_nodes=[check, optimizer, histo, histo2])#printer, 

def make_force_graph(NN, N_hidden):
    graph = hoomd.tensorflow_plugin.graph_builder(NN)
    #get neighbor list
    nlist = graph.nlist[:,:,:3]
    #get the interatomic radii
    r = hoomd.tensorflow_plugin.graph_builder.safe_norm(nlist, axis=2)
    r_inv = hoomd.tensorflow_plugin.graph_builder.safe_div(1.,r)
    #make weights tensors, using our number of hidden nodes
    #NxNN out because we want pairwise forces
    weights = {}
    weights['h1']= tf.Variable(tf.random_normal([NN, N_hidden]), name='weight_h1')
    weights['h2']= tf.Variable(tf.random_normal([NN, N_hidden]), name='weight_h2')
    weights['out']= tf.Variable(tf.random_normal([N_hidden, NN]), name='weight_out')
    
    #biases must be same dimensionality as we are applying them to
    biases = {}
    biases['b1']= tf.Variable(tf.random_normal([N_hidden]), name='bias_b1')
    biases['b2']= tf.Variable(tf.random_normal([N_hidden]), name='bias_b2')
    biases['out']= tf.Variable(tf.random_normal([NN]), name='bias_out')
    
    
    keep_prob = tf.Variable(1.0, name='keep_prob')
    
    input_layer = multilayer_perceptron_layer(r_inv, weights['h1'], biases['b1'], keep_prob)
    first_hidden_layer = multilayer_perceptron_layer(input_layer, weights['h2'], biases['b2'], keep_prob)
    output_layer = multilayer_perceptron_end_layer(first_hidden_layer, weights['out'], biases['out'])
    calculated_energies = tf.reduce_sum(output_layer, axis=1)
    #same forces as before
    calculated_forces = graph.compute_forces(calculated_energies)
    #no cost nor minimizer this time
    graph.save(model_directory='/tmp/ann-inference', force_tensor=calculated_forces)

make_train_graph(NN, N_hidden)
make_force_graph(NN, N_hidden)
