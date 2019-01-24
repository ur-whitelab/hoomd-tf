import hoomd
import tensorflow as tf
import numpy as np
import hoomd.tensorflow_plugin
from sys import argv as argv



#make a simple ANN, c/o: https://medium.com/@curiousily/tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b
def multilayer_perceptron_layer_biased(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights), biases)#tf.matmul(x, weights)
    layer_1 = tf.nn.relu(layer_1)#sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    return layer_1

def multilayer_perceptron_end_layer_biased(x, weights, biases):
    out_layer = tf.add(tf.matmul(x, weights), biases)
    return out_layer

def multilayer_perceptron_layer_unbiased(x, weights, biases, keep_prob):
    layer_1 = tf.matmul(x, weights)#tf.matmul(x, weights)
    layer_1 = tf.nn.relu(layer_1)#sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    return layer_1

def multilayer_perceptron_end_layer_unbiased(x, weights, biases):
    out_layer = tf.matmul(x, weights)
    return out_layer


NN = 63
N_hidden = 6
def make_train_graph(NN, N_hidden):
    graph = hoomd.tensorflow_plugin.graph_builder(NN, output_forces=False)
    
    #get neighbor list
    nlist = graph.nlist[:,:,:3]
    #get the interatomic radii
    r = hoomd.tensorflow_plugin.graph_builder.safe_norm(nlist, axis=2)
    r_inv = hoomd.tensorflow_plugin.graph_builder.safe_div(1.,r)
    print('r_inv shape: {}'.format(r_inv.shape))
    #make weights tensors, using our number of hidden nodes
    #NxNN out because we want pairwise forces
    r_inv = tf.reshape(r_inv, shape=[-1,1])
    weights = {}
    weights['h1']= tf.Variable(tf.random_uniform([1, N_hidden], minval=0., maxval=0.01), name='weight_h1')
    weights['h2']= tf.Variable(tf.random_uniform([N_hidden, N_hidden], minval=0., maxval=0.01), name='weight_h2')
    weights['h3']= tf.Variable(tf.random_uniform([N_hidden, N_hidden], minval=0., maxval=0.01), name='weight_h2')
    weights['out']= tf.Variable(tf.random_uniform([N_hidden, 1], minval=0., maxval=0.05), name='weight_out')

    biases = {}
    biases['b1']= tf.Variable(tf.random_uniform([N_hidden], minval=0., maxval=0.05), name='bias_b1')
    biases['b2']= tf.Variable(tf.random_uniform([N_hidden], minval=0., maxval=0.05), name='bias_b2')
    biases['b3']= tf.Variable(tf.random_uniform([N_hidden], minval=0., maxval=0.05), name='bias_b3')
    biases['out']= tf.Variable(tf.random_uniform([1], minval=0., maxval=0.05), name='bias_out')

    keep_prob = tf.Variable(1.0, trainable=False, name='keep_prob')
    #specify the network structure
    
    input_layer = multilayer_perceptron_layer_biased(r_inv, weights['h1'], biases['b1'], keep_prob)
    first_hidden_layer = multilayer_perceptron_layer_biased(input_layer, weights['h2'], biases['b2'], keep_prob)
    second_hidden_layer = multilayer_perceptron_layer_biased(first_hidden_layer, weights['h3'], biases['b3'], keep_prob)
    output_layer = multilayer_perceptron_end_layer_biased(second_hidden_layer, weights['out'], biases['out'])
    print('input layer shape: {}'.format(input_layer.shape))
    print('output layer shape: {}'.format(output_layer.shape))
    nn_energies = tf.reshape(output_layer, shape=[-1, NN])#recover structure
    print('nn_energies shape: {}'.format(nn_energies.shape))
    #calculate the forces
    calculated_energies = tf.reduce_sum(nn_energies, axis=1)
    print('calculated_energies shape: {}'.format(calculated_energies.shape))
    calculated_forces = graph.compute_forces(calculated_energies)
    printer = tf.Print(calculated_forces, [calculated_forces], summarize=10, message = 'calculated_forces is: ')
    #compare calculated forces to HOOMD's forces
    cost = tf.losses.mean_squared_error(graph.forces, calculated_forces)
    #need to minimize the cost
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    #print summaries for tensorboard
    tf.summary.scalar('cost', cost)
    histo = tf.summary.histogram('calculated forces', calculated_forces)
    histo2 = tf.summary.histogram('hoomd forces', graph.forces)
    #print cost for a more granular plot
    printer2 = tf.Print(cost, [cost], summarize=100, message = "cost is: ")
    #check = tf.add_check_numerics_ops()
    
    graph.save(model_directory='/tmp/ann-training', out_nodes=[optimizer, histo, histo2, printer, printer2])#optimizer, check, printer, 

def make_force_graph(NN, N_hidden):
    graph = hoomd.tensorflow_plugin.graph_builder(NN)
    #get neighbor list
    nlist = graph.nlist[:,:,:3]
    #get the interatomic radii
    r = hoomd.tensorflow_plugin.graph_builder.safe_norm(nlist, axis=2)
    r_inv = hoomd.tensorflow_plugin.graph_builder.safe_div(1.,r)
    r_inv = tf.reshape(r_inv, shape=[-1,1])
    #make weights tensors, using our number of hidden nodes
    weights = {}
    weights['h1']= tf.Variable(tf.random_uniform([1, N_hidden], minval=0., maxval=0.01), name='weight_h1')
    weights['h2']= tf.Variable(tf.random_uniform([N_hidden, N_hidden], minval=0., maxval=0.01), name='weight_h2')
    weights['h3']= tf.Variable(tf.random_uniform([N_hidden, N_hidden], minval=0., maxval=0.01), name='weight_h3')
    weights['out']= tf.Variable(tf.random_uniform([N_hidden, 1], minval=0., maxval=0.01), name='weight_out')

    biases = {}
    biases['b1']= tf.Variable(tf.random_uniform([N_hidden], minval=0., maxval=0.05), name='bias_b1')
    biases['b2']= tf.Variable(tf.random_uniform([N_hidden], minval=0., maxval=0.05), name='bias_b2')
    biases['b3']= tf.Variable(tf.random_uniform([N_hidden], minval=0., maxval=0.05), name='bias_b3')
    biases['out']= tf.Variable(tf.random_uniform([1], minval=0., maxval=0.05), name='bias_out')

    keep_prob = tf.Variable(1.0, name='keep_prob', trainable=False)
    
    input_layer = multilayer_perceptron_layer_biased(r_inv, weights['h1'], biases['b1'], keep_prob)
    first_hidden_layer = multilayer_perceptron_layer_biased(input_layer, weights['h2'], biases['b2'], keep_prob)
    second_hidden_layer = multilayer_perceptron_layer_biased(first_hidden_layer, weights['h3'], biases['b3'], keep_prob)
    output_layer = multilayer_perceptron_end_layer_biased(first_hidden_layer, weights['out'],biases['out'])
    nn_energies = tf.reshape(output_layer, shape=[-1, NN])#recover structure
    calculated_energies = tf.reduce_sum(nn_energies, axis=1)
    #same forces as before
    calculated_forces = graph.compute_forces(calculated_energies)
    printer = tf.Print(calculated_forces, [calculated_forces], summarize=10, message = 'calculated_forces is: ')
    #no cost nor minimizer this time
    graph.save(model_directory='/tmp/ann-inference', force_tensor=calculated_forces, out_nodes=[printer])

make_train_graph(NN, N_hidden)
make_force_graph(NN, N_hidden)
