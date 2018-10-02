import hoomd.tensorflow_plugin
import tensorflow as tf

def make_eds_graph(N, NN, directory, cv_op, set_point):
    '''Currently only computes running mean'''
    graph = hoomd.tensorflow_plugin.graph_builder(N, NN, output_forces=False)
    steps = tf.Variable(1.0, name='steps')
    mean = tf.Variable(0.0, name='mean')
    delta = (set_point - cv_op(graph)) / steps
    update_mean_op = mean.assign_add(delta)
    update_step_op = steps.assign_add(1.0)
    #the second arg means we need those two ops to be run before we can print
    print_op = tf.Print(update_mean_op, [mean], message='CV mean is')
    graph.save(model_directory=directory, out_nodes=[update_step_op, update_mean_op, print_op])

def avg_r(graph):
    '''returns average distance from center'''
    rs = graph.safe_norm(graph.positions[:,:3])
    return tf.reduce_mean(rs)

make_eds_graph(64, 16, '/tmp/eds', avg_r, 0)