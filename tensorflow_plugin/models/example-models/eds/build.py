import hoomd.tensorflow_plugin as htf
import sys
import tensorflow as tf


def make_eds_graph(N, NN, directory, cv_op, set_point):
    '''Currently only computes running mean'''

    graph = htf.graph_builder(N-1, output_forces=True)
    A = tf.constant(0.3, name='A', dtype=tf.float32)
    kt = 1.0  # should match the kt from the nvt ensemble in hoomd
    beta = 1/kt
    bias_energy = tf.Variable(tf.ones(N, dtype=tf.float32),
                              dtype=tf.float32, name='energy')
    steps = tf.Variable(1.0, name='steps')
    mean = tf.Variable(0.1, name='mean')
    gradient = tf.Variable(0.1, name='gradient')
    # gradient for Gradient Descend algorithm
    alpha = tf.Variable(0.001, name='alpha')  # coupling constant
    set_point = tf.Variable(set_point, name='set_point')
    # reference value or the bias reference
    colvar = tf.Variable(0.0, name='colvar_inst')
    # col var for outputing the instantenous cv
    colvar_sq = tf.Variable(0.0, name='cv0')
    # col var for outputing the instantenous cv^2
    aver_r, cv_sq = cv_op(graph, N)
    # this outputs the mean cv, cv^2 and a print
    update_colvar_op = colvar.assign(aver_r)
    # appending the new cv to the colvar variable
    update_cv_op = colvar_sq.assign(cv_sq)
    # appedning the new cv^2 to the colvar_sq variable
    run_cv = graph.running_mean(colvar, name='mean_cv')
    # takes the running mean of the cv
    variance = graph.running_mean(colvar_sq, name='cv2')-(run_cv**2)
    # variance of the collective variable
    gradient_new = -2*beta*(graph.safe_div(run_cv, set_point)-1)*(variance)
    # g_tau (gradient at time tau) in the EDS paper aka eq 5
    norm_gradient = (gradient_new**2+gradient**2)**0.5
    # sq root of the gradients squared aka A/learning_rate
    learning_rate = graph.safe_div(A, norm_gradient)  # learning rate
    alpha_val = alpha-learning_rate*gradient_new
    # new coupling constant
    bias_potential = tf.reduce_sum((graph.safe_div((alpha_val*aver_r),
                                                   set_point)))
    # computing the potential energy due to the bias
    update_step_op = steps.assign_add(1.0)
    update_norm_gradient = gradient.assign(gradient_new)
    update_alpha_op = alpha.assign(alpha_val)
    update_mean_op = mean.assign(run_cv)
    bias_energy2 = bias_energy*bias_potential
    # expanding the bais potential to be for for n particles
    forces = graph.compute_forces(bias_energy2)
    # computing the forces from the bias energy
    graph.save(model_directory=directory, out_nodes=[
            update_step_op, update_colvar_op, update_cv_op, update_mean_op,
            print_op3, update_norm_gradient, update_alpha_op],
               force_tensor=forces, virial=None)


def avg_r(graph, N):
    '''returns average distance from center of mass'''
    com = tf.reduce_mean(graph.positions[:, :2], 0)
    rs = graph.safe_norm(tf.math.subtract(graph.positions[:, :2], com), axis=1)
    real_cv = tf.reduce_mean(rs)
    return real_cv, real_cv**2


make_eds_graph(64, 30, '/tmp/eds', avg_r, 5.8)
