import hoomd.tensorflow_plugin
import tensorflow as tf

def make_eds_graph(N, NN, directory, cv_op, set_point):
    '''Currently only computes running mean'''
    graph = hoomd.tensorflow_plugin.graph_builder(N, NN, output_forces=False)
    A=tf.constant(3.0,name='const')
    #energy=tf.Variable(0.0, expected_shape=(N,4),name='energy')
    steps = tf.Variable(1.0, name='steps')
    mean = tf.Variable(0.0, name='mean')
    variance=tf.Variable(0.0, name='variance')
    gradient=tf.Variable(0.1,name='gradient')
    alpha=tf.Variable(0.0,name='alpha')
    A=3#3kbt
    aver_r,print_op_r=cv_op(graph,N)
    delta = (aver_r-mean) / (steps+1)
    delta2=aver_r-(mean+delta)
    variance2=delta*delta2
    gradient_new=-2*(mean/set_point-1)*(variance)
    norm_gradient=(gradient_new**2+gradient**2)**0.5
    alpha_val=alpha-gradient_new*A/norm_gradient
    #energy[:,3].assign((alpha_val*aver_r/set_point))
    #foce=graph.forces[:, :, :3]
    #print(energy.shape)
    #energy = tf.reduce_sum(bias_energy, axis=1)
    #bias_force=alpha_val/set_point
    #force=force+bias_force
    update_mean_op = mean.assign_add(delta)
    update_variance_op=variance.assign_add(variance2)
    update_step_op = steps.assign_add(1.0)
    update_norm_gradient=gradient.assign(norm_gradient)
    update_alpha_op=alpha.assign(alpha_val)
    
    #forces = graph.compute_forces(energy)
    #print(forces.shape[0])
    
    #print_op=tf.Print(energy,[energy],'energy')
    #print_op = tf.Print(update_mean_op, [mean], message='CV mean is')
    #print_op_gradient=tf.Print(update_norm_gradient,[gradient], message='Norm of the Gradient is')
    #print_op_alpha=tf.Print(update_alpha_op,[alpha], message='Coupling Constant is')
    
    #print_op_var=tf.Print(update_variance_op,[variance], message='Var is')
    #print_op_2= tf.Print(update_alpha_op, [alpha], message='alpha  is')
    graph.save(model_directory=directory, out_nodes=[update_step_op, update_mean_op,update_variance_op,update_norm_gradient,update_alpha_op])#,force_tensor=forces,virial=None)
    
def avg_r(graph,N):
    '''returns average distance from center of mass'''
    
   
    com=tf.reduce_mean(graph.positions[:,:3],0)
    #com=tf.scalar_mul(1/N,total_position)
    print_op=tf.Print(com,[com],message='center_of_mass is')
    rs = graph.safe_norm(tf.subtract(graph.positions[:,:3],com))
    return tf.reduce_mean(rs),print_op

    

make_eds_graph(64, 16, '/tmp/eds', avg_r,6)
