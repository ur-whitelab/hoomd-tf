import hoomd.tensorflow_plugin as htf
import sys
import tensorflow as tf


out_aver=open('output_aver_r_running_mean.txt', 'a')
def make_eds_graph(N, NN, directory, cv_op, set_point):
    '''Currently only computes running mean'''

    graph =htf.graph_builder(N-1,output_forces=True)

    A=tf.constant(0.3,name='A',dtype=tf.float32)
    kt=1.0
    beta=1/kt
    bias_energy=tf.Variable(tf.ones(N,dtype=tf.float32), dtype=tf.float32,name='energy')
    steps = tf.Variable(1.0, name='steps')#,dtype=tf.float64)
    mean = tf.Variable(0.1, name='mean')#,dtype=tf.float64)
    #variance=tf.Variable(0.1, name='variance')#,dtype=tf.float64)
    gradient=tf.Variable(0.1,name='gradient')#,dtype=tf.float64)
    alpha=tf.Variable(0.001,name='alpha')#,dtype=tf.float64)
    bias_val=tf.Variable(1.0,name='bias-val')#,expected_shape=(1),dtype=tf.float64)
    set_point=tf.Variable(set_point,name='set_point')#,dtype=tf.float64)
    colvar=tf.Variable(0.0, name='colvar_inst')          #col var for outputing the instantenous cv
    colvar_sq=tf.Variable(0.0, name='cv0')          #col var for outputing the instantenous cv^2
    aver_r,cv_sq=cv_op(graph,N)          #this outputs the mean cv, cv^2 and a print statement
    update_colvar_op = colvar.assign(aver_r)
    update_cv_op=colvar_sq.assign(cv_sq)
    run_cv=graph.running_mean(colvar,name='mean_cv')    #takes the running mean of the cv
    variance=graph.running_mean(colvar_sq,name='cv2')-(run_cv**2)        #variance of the collective variable
    gradient_new=-2*beta*(graph.safe_div(run_cv,set_point)-1)*(variance)  #g_tau (gradient at time tau) in the EDS paper aka eq 5
    norm_gradient=(gradient_new**2+gradient**2)**0.5                 #sq root of the gradients squared aka A/learning_rate
    learning_rate=graph.safe_div(A,norm_gradient)                    #learning rate
    alpha_val=alpha-learning_rate*gradient_new                       #new coupling constant
    print_op2=tf.print('alpha',alpha_val,output_stream=sys.stdout)   
    print_op3=tf.print('aver_r after reduce mean',aver_r,output_stream=sys.stdout)
    bias_potential=tf.reduce_sum((graph.safe_div((alpha_val*aver_r),set_point)))
    update_step_op = steps.assign_add(1.0)
    update_norm_gradient=gradient.assign(gradient_new)
    update_alpha_op=alpha.assign(alpha_val)
    update_mean_op=mean.assign(run_cv)
    bias_energy2=bias_energy*bias_potential #expanding the bais potential to be for for n particles
    forces = graph.compute_forces(bias_energy2) #computing the forces from the bias energy
    print_op=tf.print('bias energy is',bias_energy2,output_stream=sys.stdout)
    print_op_mean= tf.print('CV mean is', mean,output_stream=sys.stdout)
    graph.save(model_directory=directory, out_nodes=[update_step_op,update_colvar_op,update_cv_op,print_op2,update_mean_op,print_op3,update_norm_gradient,update_alpha_op,print_op_mean],force_tensor=forces,virial=None)
    
def avg_r(graph,N):
    '''returns average distance from center of mass'''
    
    com=tf.reduce_mean(graph.positions[:,:2],0)
    rs= graph.safe_norm(tf.math.subtract(graph.positions[:,:2],com),axis=1)
    real=tf.reduce_mean(rs)
    return real,real**2
    

make_eds_graph(64, 30, 'manuscrpt/eds', avg_r,5.8)
out_aver.close()
