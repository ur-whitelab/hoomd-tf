import hoomd.tensorflow_plugin as htf
import sys
import tensorflow as tf


out_aver=open('output_aver_r_running_mean.txt', 'a')
def make_eds_graph(N, NN, directory, cv_op, set_point):
    '''Currently only computes running mean'''

    graph =htf.graph_builder(N,output_forces=True)

    A=tf.constant(0.30,name='A',dtype=tf.float32)
    
    bias_energy=tf.Variable(tf.ones(N,dtype=tf.float32), dtype=tf.float32,name='energy')
    steps = tf.Variable(1.0, name='steps')#,dtype=tf.float64)
    mean = tf.Variable(0.1, name='mean')#,dtype=tf.float64)
    variance=tf.Variable(0.1, name='variance')#,dtype=tf.float64)
    gradient=tf.Variable(0.1,name='gradient')#,dtype=tf.float64)
    alpha=tf.Variable(1.0,name='alpha')#,dtype=tf.float64)
    bias_val=tf.Variable(1.0,name='bias-val')#,expected_shape=(1),dtype=tf.float64)
    set_point=tf.Variable(set_point,name='set_point')#,dtype=tf.float64)
    #sess = tf.Session()
    #init = tf.global_variables_initializer()
    #sess.run(init)
    #aver_r=tf.Variable(1.0,name='aver_r')
    colvar=tf.Variable(1.0, name='cv0')# col var
    
    aver_r,cv,print_op_r=cv_op(graph,N)#,a,a1,a2,bccc
    #aver_r.assign(aver_r1)
    run_cv=graph.running_mean(mean,name='cv')
    delta = graph.safe_div((aver_r-run_cv), (steps+1.0))
    delta2=aver_r-(mean+delta)
    variance2=delta*delta2
    varience=graph.running_mean(cv,name='cv2')-(run_cv**2)
    gradient_new=-2*(graph.safe_div(run_cv,set_point)-1)*(variance)
    norm_gradient=(gradient_new**2+gradient**2)**0.5
    
    alpha_val=alpha-graph.safe_div((gradient_new*A),norm_gradient)
    print_op2=tf.print('alpha',alpha_val,output_stream=sys.stdout)
    print_op3=tf.print('aver_r after reduce mean',aver_r,output_stream=sys.stdout)

    b=(tf.reduce_sum((graph.safe_div((alpha_val*aver_r),set_point))))
    #result = sess.run(b)
    #print(result, 'b')
    update_mean_op = mean.assign_add(delta)
    update_variance_op=variance.assign_add(variance2)
    update_step_op = steps.assign_add(1.0)
    update_norm_gradient=gradient.assign(norm_gradient)
    update_alpha_op=alpha.assign(alpha_val)
    update_cv_op=colvar.assign(cv)
    bias_energy2=(bias_energy*b)#ias_val)
    #print(sess.run(bias_energy2),'bias_en2')
    forces = graph.compute_forces(bias_energy2)
    print_op=tf.print('bias energy is',bias_energy,output_stream=sys.stdout)
    print_op_mean= tf.print('CV mean is', mean,output_stream=sys.stdout)
    #print_op_gradient=tf.Print(update_norm_gradient,[gradient], message='Norm of the Gradient is')
    #print_op_alpha=tf.Print(update_alpha_op,[alpha], message='Coupling Constant is')
    
    #print_op_var=tf.Print(update_variance_op,[variance], message='Var is')
    #print_op_2= tf.Print(update_alpha_op, [alpha], message='alpha  is')
    graph.save(model_directory=directory, out_nodes=[update_step_op,update_mean_op,update_variance_op,print_op2,print_op3,update_norm_gradient,update_alpha_op,print_op_mean,print_op_r,update_cv_op],force_tensor=forces[:,:2],virial=None)
    
def avg_r(graph,N):
    '''returns average distance from center of mass'''
    

    # positions=tf.cast(positions1,tf.float64)
    com=tf.reduce_mean(graph.positions[:,:2],0)#,name='com')#,dtype=tf.float64)
    #com=tf.scalar_mul(1/N,total_position)
    print_op=tf.print('center_of_mass is',com,output_stream=sys.stdout)
    print_op2=tf.print('subtract',tf.math.subtract(graph.positions[:,:2],tf.reshape(tf.tile(com,[N]),(N,2))),output_stream=sys.stdout)
    
    aaaa=tf.print('tile',tf.tile(com,[N]),output_stream=sys.stdout)
    aaa=tf.print('com in long term',tf.reshape(tf.tile(com,[N]),(N,2)),output_stream=sys.stdout)
    rs= graph.safe_norm(tf.math.subtract(graph.positions[:,:2],com))#,tf.reshape(tf.tile(com,[N]),(N,2))),axis=1)


    a=tf.print('rs not reduce mean',rs,output_stream=sys.stdout)
    #out_aver.write('{}\n'.format(rs))
    real=tf.reduce_mean(rs)
    #graph.running_mean(graph.positions[:,:2], 'com')#
    return real,real**2,print_op#,a,print_op2,aaa,aaaa

    

make_eds_graph(64, 30, 'manuscrpt/eds', avg_r,5.8)
out_aver.close()
