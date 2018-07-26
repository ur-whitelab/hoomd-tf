import tensorflow as tf
import os


N = 3 * 3
NN = N - 1
save_loc = '/tmp'

nlist = tf.ones ([N, NN, 4], name='nlist')
_ = tf.ones([N, 4], name='positions')

with tf.name_scope('force-calc') as scope:
    neighs_rs = tf.norm(nlist, axis=1, keepdims=True)
    #no need to use netwon's law because nlist should be double counted
    fr = tf.multiply(-1.0, tf.multiply(tf.reciprocal(neighs_rs), nlist), name='nan-pairwise-forces')
    with tf.name_scope('remove-nans') as scope:
        zeros = tf.zeros(tf.shape(nlist))
        real_fr = tf.where(tf.is_nan(fr), zeros, fr, name='pairwise-forces')
forces = tf.reduce_sum(real_fr, axis=1, name='forces')
#need to make variable so we have something to save.
#confusing, I know
tf.Variable(forces, name='force-save')

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(save_loc, 'model'))