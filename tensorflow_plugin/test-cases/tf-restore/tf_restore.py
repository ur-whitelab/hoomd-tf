import tensorflow as tf
import os, tempfile, multiprocessing

N = 8
x = tf.Variable(tf.ones([N, 4]))
w = tf.Variable(tf.ones([N, 4]))
y = tf.multiply(x, w)
z = tf.reshape(y, [-1, 4])

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    save_loc = tempfile.mkdtemp()
    saver.save(sess, os.path.join(save_loc, 'model'))

def load(save_loc):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(os.path.join(save_loc,'model.meta'))
        new_saver.restore(sess, tf.train.latest_checkpoint(save_loc))
    print('load successful')

#test same process
load(save_loc)
#test different thread
p = multiprocessing.Process(target=load, args=('save_loc',))
p.start()

print('started, now waiting')
p.join()