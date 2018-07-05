import tensorflow as tf

#assumes nlist to be nearest

x = tf.Variable(tf.random_uniform([32, 4], name='nlist:0'))
w = tf.Variable(tf.random_uniform([32, 4]), name='positions:0')
y = tf.multiply(x, w)
z = tf.reshape(y, [-1, 4], name='forces:0')

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'model')