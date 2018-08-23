import tensorflow as tf

x = tf.Variable(tf.random_uniform([32, 4], name='input'))
w = tf.Variable(tf.random_uniform([32, 4]), name='random')
y = tf.multiply(x, w)
z = tf.reshape(y, [-1, 4], name='output')

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'model/test')