import tensorflow as tf

x = tf.Variable(tf.random_uniform([32, 4], name='input'))
w = tf.Variable(tf.random_uniform([4, 32]))
y = tf.matmul(x, w)
z = tf.identity(y, name='output')

with open('graph.pb', 'wb') as f:
  f.write(y.graph.as_graph_def().SerializeToString())