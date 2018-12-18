import tensorflow as tf
import numpy as np
from ipc_tester import IpcTester

ipct = IpcTester(32)
hoomd_to_tf_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/hoomd2tf/lib_hoomd2tf_op.so')
hoomd_to_tf = hoomd_to_tf_module.hoomd_to_tf
tf_to_hoomd_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/tf2hoomd/lib_tf2hoomd_op.so')
tf_to_hoomd = tf_to_hoomd_module.tf_to_hoomd

graph_input = hoomd_to_tf(address=ipct.get_input_buffer(), size=32, T=np.float32)

print('prior', ipct.get_output_array())
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('model/test.meta', input_map={'input:0': graph_input})
with tf.Session() as sess:
  new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
  graph = tf.get_default_graph()

  print([n.name for n in tf.get_default_graph().as_graph_def().node])
  out = tf.get_default_graph().get_tensor_by_name('output:0')
  graph_output = tf_to_hoomd(out, address=ipct.get_output_buffer(), size=32)

  #print(ipct.get_input_array())
  print('about to print out')
  p = tf.Print(out, [out, graph_input])
  sess.run(graph_output)
print('post', ipct.get_output_array())