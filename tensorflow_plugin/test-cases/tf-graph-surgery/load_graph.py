import tensorflow as tf
import numpy as np
from ipc_tester import IpcTester

ipct = IpcTester(32)
ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor
tensor_to_ipc_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/tensor2ipc/lib_tensor2ipc_op.so')
tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc

graph_input = ipc_to_tensor(address=ipct.get_input_buffer(), size=32, T=np.float32)

print('prior', ipct.get_output_array())
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('model/test.meta', input_map={'input:0': graph_input})
with tf.Session() as sess:
  new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
  graph = tf.get_default_graph()

  print([n.name for n in tf.get_default_graph().as_graph_def().node])
  out = tf.get_default_graph().get_tensor_by_name('output:0')
  graph_output = tensor_to_ipc(out, address=ipct.get_output_buffer(), size=32)

  #print(ipct.get_input_array())
  print('about to print out')
  p = tf.Print(out, [out, graph_input])
  sess.run(graph_output)
print('post', ipct.get_output_array())