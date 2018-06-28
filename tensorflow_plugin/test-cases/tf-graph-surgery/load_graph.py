import tensorflow as tf
import numpy as np

#make a block of memory in C
array = np.ones((32, 4), np.float32)
array1 = [1.0 for _ in range(32 * 4)]
address1 = id(array1)
#array2 = np.zeros((32,4), np.float32)
array2 = [0.0 for _ in range(32 * 4)]
address2 = id(array2)
ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor
tensor_to_ipc_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/tensor2ipc/lib_tensor2ipc_op.so')
tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc

graph_input = ipc_to_tensor(address=address1, size=32, T=np.float32)
print(graph_input)
output_tensor = tf.Variable(expected_shape=(32, 4), name='output_tensor', dtype=np.float32, initial_value=0)

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('model/test.meta', input_map={'input:0': graph_input})
  new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
  graph = tf.get_default_graph()

  print([n.name for n in tf.get_default_graph().as_graph_def().node])
  out = tf.get_default_graph().get_tensor_by_name('output:0')
  #print(tf.get_default_graph().as_graph_def().SerializeToString())
  #graph_output = tensor_to_ipc(out, address=address2, size=32)

  print(array2)
  sess.run(tf.Print(out, [out, graph_input]))
print(array2)

with tf.Session() as sess:
  sess.run(tf.Print(graph_input, [graph_input]))