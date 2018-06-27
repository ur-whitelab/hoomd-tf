import tensorflow as tf
import numpy as np

#make a block of memory in C
array1 = np.ones((32,4), np.float32)
address1 = id(array1)
array2 = np.zeros((32,4), np.float32)
address2 = id(array2)
ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor
tensor_to_ipc_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/tensor2ipc/lib_tensor2ipc_op.so')
tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc

graph_input = ipc_to_tensor(address=address1, size=32, T=np.float32)
output_tensor = tf.Variable(expected_shape=(32, 4), name='output_tensor', dtype=np.float32, initial_value=0)
graph_output = tensor_to_ipc(output_tensor, address=address2, size=32)

with open('graph.pb', 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, input_map={'input:0': graph_input,
                                            'output:0': output_tensor})
  print(array2)
  with tf.Session() as sess:
    sess.run(graph_output)
  print(array2)