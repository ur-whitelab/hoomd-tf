import hoomd.tensorflow_plugin
import unittest
import os, tempfile, shutil, pickle
import numpy as np, math, scipy
import tensorflow as tf

class test_ipc(unittest.TestCase):
    def test_ipc_to_tensor(self):
        ipc_to_tensor_module = hoomd.tensorflow_plugin.tfmanager.load_op_library('ipc2tensor')
        shape = [9, 4, 8]
        data = np.array(np.random.random_sample(shape), dtype=np.float32)
        pointer, _ = data.__array_interface__['data']
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor(address=pointer, T=np.float32, shape=shape)
        diff = tf.convert_to_tensor(data) - ipc_to_tensor
        sqe = tf.reduce_sum(diff**2)
        with tf.Session() as sess:
            result = sess.run(sqe)
        assert result < 10**-10

    def test_tensor_to_ipc(self):
        tensor_to_ipc_module = hoomd.tensorflow_plugin.tfmanager.load_op_library('tensor2ipc')
        shape = [8, 3, 2]
        data = np.ones(shape, dtype=np.float32)
        pointer, _ = data.__array_interface__['data']
        tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc(tf.zeros(shape, dtype=tf.float32), address=pointer, maxsize=np.prod(shape))
        with tf.Session() as sess:
            result = sess.run(tensor_to_ipc)
        assert np.sum(data) < 10**-10

if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])