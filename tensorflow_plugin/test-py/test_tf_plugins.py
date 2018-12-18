import hoomd.tensorflow_plugin
import unittest
import os, tempfile, shutil, pickle
import numpy as np, math, scipy
import tensorflow as tf

class test_ipc(unittest.TestCase):
    def test_hoomd_to_tf(self):
        hoomd_to_tf_module = hoomd.tensorflow_plugin.tfmanager.load_op_library('hoomd2tf')
        shape = [9, 4, 8]
        data = np.array(np.random.random_sample(shape), dtype=np.float32)
        pointer, _ = data.__array_interface__['data']
        with tf.device("/cpu:0"):
            hoomd_to_tf = hoomd_to_tf_module.hoomd_to_tf(address=pointer, T=np.float32, shape=shape)
            diff = tf.convert_to_tensor(data) - hoomd_to_tf
            sqe = tf.reduce_sum(diff**2)
        with tf.Session() as sess:
            result = sess.run(sqe)
        assert result < 10**-10

    def test_tf_to_hoomd(self):
        tf_to_hoomd_module = hoomd.tensorflow_plugin.tfmanager.load_op_library('tf2hoomd')
        shape = [8, 3, 2]
        data = np.ones(shape, dtype=np.float32)
        pointer, _ = data.__array_interface__['data']
        with tf.device("/cpu:0"):
            tf_to_hoomd = tf_to_hoomd_module.tf_to_hoomd(tf.zeros(shape, dtype=tf.float32), address=pointer, maxsize=np.prod(shape))
        with tf.Session() as sess:
            result = sess.run(tf_to_hoomd)
        assert np.sum(data) < 10**-10

if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])