import unittest


class test_imports(unittest.TestCase):
    def test_load_hoomd(self):
        import hoomd

    def test_load_htf(self):
        import hoomd.htf

    def test_load_tf(self):
        import tensorflow as tf


class test_op(unittest.TestCase):
    def test_op_load(self):
        import hoomd
        import hoomd.htf
        import tensorflow as tf
        from hoomd.htf.tfmanager import load_htf_op_library

        hoomd_to_tf_module = load_htf_op_library('hoomd2tf_op')

    def test_op_nparray_create(self):
        import hoomd
        import hoomd.htf
        import tensorflow as tf
        import numpy as np
        from hoomd.htf.tfmanager import load_htf_op_library
        hoomd_to_tf_module = load_htf_op_library('hoomd2tf_op')
        hoomd_to_tf = hoomd_to_tf_module.hoomd_to_tf

        with tf.device("/cpu:0"):
            tf_op = hoomd_to_tf(address=0,
                                shape=[2], T=tf.float64,
                                name='array-64-test')


if __name__ == '__main__':
    unittest.main()
