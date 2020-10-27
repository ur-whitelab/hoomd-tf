import tensorflow as tf
import os
import unittest
import onnx, onnx_tf
import keras2onnx
import hoomd.htf
import build_examples
import numpy as np

class test_onnx(unittest.TestCase):
    def test_get_tf_graph(self):
        # model = build_examples.LJModel(32)

        class SimplePotential(tf.keras.Model):
            def call(self, inputs):
                nlist = inputs[0]
                positions = inputs[1]
                nlist = nlist[:, :, :3]
                neighs_rs = tf.norm(tensor=nlist, axis=2, keepdims=True)
                # no need to use netwon's law because nlist should be double counted
                fr = tf.multiply(-1.0, tf.multiply(tf.math.reciprocal(neighs_rs), nlist),
                                name='nan-pairwise-forces')
                zeros = tf.zeros_like(nlist)
                real_fr = tf.where(tf.math.is_inf(fr), zeros, fr,
                                name='pairwise-forces')
                forces = tf.reduce_sum(input_tensor=real_fr, axis=1, name='forces')
                return forces
        
        model = SimplePotential()
        
        nlist_input = np.random.normal(size=(64, 8, 4))
        positions_input = np.random.normal(size=(64,4))
        expected = model((nlist_input, positions_input))
        model._set_inputs((nlist_input, positions_input))
        # create an onnx model
        onnx_graph = keras2onnx.convert_keras(model, target_opset = 10)

        keras2onnx.save_model(onnx_graph,"/tmp/model.onnx" )

        class ModelFromONNX(hoomd.htf.SimModel):
            def setup(self):
                onnx_model = onnx.load("/tmp/model.onnx")
                self.tf_rep = onnx_tf.backend.prepare(onnx_model)
            def compute(self, nlist, positions):
                forces = self.tf_rep.run((nlist, positions))
                return forces

        from_onnx = ModelFromONNX(8)
        onnx_forces = from_onnx.compute(nlist_input, positions_input)
        np.testing.assert_allclose(expected,onnx_forces)

