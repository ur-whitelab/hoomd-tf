import tensorflow as tf
import os

def load_htf_op_library(op):
    import hoomd.htf
    path = hoomd.htf.__path__[0]
    try:
        op_path = os.path.join(path, op, 'lib_{}'.format(op))
        if os.path.exists(op_path + '.so'):
            op_path += '.so'
        elif os.path.exists(op_path + '.dylib'):
            op_path += '.dylib'
        else:
            raise OSError()
        mod = tf.load_op_library(op_path)
    except OSError:
        raise OSError('Unable to load OP {}. '
                      'Expected to be in {}'.format(op, path))
    return mod

class Hoomd2TF(tf.keras.layers.Layer):
    def __init__(self, nonbatch_input_shape, **kwargs):
        super(Hoomd2TF, self).__init__(**kwargs)
        self.address = 0
        self.nonbatch_input_shape = nonbatch_input_shape

    def build(self, input_shape):
        if self.address == 0:
            raise ValueError('Memory address must be set before build')
        hoomd_to_tf_module = load_htf_op_library('hoomd2tf_op')
        hoomd_to_tf = hoomd_to_tf_module.hoomd_to_tf
        self.hoomd_to_tf = hoomd_to_tf(
            address=self.address,
            shape = [input_shape[0]] + self.nonbatch_input_shape,
            T=self.dtype,
            name=self.name + '-input'
        )

    @tf.function
    def call(self, inputs, **kwargs):
        return self.hoomd_to_tf(inputs)

class TF2Hoomd(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TF2Hoomd, self).__init__(**kwargs)
        self.address = 0

    def build(self, input_shape):
        if self.address == 0:
            raise ValueError('Memory address must be set before build')
        tf_to_hoomd_module = load_htf_op_library('tf2hoomd_op')
        tf_to_hoomd = tf_to_hoomd_module.tf_to_hoomd
        self.tf_to_hoomd = tf_to_hoomd(
            address=self.address,
            name=self.name + '-output'
        )

    @tf.function
    def call(self, inputs, **kwargs):
        return self.tf_to_hoomd(inputs)


class NList(Hoomd2TF):
    def __init__(self, nneighbor_cutoff):
        super(NList, self).__init__([4 * nneighbor_cutoff], name='htf-nlist')
        self.nneighbor_cutoff = nneighbor_cutoff
    def call(self, inputs, **kwargs):
        result = super(NList, self).call(inputs, **kwargs)
        return tf.reshape(result, [[-1, self.nneighbor_cutoff, 4]])

class OutputForces(TF2Hoomd):
    def __init__(self):
        super(OutputForces, self).__init__(name='htf-forces')
