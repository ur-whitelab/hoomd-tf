import tensorflow as tf


class RBFExpansion(tf.keras.layers.Layer):
    def __init__(self, low, high, count):
        super(RBFExpansion, self).__init__(name='rbf-layer')
        self.low = low
        self.high = high
        self.centers = tf.cast(tf.linspace(low, high, count), dtype=tf.float32)
        self.gap = self.centers[1] - self.centers[0]

    def call(self, inputs):
        # input shpe
        x = tf.reshape(inputs, (-1,))
        rbf = tf.math.exp(-(x[:, tf.newaxis] - self.centers)**2 / self.gap)
        # remove 0s
        rbf *= tf.cast(x > 1e-5, tf.float32)[:, tf.newaxis]
        return tf.reshape(rbf, tf.concat((tf.shape(inputs), self.centers.shape), axis=0))


class WCARepulsion(tf.keras.layers.Layer):
    def __init__(self, start):
        super(WCARepulsion, self).__init__(name='wca-repulsion')
        # we make it big to ensure
        # high learning rates don't move it real fast
        self.sigma = self.add_variable(
            shape=(),
            regularizer=lambda x: -x,
            initializer=tf.keras.initializers.Constant(value=start)
        )

    def call(self, nlist):
        rinv = htf.nlist_rinv(nlist)
        true_sig = self.sigma
        rp = (true_sig * rinv)**6
        # make it so anything above cut is 0
        r = tf.norm(nlist[:, :, :3], axis=2)
        r_pair_energy = tf.cast(r < true_sig * 2**(1/3), tf.float32) * rp
        return r_pair_energy
