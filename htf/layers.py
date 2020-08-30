import tensorflow as tf
from .simmodel import *


class RBFExpansion(tf.keras.layers.Layer):
    R''' A  continuous-filter convolutional radial basis filter input from `SchNet <https://arxiv.org/pdf/1706.08566.pdf>`_.
    The input should be a rank ``K`` tensor of distances. The output will be rank ``K``
    with the new axis being of dimension ``count``. The distances are converted with
    :math:`\exp\gamma\left(d - \mu\right)^2` where :math:`\mu` is an evenly spaced
    grid from ``low`` to ``high`` containing ``count`` elements. The distance between
    elements is :math:`1 / \gamma`.
    '''

    def __init__(self, low, high, count):
        R'''
        :param low: lowest :math:`\mu`
        :type low: float

        :param high: high :math:`\mu` (inclusive)
        :type high: float

        :param count: Number of elements in :math:`\mu` and output last axis dimension
        :type count: int

        '''
        super(RBFExpansion, self).__init__(name='rbf-layer')
        self.low = low
        self.high = high
        self.centers = tf.cast(tf.linspace(low, high, count), dtype=tf.float32)
        self.gap = self.centers[1] - self.centers[0]

    def call(self, inputs):
        # input shpe
        #x = tf.reshape(inputs, (-1,))
        rbf = tf.math.exp(-(inputs[..., tf.newaxis] -
                            self.centers)**2 / self.gap)
        # remove 0s
        #rbf *= tf.cast(x > 1e-5, tf.float32)[:, tf.newaxis]
        return rbf


class WCARepulsion(tf.keras.layers.Layer):
    R'''A trainable Weeks-Chandler-Anderson repulsion. The input
    should be the neighbor list.

    .. math::

        U(r) = \begin{cases}
                \left(\frac{\sigma}{r}\right)^6 & r\leq 2^{1/3}\sigma \\
                0 & r\geq 2^{1/3}\sigma
               \end{cases}

    where :math:`\sigma` is a trainable variable set by ``start``.
    '''

    def __init__(self, sigma):
        R'''
        :param sigma: starting :math:`\sigma` value
        :type sigma: float
        '''
        super(WCARepulsion, self).__init__(name='wca-repulsion')
        # we make it big to ensure
        self.sigma = self.add_weight(
            shape=(),
            regularizer=lambda x: -1e-3 * x,
            initializer=tf.keras.initializers.Constant(value=sigma)
        )

    def call(self, nlist):
        rinv = nlist_rinv(nlist)
        true_sig = self.sigma
        rp = (true_sig * rinv)**6
        # make it so anything above cut is 0
        r = tf.norm(nlist[:, :, :3], axis=2)
        r_pair_energy = tf.cast(r < true_sig * 2**(1/3), tf.float32) * rp
        return tf.clip_by_value(r_pair_energy, 0, 10)
