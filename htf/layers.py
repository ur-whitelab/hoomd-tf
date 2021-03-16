# Copyright (c) 2020 HOOMD-TF Developers

import tensorflow as tf
import hoomd.htf


class RBFExpansion(tf.keras.layers.Layer):
    R''' A  continuous-filter convolutional radial basis filter input from
    `SchNet <https://arxiv.org/pdf/1706.08566.pdf>`_.
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

    def get_config(self):
        config = super(RBFExpansion, self).get_config()
        config.update(
            {
                'low': self.low,
                'high': self.high,
                'count': self.centers.shape[0]
            })
        return config

    def call(self, inputs):
        rbf = tf.math.exp(-(inputs[..., tf.newaxis] -
                            self.centers)**2 / self.gap)
        return rbf


class WCARepulsion(tf.keras.layers.Layer):
    R'''A trainable Weeks-Chandler-Anderson repulsion. The input
    should be the neighbor list.

    .. math::

        U(r) = \begin{cases}
                \left(\frac{\sigma}{r}\right)^6 & r\leq 2^{1/3}\sigma \\
                0 & r\geq 2^{1/3}\sigma
               \end{cases}

    where :math:`\sigma` is a trainable variable set by ``start``. A regularization
    is applied to push :math:`\sigma` to higher distances.
    '''

    def __init__(self, sigma, regularization_strength=1e-3):
        R'''
        :param sigma: starting :math:`\sigma` value
        :type sigma: float

        :param regularization_strength: factor on regularization
        :type regularization_strength: float
        '''
        super(WCARepulsion, self).__init__(name='wca-repulsion')
        # we make it big to ensure
        self.sigma = self.add_weight(
            shape=(),
            regularizer=lambda x: -regularization_strength * x,
            initializer=tf.keras.initializers.Constant(value=sigma),
            name='sigma'
        )

    def get_config(self):
        config = super(WCARepulsion, self).get_config()
        config.update(
            {
                'sigma': float(self.sigma.value())
            })
        return config

    def call(self, nlist):
        rinv = hoomd.htf.nlist_rinv(nlist)
        true_sig = self.sigma
        rp = (true_sig * rinv)**6
        # make it so anything above cut is 0
        r = tf.norm(nlist[:, :, :3], axis=2)
        r_pair_energy = tf.cast(r < true_sig * 2**(1/3), tf.float32) * rp
        return tf.clip_by_value(r_pair_energy, 0, 10)


class EDSLayer(tf.keras.layers.Layer):
    R''' This layer computes and returns the Lagrange multiplier/EDS coupling constant (alpha)
    to be used as the EDS bias in the simulation. You call the layer on the
    collective variable at each step to get the current value of alpha.

    :param set_point: The set point value of the collective variable.
        This is a constant value which is pre-determined by the user and unique to each cv.
    :param period: Time steps over which the coupling constant is updated.
        Hoomd time units are used. If period=100 alpha will be updated each 100 time steps.
    :param learninig_rate: Learninig_rate in the EDS method.
    :param cv_scale: Used to adjust the units of the bias to Hoomd units.
    :param name: Name to be used for layer
    :return: Alpha, the EDS coupling constant.
    '''

    def __init__(self, set_point, period, learning_rate=1e-2,
                 cv_scale=1.0, name='eds-layer', **kwargs):
        if not tf.is_tensor(set_point):
            set_point = tf.convert_to_tensor(set_point)
        if set_point.dtype not in (tf.float32, tf.float64):
            raise ValueError(
                'EDS only works with floats, not dtype' +
                str(set_point.dtype))
        super().__init__(name, dtype=set_point.dtype, **kwargs)
        self.set_point = set_point
        self.period = tf.cast(period, tf.int32)
        self.cv_scale = cv_scale
        self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    def get_config(self):
        base = super().get_config()
        c = {
            'set_point': self.set_point.numpy(),
            'period': self.period,
            'cv_scale': self.cv_scale,
            'learning_rate': self.learning_rate,
        }
        c.update(base)
        return c

    def build(self, input_shape):
        # set-up variables
        self.mean = self.add_weight(
            initializer=tf.zeros_initializer(), dtype=self.dtype,
            shape=input_shape, name='{}.mean'.format(self.name),
                                    trainable=False)
        self.ssd = self.add_weight(
            initializer=tf.zeros_initializer(), dtype=self.dtype,
            shape=input_shape, name='{}.ssd'.format(self.name),
            trainable=False)
        self.n = self.add_weight(
            initializer=tf.zeros_initializer(),
            shape=input_shape, dtype=tf.int32, name='{}.n'.format(
                self.name), trainable=False)
        self.alpha = self.add_weight(initializer=tf.zeros_initializer(
        ), shape=input_shape, name='{}.a'.format(self.name), dtype=self.dtype)

    @tf.function
    def call(self, cv):
        reset_mask = tf.cast((self.n != 0), self.dtype)

        # reset statistics if n is 0
        self.mean.assign(self.mean * reset_mask)
        self.ssd.assign(self.ssd * reset_mask)

        # update statistics
        # do we update? - masked
        update_mask = tf.cast(self.n > self.period // 2, self.dtype)
        delta = (cv - self.mean) * update_mask
        self.mean.assign_add(
            tf.math.divide_no_nan(
                delta,
                tf.cast(self.n - self.period // 2, self.dtype)
            )
        )

        self.ssd.assign_add(delta * (cv - self.mean))

        # update grad
        update_mask = tf.cast(
            tf.equal(self.n, self.period - 1), self.dtype)
        gradient = update_mask * -  2 * \
            (self.mean - self.set_point) * self.ssd / \
            tf.cast(self.period, self.dtype) / 2 / self.cv_scale

        tf.cond(pred=tf.equal(self.n, self.period - 1),
                true_fn=lambda: self.optimizer.apply_gradients([(gradient,
                                                                 self.alpha)]),
                false_fn=lambda: tf.no_op())

        # update n. Should reset at period
        self.n.assign((self.n + 1) % self.period)

        return self.alpha
