import tensorflow as tf
import os, pickle

class GraphBuilder:
    '''Use safe_div class method to avoid nan forces if doing 1/r or equivalent force calculations'''
    def __init__(self, atom_number, nneighbor_cutoff):
        self.atom_number = atom_number
        self.nneighbor_cutoff = nneighbor_cutoff
        self.nlist = tf.zeros ([atom_number, nneighbor_cutoff, 4], name='nlist')
        self.positions = tf.zeros ([atom_number, 4], name='positions')

    def compute_forces(self, energy, overwrite_w=True):

        with tf.name_scope('force-gradient'):
            #compute -gradient wrt positions
            pos_forces = tf.gradients(tf.negative(energy), self.positions)[0]
            if pos_forces is not None:
                pos_forces = tf.identity(pos_forces, name='pos-force-gradient')
            #minus sign cancels when going from force on neighbor to force on origin in nlist
            nlist_forces = tf.gradients(energy, self.nlist)[0]
            if nlist_forces is not None:
                nlist_forces = tf.identity(nlist_forces, name='nlist-pairwise-force-gradient-raw')
                zeros = tf.zeros(tf.shape(nlist_forces))
                nlist_forces = tf.where(tf.is_nan(nlist_forces), zeros, nlist_forces, name='nlist-pairwise-force-gradient')
                nlist_reduce = tf.reduce_sum(nlist_forces, axis=1, name='nlist-force-gradient')
        if pos_forces is not None and nlist_forces is not None:
            forces = tf.add(nlist_reduce, pos_forces, name='forces-badw')
        elif pos_forces is None:
            forces = nlist_reduce
        else:
            forces = pos_forces

        #make sure w doesn't change
        if overwrite_w:
            forces = forces[:,:3]
        return tf.identity(forces, name='forces')

    @staticmethod
    def safe_div(numerator, denominator, name):
        '''Divides two values, returning 0 if the denominator is <= 0.
        Args:
            numerator: A real `Tensor`.
            denominator: A real `Tensor`, with dtype matching `numerator`.
            name: Name for the returned op.
        Returns:
            0 if `denominator` <= 0, else `numerator` / `denominator`
        Taken from tensorflow/contrib/metrics/python/ops/metric_ops.py
        '''
        return tf.where(
            tf.greater(denominator, 0),
            tf.truediv(numerator, denominator),
            tf.zeros_like(denominator),
        name=name)



    def save(self, force_tensor, model_directory):

        assert force_tensor.shape[0] == self.atom_number
        if force_tensor.shape[1] == 3:
            with tf.name_scope('add-ws'):
                force_tensor = tf.concat([force_tensor, tf.reshape(self.positions[:,3], [-1,  1])], axis=1, name='forces')

        self.forces = force_tensor
        if force_tensor.name != 'force:0':
            self.forces = tf.identity(force_tensor, name='force')

        tf.Variable(self.forces, name='force-save')

        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, os.path.join(model_directory, 'model'))
        #save metadata of class
        graph_info = {  'N': self.atom_number,
                        'NN': self.nneighbor_cutoff,
                        'model_directory': model_directory,
                        'forces': self.forces.name,
                        'positions': self.positions.name,
                        'nlist': self.nlist.name,
                        'dtype': self.nlist.dtype}
        with open(os.path.join(model_directory, 'graph_info.p'), 'wb') as f:
            pickle.dump(graph_info, f)