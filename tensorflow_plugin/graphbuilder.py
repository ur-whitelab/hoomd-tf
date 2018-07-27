import tensorflow as tf
import os, pickle

class GraphBuilder:
    def __init__(self, atom_number, nneighbor_cutoff):
        self.atom_number = atom_number
        self.nneighbor_cutoff = nneighbor_cutoff
        self.nlist = tf.zeros ([atom_number, nneighbor_cutoff, 4], name='nlist')
        self.positions = tf.zeros ([atom_number, 4], name='positions')

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