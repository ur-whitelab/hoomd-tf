import tensorflow as tf
import os, pickle

class graph_builder:
    '''Use safe_div class method to avoid nan forces if doing 1/r or equivalent force calculations'''

    def __init__(self, atom_number, nneighbor_cutoff, output_forces=True):
        '''output_forces -> should the graph output forces'''
        #clear any previous graphs
        tf.reset_default_graph()
        self.atom_number = atom_number
        self.nneighbor_cutoff = nneighbor_cutoff
        #use zeros so that we don't need to feed to start session
        self.nlist = tf.zeros ([atom_number, nneighbor_cutoff, 4], name='nlist')
        self.virial = None
        self.positions = tf.zeros ([atom_number, 4], name='positions')
        if not output_forces:
            self.forces = tf.zeros([atom_number, 4], name='forces')
        self.output_forces = output_forces

    def compute_forces(self, energy, name='forces'):

        with tf.name_scope('force-gradient'):
            #compute -gradient wrt positions
            pos_forces = tf.gradients(tf.negative(energy), self.positions)[0]
            if pos_forces is not None:
                pos_forces = tf.identity(pos_forces, name='pos-force-gradient')
            #minus sign cancels when going from force on neighbor to force on origin in nlist
            nlist_forces = tf.gradients(energy, self.nlist)[0]
            if nlist_forces is not None:
                nlist_forces = tf.identity(2.0 * nlist_forces, name='nlist-pairwise-force-gradient-raw')
                zeros = tf.zeros(tf.shape(nlist_forces))
                nlist_forces = tf.where(tf.is_nan(nlist_forces), zeros, nlist_forces, name='nlist-pairwise-force-gradient')
                nlist_reduce = tf.reduce_sum(nlist_forces, axis=1, name='nlist-force-gradient')
                with tf.name_scope('virial-calc'):
                    #now treat virial
                    nlist3 = self.nlist[:, :, :3]
                    rij_outter = tf.einsum('ijk,ijl->ijkl', nlist3, nlist3)
                    #F / rs
                    self.nlist_r_mag = tf.norm(nlist3, axis=2, name='nlist-r-mag')
                    self.nlist_force_mag = tf.norm(nlist_forces, axis=2, name='nlist-force-mag')
                    F_rs = self.safe_div(self.nlist_force_mag, 2.0 * self.nlist_r_mag)
                    #sum over neighbors: F / r * (r (outter) r)
                    self.virial = -1.0 * tf.einsum('ij,ijkl->ikl', F_rs, rij_outter)
        if pos_forces is not None and nlist_forces is not None:
            forces = tf.add(nlist_reduce, pos_forces, name='forces-added')
        elif pos_forces is None:
            forces = nlist_reduce
        else:
            forces = pos_forces

        #set w to be potential energy
        if len(energy.shape) > 1:
            #reduce energy to be correct shape
            print('WARNING: Your energy is multidimensional per particle. Hopefully that is intentional')
            energy = tf.reshape(tf.reduce_sum(energy, axis=list(range(1, len(energy.shape)))), [forces.shape[0], 1])
            forces = tf.concat([forces[:,:3], energy], -1)
        elif len(energy.shape) == 1 and energy.shape[0] == 1:
            forces = tf.concat([forces[:,:3], tf.tile(energy, forces.shape[0])], -1)
        elif energy.shape[0] == forces.shape[0]:
            forces = tf.concat([forces[:,:3], tf.reshape(energy, [forces.shape[0], 1])], -1)
        else:
            raise ValueError('energy must either be scalar or have first dimension of atom_number')
        return tf.identity(forces, name='computed-forces')

    @staticmethod
    def safe_div(numerator, denominator, name='graphbuild-safe-div'):
        '''Divides two values, returning 0 if the denominator is <= 0.
        Args:
            numerator: A real `Tensor`.
            denominator: A real `Tensor`, with dtype matching `numerator`.
            name: Name for the returned op.
        Returns:
            0 if `denominator` <= 0, else `numerator` / `denominator`
        Taken from tensorflow/contrib/metrics/python/ops/metric_ops.py
        '''
        with tf.name_scope('graphbuilder-safe-div'):
            op = tf.where(
                tf.greater(denominator, 0),
                tf.truediv(numerator, denominator),
                tf.zeros_like(denominator),
                name=name)

        return op



    def save(self, model_directory, force_tensor = None, virial = None, out_nodes=[]):

        if force_tensor is None and self.output_forces:
            raise ValueError('You must provide force_tensor if you are outputing forces')

        if type(out_nodes) != list:
            raise ValueError('out_nodes must be a list')

        if self.output_forces:
            if force_tensor.shape[0] != self.atom_number:
                raise ValueError('Dimension of force_tensor should be same as atom number')

            if force_tensor.shape[1] == 3:
                #add w information if it was removed
                with tf.name_scope('add-ws'):
                    force_tensor = tf.concat([force_tensor, tf.reshape(self.positions[:,3], [-1,  1])], axis=1, name='forces')

            self.forces = force_tensor
            if virial is None:
                if self.virial is not None:
                    virial = self.virial
                else:
                    print('WARNING: You did not provide a virial for {}, so per particle virials will not be correct'.format(model_directory))
            else:
                assert virial.shape == [self.atom_number, self.nneighbor_cutoff, 3, 3]
        else:
            if len(out_nodes) == 0:
                raise ValueError('You must provide nodes to run (out_nodes) if you are not outputting forces')

        os.makedirs(model_directory, exist_ok=True)
        meta_graph_def = tf.train.export_meta_graph(filename=(os.path.join(model_directory, 'model.meta')))
        #with open(os.path.join(model_directory, 'model.pb2'), 'wb') as f:
        #    f.write(tf.get_default_graph().as_graph_def().SerializeToString())
        #save metadata of class
        graph_info = {  'N': self.atom_number,
                        'NN': self.nneighbor_cutoff,
                        'model_directory': model_directory,
                        'forces': self.forces.name,
                        'positions': self.positions.name,
                        'virial': None if virial is None else virial.name,
                        'nlist': self.nlist.name,
                        'dtype': self.nlist.dtype,
                        'output_forces': self.output_forces,
                        'out_nodes': [x.name for x in out_nodes]
                        }
        with open(os.path.join(model_directory, 'graph_info.p'), 'wb') as f:
            pickle.dump(graph_info, f)
