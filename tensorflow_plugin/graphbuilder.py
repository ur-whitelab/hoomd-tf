# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import tensorflow as tf
import os, pickle

class graph_builder:
    '''This is a python class that builds the TF graph.

    Use safe_div class method to avoid nan forces if doing 1/r or equivalent force calculations
    '''

    def __init__(self, nneighbor_cutoff, output_forces=True):
        '''
        Parameters
        ------------
        nneighbor_cutoff
            The maximum number of neigbhors to consider (can be 0)
        output_forces
            True if your graph will compute forces to be used in TF
        '''
        #clear any previous graphs
        atom_number = None
        tf.reset_default_graph()
        self.nneighbor_cutoff = nneighbor_cutoff
        #use zeros so that we don't need to feed to start session
        self.nlist = tf.placeholder (tf.float32, shape=[atom_number, nneighbor_cutoff, 4], name='nlist-input')
        self.virial = None
        self.positions = tf.placeholder(tf.float32, shape=[atom_number, 4], name='positions-input')
        if not output_forces:
            self.forces = tf.placeholder(tf.float32, shape=[atom_number, 4], name='forces-input')
        self.output_forces = output_forces
        self._nlist_rinv = None
        self.tensor_step = tf.get_variable('htf-step', dtype=tf.float32, initializer=0.0)
        update_step_op = self.tensor_step.assign_add(1.0)
        self.out_nodes = [update_step_op]

    @property
    def nlist_rinv(self):
        ''' Returns an N x NN tensor of 1 / r for each neighbor
        '''
        if self._nlist_rinv is None:
            r = tf.norm(self.nlist[:,:,:3], axis=2)
            self._nlist_rinv = self.safe_div(1.0, r)
        return self._nlist_rinv

    def masked_nlist(self, type_i = None, type_j = None, nlist = None, type_tensor = None):
        '''Returns a neighbor list masked by the given types.

        Parameters
        ---------
        name
            The name of the tensor
        type_i, type_j
            Use these to select only a certain particle type.
        nlist
            By default it will use self.nlist
        type_tensor
            An N x 1 tensor containing the types of the nlist origin. If None,
            then self.positions will be used
        '''
        if nlist is None:
            nlist = self.nlist
        if type_tensor is None:
            type_tensor = self.positions[:,3]
        if type_i is not None:
            nlist = tf.boolean_mask(nlist, tf.equal(type_tensor, type_i))
        if type_j is not None:
            # cannot use boolean mask due to size
            mask = tf.cast(tf.equal(nlist[:,:,3],type_j), tf.float32)
            # make it correct size to mask
            mask = tf.reshape(tf.tile(mask, [1, nlist.shape[2]]), [-1, self.nneighbor_cutoff, nlist.shape[2]])
            nlist = nlist * mask
        return nlist

    def compute_rdf(self, r_range, name, nbins=100, type_i = None, type_j = None, nlist = None, positions = None):
        '''Creates a tensor that has the rdf for a given frame.

        Parameters
        ----------
        bins
            The bins to use for the RDF
        name
            The name of the tensor containing rdf. The name will be concatenated with
            '-r' to create a tensor containing the r values of the rdf.
        type_i, type_j
            Use these to select only a certain particle type.
        nlist
            By default it will use self.nlist
        positions
            By default will used built-in positions. This tensor is only used
            to get the origin particle's type. So if you're making your own,
            just make sure column 4 has the type index.

        '''
        # to prevent type errors later on
        r_range = [float(r) for r in r_range]
        if nlist is None:
            nlist = self.nlist
        if positions is None:
            positions = self.positions
        # filter types
        nlist = self.masked_nlist(type_i, type_j, nlist)
        r = tf.norm(nlist[:,:,:3], axis=2)
        hist = tf.cast(tf.histogram_fixed_width(r, r_range, nbins + 2), tf.float32)
        shell_rs = tf.linspace(r_range[0], r_range[1], nbins + 1)
        vis_rs = tf.multiply((shell_rs[1:] + shell_rs[:-1]), 0.5, name=name + '-r')
        vols = shell_rs[1:]**3 - shell_rs[:-1]**3
        # remove 0s and Ns
        result = hist[1:-1] / vols
        self.out_nodes.extend([result, vis_rs])
        return result

    def running_mean(self, tensor, name):
        '''Computes running mean of the given tensor

        Parameters
        ----------
            tensor
                The tensor for which you're computing running mean
            name
                The name of the variable in which the running mean will be stored

        Returns
        -------
            A variable containing the running mean

        '''

        store = tf.get_variable(name, initializer=tf.zeros_like(tensor), validate_shape=False)
        update_op = store.assign_add( (tensor - store) / self.tensor_step)
        self.out_nodes.append(update_op)
        return store


    def compute_forces(self, energy, virial=None,positions=None,nlist=None,name=None):
        ''' Computes pairwise or position-dependent forces (field) given
        a potential energy function that computes per-particle or overall energy

        Parameters
        ----------
        energy
            The potential energy
        virial
             None - (default) virial contribution will be computed if the graph outputs forces
             Can be set with True/False instead. Note that the virial term that depends on positions is
             not computed.
        Returns
        --------
        The TF force tensor. Note that the virial part will be stored as the class attribute virial and will
        be saved automatically.

        '''
        if virial is None:
            if self.output_forces:
                virial = True
            else:
                virial = False
        if nlist is None:
            nlist = self.nlist
        if positions is None:
            positions = self.positions
        with tf.name_scope('force-gradient'):
            #compute -gradient wrt positions
            if positions is not False:
                pos_forces = tf.gradients(tf.negative(energy), positions)[0]
            else:
                pos_forces = None
            if pos_forces is not None:
                pos_forces = tf.identity(pos_forces, name='pos-force-gradient')
            #minus sign cancels when going from force on neighbor to force on origin in nlist
            nlist_forces = tf.gradients(energy, nlist)[0]
            if nlist_forces is not None:
                nlist_forces = tf.identity(2.0 * nlist_forces, name='nlist-pairwise-force-gradient-raw')
                zeros = tf.zeros(tf.shape(nlist_forces))
                nlist_forces = tf.where(tf.is_finite(nlist_forces), nlist_forces, zeros, name='nlist-pairwise-force-gradient')
                nlist_reduce = tf.reduce_sum(nlist_forces, axis=1, name='nlist-force-gradient')
                if virial:
                    with tf.name_scope('virial-calc'):
                        #now treat virial
                        nlist3 = nlist[:, :, :3]
                        rij_outter = tf.einsum('ijk,ijl->ijkl', nlist3, nlist3)
                        #F / rs
                        self.nlist_r_mag = graph_builder.safe_norm(nlist3, axis=2, name='nlist-r-mag')
                        self.nlist_force_mag = graph_builder.safe_norm(nlist_forces, axis=2, name='nlist-force-mag')
                        F_rs = self.safe_div(self.nlist_force_mag, 2.0 * self.nlist_r_mag)
                        #sum over neighbors: F / r * (r (outter) r)
                        self.virial = -1.0 * tf.einsum('ij,ijkl->ikl', F_rs, rij_outter)
        if pos_forces is None and nlist_forces is None:
            raise ValueError('Found no dependence on positions or neighbors so forces cannot be computed')
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
            energy = tf.reshape(tf.reduce_sum(energy, axis=list(range(1, len(energy.shape)))), [tf.shape(forces)[0], 1])
            forces = tf.concat([forces[:,:3], energy], -1)
        elif len(energy.shape) == 1 and energy.shape[0] == 1:
            forces = tf.concat([forces[:,:3], tf.tile(energy, tf.shape(forces)[0])], -1)
        else:
            forces = tf.concat([forces[:,:3], tf.reshape(energy, [tf.shape(forces)[0], 1])], -1)
        return tf.identity(forces, name='computed-forces')

    @staticmethod
    def safe_div(numerator, denominator, delta=3e-6, **kwargs):
        '''
        There are some numerical instabilities that can occur during learning
        when gradients are propagated. The delta is problem specific.
        '''
        op = tf.where(
               tf.greater(denominator, delta),
               tf.truediv(numerator, denominator + delta),
               tf.zeros_like(denominator))

        #op = tf.divide(numerator, denominator + delta, **kwargs)
        return op


    @staticmethod
    def safe_norm(tensor, delta=1e-7, **kwargs):
        '''
        There are some numerical instabilities that can occur during learning
        when gradients are propagated. The delta is problem specific.
        NOTE: delta of safe_div must be > sqrt(3) * (safe_norm delta)
        #https://github.com/tensorflow/tensorflow/issues/12071
        '''
        return tf.norm(tensor + delta, **kwargs)

    def save(self, model_directory, force_tensor = None, virial = None, out_nodes=[], move_previous=True):
        '''Save the graph model to specified directory.

        Parameters
        ----------
        model_directory
            Multiple files will be saved, including a dictionary with information specific to hoomd-tf and TF
            model files.
        force_tensor
            The forces that should be sent to hoomd
        virial
            The virial which should be sent to hoomd. If None and you called compute_forces, then
            the virial computed from that function will be saved.
        out_nodes
            Any additional TF graph nodes that should be executed.
            For example, optimizers, printers, etc.
        '''
        if force_tensor is None and self.output_forces:
            raise ValueError('You must provide force_tensor if you are outputing forces')

        if force_tensor is not None and not self.output_forces:
            raise ValueError('You should not provide forces since you set output_forces to be False in constructor')

        if type(out_nodes) != list:
            raise ValueError('out_nodes must be a list')

        # add any attribute out_nodes
        out_nodes += self.out_nodes

        if self.output_forces:
            if len(force_tensor.shape) != 2:
                raise ValueError('force_tensor should be N x 3 or N x 4. You gave a ' + ','.join([str(x) for x in force_tensor.shape]))
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
                assert virial.shape == [None, self.nneighbor_cutoff, 3, 3]
        else:
            if len(out_nodes) == 0:
                raise ValueError('You must provide nodes to run (out_nodes) if you are not outputting forces')

        os.makedirs(model_directory, exist_ok=True)

        if move_previous and len(os.listdir(model_directory)) > 0:
            bkup_int = 0
            bkup_str = 'previous_model_{}'.format(bkup_int)
            while bkup_str in os.listdir(model_directory):
                bkup_int += 1
                bkup_str = 'previous_model_{}'.format(bkup_int)
            os.makedirs(os.path.join(model_directory, bkup_str))
            for i in os.listdir(model_directory):
                if os.path.isfile(os.path.join(model_directory, i)):
                    os.rename(os.path.join(model_directory, i),
                              os.path.join(model_directory, bkup_str, i))
            print('Note: Backed-up {} previous model to {}'.format(model_directory, os.path.join(model_directory, bkup_str)))


        meta_graph_def = tf.train.export_meta_graph(filename=(os.path.join(model_directory, 'model.meta')))
        #with open(os.path.join(model_directory, 'model.pb2'), 'wb') as f:
        #    f.write(tf.get_default_graph().as_graph_def().SerializeToString())
        #save metadata of class
        graph_info = {  'NN': self.nneighbor_cutoff,
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
