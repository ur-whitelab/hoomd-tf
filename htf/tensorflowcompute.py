# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import htf._htf as _htf
from tfmanager import main
import sys
import math
import numpy as np
import pickle
import queue
import threading
import os
import time
import hoomd
import hoomd.md.nlist
import hoomd.comm
import tensorflow as tf

## \internal
# \brief TensorFlow HOOMD compute class
# \details
# Integrates tensorflow into HOOMD-blue, with options to load and save models,
# write a tensorboard file, run in "mock mode", and use XLA if available.

class tfcompute(hoomd.compute._compute):
    R""" TensorFlow Computations for HTF.

        :param tf_model_directory: Directory in which to save the TensorFlow model files.
        :param log_filename: Name to use for the TensorFlow log file.
        :param device: Device (GPU) on which to execute, if a specific one is desired.
        :param bootstrap: If set to a directory, will search for and
            load a previously-saved model file
        :param bootstrap_map: A dictionary to be used when bootstrapping,
            pairing old models' tensor variable names with new ones.
            Key is new name, value is older model's.
        :param _deubug_mode: Set this to True to see more debug messages.
        :param _mock_mode: Set this to True to run a "fake" calculation
            of forces that would be passed to the HOOMD simulation without applying them.
        :param write_tensorboard: If True, a tensorboard file will be written
            in the tf_model_directory.
        :param use_xla: If True, enables the accelerated linear algebra library
            in TensorFlow, which can be useful for large and complicated tensor operations.
    """
    ## \internal
    # \brief Constructs the tfcompute class
    # \details
    # Initializes the tfcompute class with options to manage how and where TensorFlow saves,
    # whether to use a tensorboard, and some execution preferences.
    def __init__(self, tf_model_directory,
                 log_filename='tf_manager.log', device=None,
                 bootstrap=None, bootstrap_map=None,
                 _debug_mode=False, _mock_mode=False, write_tensorboard=False,
                 use_xla=False):
        R""" Initialize a tfcompute class instance
        """

        # so delete won't fail
        self.tfm = None
        self.debug_mode = _debug_mode
        self.tf_model_directory = tf_model_directory
        self.log_filename = log_filename
        try:
            with open(os.path.join(tf_model_directory,
                                   'graph_info.p'), 'rb') as f:
                self.graph_info = pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError('Unable to load model'
                               ' in directory {}'.format(tf_model_directory))
        self.mock_mode = _mock_mode
        self.device = device
        self.write_tensorboard = write_tensorboard
        self.bootstrap = bootstrap
        self.bootstrap_map = bootstrap_map
        self.feed_dict = None
        self.use_xla = use_xla

    ## \var tfm
    # \internal
    # \brief TensorFlow manager used by the computer
    # \details
    # The TensorFlow manager instance used by this class to inteact with and
    # manage memory communication between HOOMD and TensorFlow.

    ## \var debug_mode
    # \internal
    # \brief Whether to print debug statements

    ## \var tf_model_directory
    # \internal
    # \brief Where to save TensorFlow model files

    ## \var log_filename
    # \internal
    # \brief Name of the log file

    ## \var graph_info
    # \internal
    # \brief The structure of the graph
    # \details
    # The information that is loaded from a pickled tensorflow graph model
    # as a dictionary, containing things like number of neighbors and other
    # information necessary to reconstruct the graph.

    ## \var _mock_mode
    # \internal
    # \brief Sets whether to disregard force output
    # \details
    # If this is True, then htf will run as if it were outputting forces
    # to HOOMD, but will not actually output them, just report them.

    ## \var device
    # \internal
    # \brief The GPU device to run on
    # \details
    # Set this argument to a string specifying which GPU to use on machines
    # with more than one GPU installed. Otherwise, defaults to 'gpu:0' in
    # GPU execution mode, or 'cpu:0' in CPU mode.

    ## \var write_tensorboard
    # \internal
    # \brief Whether to save a tensorboard file
    # \details
    # Tensorboard can be used to visualize the structure of a TensorFlow
    # graphical model. Set this to True if you want to save one, and it
    # will save in the same directory as the tf_model_directory

    ## \var bootstrap
    # \internal
    # \brief Location of previously-saved model to be loaded
    # \details
    # TensorFlow can optionally load parameters from a previously-trained
    # model. Set bootstrap to the location of such a model to make use of it.

    ## \var bootstrap_map
    # \internal
    # \brief Dictionary of previously-specified model's structure
    # \details
    # This map is filled with information loaded from a previously-trained
    # model. bootstrap must be set to a valid path for this to work.

    ## \var feed_dict
    # \internal
    # \brief Dictionary of values to set for TensorFlow placeholders
    # \details
    # Some model hyper parameters are user-specified, such as dropout rate
    # in a neural network model. Such variables are created as TensorFlow
    # placeholders. Populate a dictionary keyed by tensor names and desired
    # values, then pass as feed_dict.
    # feed_dict should be a dictionary where the key is
    # the tensor name (can be set during graph build stage),
    # and the value is the result to be fed into the named tensor. Note
    # that if you name a tensor, typically you must
    # append ':0' to that name. For example, if your name is 'my-tensor',
    # then the actual tensor is named 'my-tensor:0'.

    ## \var use_xla
    # \internal
    # \brief Whether to use XLA in TensorFlow
    # \details
    # XLA is an accelerated linear algebra library for TensorFlow which helps
    # speed up some tensor calculations. Use this to toggle
    # whether or not to use it.

    def __enter__(self):
        R""" __enter__ method of the HOOMD compute
        Launches TensorFlow.
        """
        if not self.mock_mode:
            self._init_tf()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        R""" __exit__ method of the HOOMD compute
        Tells TensorFlow to shutdown its instance, through TFManager.
        """
        # trigger end in task lock
        if not self.mock_mode and self.tfm.is_alive():
            hoomd.context.msg.notice(2, 'Sending exit signal.\n')
            if self.tfm and self.tfm.is_alive():
                hoomd.context.msg.notice(2, 'Shutting down TF Manually.\n')
                self.shutdown_tf()


    def attach(self, nlist=None, r_cut=0, save_period=1000,
               period=1, feed_dict=None, mol_indices=None,
               batch_size=None):
        R""" Attaches the TensorFlow instance to HOOMD.
        The main method of this class, this method sets up TensorFlow and
        gets HOOMD ready to interact with it.

        :param nlist: The HOOMD neighbor list that will be used as the TensorFlow input.
        :param r_cut: Cutoff radius for neighbor listing.
        :param save_period: How often to save the TensorFlow data. Period here is measured by
            how many times the TensorFLow model is updated. See period.
        :param period: How many HOOMD steps should pass before updating the TensorFlow model.
            In combination with save_period, determines how many timesteps pass before
            TensorFlow saves its data (slow). For example, with a save_period of 200,
            a period of 4, TensorFlow will write to the tf_model_directory every 800
            simulation steps.
        :param feed_dict: The dictionary keyed by tensor names and filled with corresponding values.
            See feed_dict in __init__.
        :param mol_indices: Molecule indices for each atom,
            identifying which molecule each atom belongs to.
        :param batch_size: The size of batches if we are using batching.
            Cannot be used if molecule-wise batching is active.
        """
        # make sure we have number of atoms and know dimensionality, etc.
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error('Must attach TF after initialization\n')
            raise RuntimeError('Error creating TF')
        if self.tfm is None and not self.mock_mode:
            raise Exception('You must use the with statement to construct '
                            'and attach a tfcompute')
        # I'm not sure if this is necessary following other files
        self.enabled = True
        self.log = True
        self.cpp_force = None
        self.feed_dict = feed_dict
        self.save_period = save_period
        self.force_name = 'tfcompute'
        self.compute_name = self.force_name
        self.nneighbor_cutoff = self.graph_info['NN']
        self.atom_number = len(hoomd.context.current.group_all)
        r_cut = float(r_cut)
        self.r_cut = r_cut
        self.batch_size = 0 if batch_size is None else batch_size

        if self.batch_size > 0:
            hoomd.context.msg.notice(2, 'Using fixed batching in htf\n')

        # find molecules if necessary
        if 'mol_indices' in self.graph_info and \
                self.graph_info['mol_indices'] is not None:
            if self.batch_size != 0:
                raise ValueError('Cannot batch by molecule and by batch_number')
            if hoomd.comm.get_num_ranks() > 1:
                raise ValueError('Molecular batches are '
                                 'not supported with spatial decomposition (MPI)')
            hoomd.context.msg.notice(2, 'Using molecular batching in htf\n')
            if mol_indices is None:
                sys = hoomd.data.system_data(hoomd.context.current.system_definition)
                mol_indices = \
                    htf.find_molecules(sys)
            self.mol_indices = mol_indices
            if type(self.mol_indices) != list:
                raise ValueError('mol_indices must be nested python list')
            if type(self.mol_indices[0]) != list:
                raise ValueError('mol_indices must be nested python list')
            # fill out the indices
            for mi in self.mol_indices:
                for i in range(len(mi)):
                    # add 1 so that an index of 0 corresponds to slicing a dummy atom
                    mi[i] += 1
                if len(mi) > self.graph_info['MN']:
                    raise ValueError('One of your molecule indices'
                                     'has more than MN indices.'
                                     'Increase MN in your graph.')
                while len(mi) < self.graph_info['MN']:
                    mi.append(0)
                    
            # now make reverse
            self.rev_mol_indices = _make_reverse_indices(self.mol_indices)

            # ok we have succeeded, now we try to disable sorting
            c = hoomd.context.current.sorter
            if c is None:
                hoomd.context.msg.notice(1, 'Unable to disable molecular sorting.'
                                         'Make sure you disable it to allow molecular batching')
            else:
                c.disable()
        else:
            self.mol_indices = None

        if nlist is not None:
            nlist.subscribe(self.rcut)
            # activate neighbor list
            nlist.update_rcut()
        elif self.nneighbor_cutoff != 0:
            raise ValueError('Must provide an nlist if you have '
                             'nneighbor_cutoff > 0')
        hoomd.util.print_status_line()
        # initialize base class
        hoomd.compute._compute.__init__(self)
        if self.graph_info['output_forces']:
            self.force_mode_code = _htf.FORCE_MODE.tf2hoomd
        else:
            self.force_mode_code = _htf.FORCE_MODE.hoomd2tf
        hoomd.context.msg.notice(2, 'Force mode is {}'
                                 ' \n'.format(self.force_mode_code))
        # if graph is not outputting (input) then tfcompute should
        # be outputting them
        state_1 = self.graph_info['output_forces']
        state_2 = _htf.FORCE_MODE.hoomd2tf
        if not state_1 and not state_2:
            raise ValueError('Your graph takes forces as input but you are'
                             ' not sending them from tfcompute')
        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = \
                _htf.TensorflowCompute(
                    self,
                    hoomd.context.current.system_definition,
                    nlist.cpp_nlist if nlist is not None else None,
                    r_cut,
                    self.nneighbor_cutoff,
                    self.force_mode_code,
                    period,
                    self.batch_size)
            # TODO: This is not correct
            if self.device is None:
                self.device = '/cpu:0'
        else:
            self.cpp_force = \
                _htf.TensorflowComputeGPU(self,
                                          hoomd.context.current.system_definition,
                                          nlist.cpp_nlist if nlist is not None else None,
                                          r_cut,
                                          self.nneighbor_cutoff,
                                          self.force_mode_code,
                                          period,
                                          self.batch_size)
            if self.device is None:
                self.device = '/gpu:0'
        # get double vs single precision
        self.dtype = tf.float32
        if self.cpp_force.isDoublePrecision():
            self.dtype = tf.float64
        # adding to forces causes the computeForces method to be called.
        hoomd.context.current.system.addCompute(self.cpp_force,
                                                self.compute_name)
        if self.force_mode_code == _htf.FORCE_MODE.tf2hoomd:
            hoomd.context.current.forces.append(self)
        else:
            integrator = hoomd.context.current.integrator
            if integrator is None:
                raise ValueError('Must have integrator set to receive forces')
            integrator.cpp_integrator.setHalfStepHook(self.cpp_force.hook())
        if not self.mock_mode:
            self._start_tf()

    def set_reference_forces(self, *forces):
        R""" Sets the HOOMD reference forces to be used by TensorFlow.
        See C++ comments in TensorFlowCompute.h
        """
        if self.force_mode_code == _htf.FORCE_MODE.tf2hoomd:
            raise ValueError('Only valid to set reference'
                             ' forces if mode is hoomd2tf')
        for f in forces:
            if not hasattr(f, 'cpp_force'):
                raise ValueError('given force does not seem'
                                 ' like a hoomd force')
            self.cpp_force.addReferenceForce(f.cpp_force)
            hoomd.context.msg.notice(5, 'Will use given force for '
                                     'TFCompute {} \n'.format(f.name))

    def rcut(self):
        R""" Define the cutoff radius used in the neighbor list.
        Adapted from hoomd/md/pair.py
        """
        # go through the list of only the active particle types in the sim
        sys_def = hoomd.context.current.system_definition
        ntypes = sys_def.getParticleData().getNTypes()
        type_list = []
        for i in range(0, ntypes):
            sys_def = hoomd.context.current.system_definition
            type_list.append(sys_def.getParticleData(
                    ).getNameByType(i))
        # update the rcut by pair type
        r_cut_dict = hoomd.md.nlist.rcut()
        for i in range(0, ntypes):
            for j in range(i, ntypes):
                # get the r_cut value
                r_cut_dict.set_pair(type_list[i], type_list[j], self.r_cut)
        return r_cut_dict

    def __del__(self):
        R""" delete method for the compute.
        Ensures that tensorflow is properly shut down before quitting.
        """
        if self.tfm and self.tfm.is_alive():
            self.shutdown_tf()

    def shutdown_tf(self):
        R""" Shut down the TensorFlow instance.
        """
        # need to terminate orphan
        if not self.q.full():
            hoomd.context.msg.notice(2, 'TF Queue is waiting, sending None\n')
            self.q.put(None)
        self.tfm.join(1)

    def _init_tf(self):
        R""" set up the TensorFlow instance.
        Create threading queue and give it to TensorFlow manager.
        """
        self.q = queue.Queue(maxsize=1)
        self.tfm = threading.Thread(target=main, args=(
                self.q, self.write_tensorboard, self.device))
        self.tfm.start()
        hoomd.context.msg.notice(2, 'Started TF Session Manager.\n')

    def _start_tf(self):
        R""" start the TensorFlow instance.
        Add our class var args to the queue, and print some graph info.
        """
        if not self.cpp_force:
            return
        args = {'log_filename': self.log_filename,
                'graph_info': self.graph_info,
                'positions_buffer': self.cpp_force.getPositionsBuffer(),
                'box_buffer': self.cpp_force.getBoxBuffer(),
                'nlist_buffer': self.cpp_force.getNlistBuffer(),
                'forces_buffer': self.cpp_force.getForcesBuffer(),
                'virial_buffer': self.cpp_force.getVirialBuffer(),
                'dtype': self.dtype,
                'use_feed': self.feed_dict is not None,
                'bootstrap': self.bootstrap,
                'bootstrap_map': self.bootstrap_map,
                'save_period': self.save_period,
                'debug': self.debug_mode,
                'primary': hoomd.comm.get_rank() == 0,
                'device': self.device,
                'use_xla': self.use_xla}
        self.q.put(args)
        message = ['Starting TF Manager with:']
        for k, v in args.items():
            if k == 'graph_info':
                continue
            else:
                message.append('\t{: <20}: {: >20}'.format(str(k), str(v)))
        message.append('\t{: <20}:'.format('graph_info'))
        for k, v in args['graph_info'].items():
            message.append('\t  {: <18}: {: >20}'.format(str(k), str(v)))
        for m in message:
            hoomd.context.msg.notice(8, m + '\n')
        self.q.join()
        if not self.tfm.is_alive():
            exit()
        hoomd.context.msg.notice(2, 'TF Session Manager has released control.'
                                 ' Starting HOOMD updates\n')

    def finish_update(self, batch_index, batch_frac):
        R""" Allow TF to read output and we wait for it to finish.

        :param batch_index: index of batch to be processed
        :param batch_frac: fractional batch index, i.e.
            ``batch_frac`` = ``batch_index / len(input)``
        """
        if self.mock_mode:
            return
        fd = {'htf-batch-index:0': batch_index, 'htf-batch-frac:0': batch_frac}
        if self.mol_indices is not None:
            fd[self.graph_info['mol_indices']] = self.mol_indices
            fd[self.graph_info['rev_mol_indices']] = self.rev_mol_indices
        if self.feed_dict is not None:
            if type(self.feed_dict) == dict:
                value = self.feed_dict
            else:
                value = self.feed_dict(self)
                assert value is not None, 'feed_dict callable failed to provide value'
            self.q.put({**value, **fd}, block=False)
        else:
            self.q.put(fd, block=False)
        self.q.join()
        if not self.tfm.is_alive():
            hoomd.context.msg.error('TF Session Manager has unexpectedly stopped\n')
            raise RuntimeError('TF Session Manager has unexpectedly stopped\n')

    def get_positions_array(self):
        R""" Retrieve positions array as numpy array
        """
        return self.scalar4_vec_to_np(self.cpp_force.getPositionsArray())

    def get_nlist_array(self):
        R""" Retrieve neighbor list array as numpy array
        """
        nl = self.scalar4_vec_to_np(self.cpp_force.getNlistArray())
        return nl.reshape(-1, self.nneighbor_cutoff, 4)

    def get_forces_array(self):
        R""" Retrieve forces array as numpy array
        """
        return self.scalar4_vec_to_np(self.cpp_force.getForcesArray())

    def get_virial_array(self):
        R""" Retrieve virial array as numpy array
        """
        array = np.array(self.cpp_force.getVirialArray())
        return array.reshape((-1, 9))

    def update_coeffs(self):
        pass

    def scalar4_vec_to_np(self, array):
        R""" Convert from scalar4 dtype to numpy array
        :param array: the scalar4 array to be processed
        """
        npa = np.empty((len(array), 4))
        for i, e in enumerate(array):
            npa[i, 0] = e.x
            npa[i, 1] = e.y
            npa[i, 2] = e.z
            npa[i, 3] = e.w
        return npa


def _make_reverse_indices(mol_indices):
    num_atoms = 0
    for m in mol_indices:
        num_atoms = max(num_atoms, max(m))
    # you would think add 1, since we found the largest index
    # but the atoms are 1-indexed to distinguish between
    # the "no atom" case (hence the - 1 below)
    rmi = [[] for _ in range(num_atoms)]
    for i in range(len(mol_indices)):
        for j in range(len(mol_indices[i])):
            index = mol_indices[i][j]
            if index > 0:
                rmi[index - 1] = [i, j]
    warned = False
    for r in rmi:
        if len(r) != 2 and not warned:
            warned = True
            hoomd.context.msg.notice(
                1,
                'Not all of your atoms are in a molecule\n')
            r.extend([-1, -1])
    return rmi
