# Copyright (c) 2020 HOOMD-TF Developers

from hoomd.htf import _htf
from .simmodel import *
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
import hoomd.htf
import tensorflow as tf


class tfcompute(hoomd.compute._compute):
    R'''
    The main class for applying :py:class:`.SimModel`
    to Hoomd simulation.

    '''

    def __init__(self, model):
        R''' Initialize a tfcompute class instance

        :param model: HOOMD-TF model
        :type model: :py:class:`.SimModel`
        '''
        self.model = model

    def attach(self, nlist=None, r_cut=0, period=1,
               batch_size=None, train=False, save_output_period=None):
        R''' Attaches the TensorFlow instance to Hoomd.
        This method sets up TensorFlow and
        gets Hoomd ready to interact with it.

        :param nlist: The Hoomd neighbor list that will be used as the TensorFlow input.
        :type nlist: Hoomd nlist
        :param r_cut: Cutoff radius for neighbor listing.
        :type r_cut: float
        :param period: How many Hoomd steps should pass before updating the TensorFlow model.
        :type period: int
        :param batch_size: The size of batches if we are using batching.
            Cannot be used if molecule-wise batching is active.
        :type batch_size: int
        :param train: Indicate if ``train_on_batch``
            Keras model method should be called at each step
            with the labels being Hoomd forces.
        :type train: bool
        :param save_output_period: How often to save output from ``model``.
            Each output is accessible after
            as attributes ``outputs`` as numpy arrays with a new axis at 0,
            representing each call. Note that
            if your model outputs forces or forces and virial, then
            these will not be present.
        :type save_output_period: int


        '''
        # make sure we're initialized, so we can have logging
        if not hoomd.init.is_initialized():
            raise RuntimeError('Must initialize hoomd first')

        self.enabled = True
        self.log = True
        self.cpp_force = None
        self.force_name = 'tfcompute'
        self.compute_name = self.force_name
        r_cut = float(r_cut)
        self.r_cut = r_cut
        self.batch_size = 0 if batch_size is None else batch_size
        self.mol_indices = None
        self.save_output_period = save_output_period
        self.outputs = None
        self._calls = 0
        self._output_offset = 0
        if self.model.output_forces:
            self._output_offset = 1
        if self.model.virial:
            self._output_offset = 2
        if train:
            # figure out from losses
            try:
                for i, l in enumerate(self.model.loss):
                    if l is None:
                        break
                self._output_offset = i
            except AttributeError:
                raise ValueError('SimModel has not been compiled')

        self.train = train

        if self.batch_size > 0:
            hoomd.context.msg.notice(2, 'Using fixed batching in htf\n')

        if issubclass(type(self.model), MolSimModel):
            if self.batch_size != 0:
                raise ValueError(
                    'Cannot batch by molecule and by batch_number')
            if hoomd.comm.get_num_ranks() > 1:
                raise ValueError('Molecular batches are '
                                 'not supported with spatial decomposition (MPI)')
            # Now we try to disable sorting
            c = hoomd.context.current.sorter
            if c is None:
                hoomd.context.msg.notice(1, 'Unable to disable molecular sorting.'
                                         'Make sure you disable it to allow molecular batching')
            else:
                c.disable()

        # get neighbor cutoff
        self.nneighbor_cutoff = self.model.nneighbor_cutoff

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

        # check if we are outputting forces
        self.force_mode_code = _htf.FORCE_MODE.hoomd2tf
        if self.model.output_forces:
            self.force_mode_code = _htf.FORCE_MODE.tf2hoomd
        hoomd.context.msg.notice(2, 'Force mode is {}'
                                 ' \n'.format(self.force_mode_code))
        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if hoomd.htf._tf_on_gpu:
                raise ValueError(
                    'Cannot run GPU/CPU mixed mode between TF and HOOMD.'
                    'You are running HOOMD on CPU and TF on GPU')
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
        else:
            if not hoomd.htf._tf_on_gpu:
                raise ValueError(
                    'Cannot run GPU/CPU mixed mode between TF and HOOMD.'
                    'You are running HOOMD on GPU and TF on CPU')
            self.cpp_force = \
                _htf.TensorflowComputeGPU(self,
                                          hoomd.context.current.system_definition,
                                          nlist.cpp_nlist if nlist is not None else None,
                                          r_cut,
                                          self.nneighbor_cutoff,
                                          self.force_mode_code,
                                          period,
                                          self.batch_size)
        # get double vs single precision
        self.dtype = tf.float32
        if self.cpp_force.isDoublePrecision():
            self.dtype = tf.float64

        # set this so disable works
        self.cpp_compute = self.cpp_force

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

    def set_reference_forces(self, *forces):
        R''' Sets the Hoomd reference forces to be used by TensorFlow.

        This allows you to choose which forces are used as the label
        for training.

        :param forces: Hoomd force objects
        '''
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
        # Define the cutoff radius used in the neighbor list.
        # Adapted from hoomd/md/pair.py
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

    def _finish_update(self, batch_index, batch_frac):
        ''' Allow TF to read output and we wait for it to finish.

        :param batch_index: index of batch to be processed
        :param batch_frac: fractional batch index, i.e.
            ``batch_frac`` = ``batch_index / len(input)``
        '''

        if batch_index == 0:
            self._calls += 1

        if not self.train:
            # compute model
            inputs = self.model.compute_inputs(
                self.dtype,
                self.cpp_force.getNlistBuffer(),
                self.cpp_force.getPositionsBuffer(),
                self.cpp_force.getBoxBuffer(),
                batch_frac)

            output = self.model(inputs, self.train)
            if self.save_output_period and self._calls % self.save_output_period == 0:
                if self.outputs is None:
                    self.outputs = [o.numpy()[np.newaxis, ...]
                                    for o in output[self._output_offset:]]
                else:
                    self.outputs = [
                        np.append(o1, o2.numpy()[np.newaxis, ...], axis=0)
                        for o1, o2 in zip(self.outputs, output[self._output_offset:])
                    ]
            # update forces
            if self.force_mode_code == _htf.FORCE_MODE.tf2hoomd:
                self.model.compute_outputs(
                    self.dtype, self.cpp_force.getForcesBuffer(),
                    self.cpp_force.getVirialBuffer(),
                    *output[:self._output_offset])
        else:
            # get inputs
            inputs = self.model.compute_inputs(
                self.dtype,
                self.cpp_force.getNlistBuffer(),
                self.cpp_force.getPositionsBuffer(),
                self.cpp_force.getBoxBuffer(),
                batch_frac,
                self.cpp_force.getForcesBuffer())

            # do we need to save output?
            if self.save_output_period and self._calls % self.save_output_period == 0:
                output = self.model(inputs[:-1], self.train)
                if self.outputs is None:
                    self.outputs = [o.numpy()[np.newaxis, ...]
                                    for o in output[self._output_offset:]]
                else:
                    self.outputs = [
                        np.append(o1, o2.numpy()[np.newaxis, ...], axis=0)
                        for o1, o2 in zip(self.outputs, output[self._output_offset:])
                    ]
            # now actually train
            self.model.train_on_batch(
                x=inputs[:-1],
                y=inputs[-1],
                reset_metrics=False)

    def get_positions_array(self):
        R''' Retrieve positions array as numpy array
        '''
        return self._scalar4_vec_to_np(self.cpp_force.getPositionsArray())

    def get_nlist_array(self):
        R''' Retrieve neighbor list array as numpy array
        '''
        nl = self._scalar4_vec_to_np(self.cpp_force.getNlistArray())
        return nl.reshape(-1, self.nneighbor_cutoff, 4)

    def get_forces_array(self):
        R''' Retrieve forces array as numpy array
        '''
        return self._scalar4_vec_to_np(self.cpp_force.getForcesArray())

    def get_virial_array(self):
        R''' Retrieve virial array as numpy array
        '''
        array = np.array(self.cpp_force.getVirialArray())
        return array.reshape((-1, 9))

    def update_coeffs(self):
        pass

    def _scalar4_vec_to_np(self, array):
        ''' Convert from scalar4 dtype to numpy array
        :param array: the scalar4 array to be processed
        '''
        npa = np.empty((len(array), 4))
        for i, e in enumerate(array):
            npa[i, 0] = e.x
            npa[i, 1] = e.y
            npa[i, 2] = e.z
            npa[i, 3] = e.w
        return npa
