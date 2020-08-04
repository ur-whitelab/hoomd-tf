# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

from hoomd.htf import _htf
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
# Integrates tensorflow into HOOMD-blue

class tfcompute(hoomd.compute._compute):
    R""" TensorFlow Computations for HTF.

        :param tf_model_directory: Kera Model
    """
    ## \internal
    # \brief Constructs the tfcompute class
    # \details
    # Initializes the tfcompute class with options to manage how and where TensorFlow saves,
    # whether to use a tensorboard, and some execution preferences.
    def __init__(self, model):
        R""" Initialize a tfcompute class instance
        """
        self.model = model

    def attach(self, nlist=None, r_cut=0, save_period=1000,
               period=1, mol_indices=None, max_molecule_size=None,
               batch_size=None,label_data=None):
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
        # make sure we're initialized, so we can have logging
        if not hoomd.init.is_initialized():
            raise RuntimeError('Must initialize hoomd first')

        self.enabled = True
        self.log = True
        self.cpp_force = None
        self.save_period = save_period
        self.force_name = 'tfcompute'
        self.compute_name = self.force_name
        r_cut = float(r_cut)
        self.r_cut = r_cut
        self.batch_size = 0 if batch_size is None else batch_size
        self.mol_indices = None

        if self.batch_size > 0:
            hoomd.context.msg.notice(2, 'Using fixed batching in htf\n')
        if mol_indices is not None:
            #TODO
            pass

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

    def finish_update(self, batch_index, batch_frac):
        R""" Allow TF to read output and we wait for it to finish.

        :param batch_index: index of batch to be processed
        :param batch_frac: fractional batch index, i.e.
            ``batch_frac`` = ``batch_index / len(input)``
        """
        if self.force_mode_code == _htf.FORCE_MODE.tf2hoomd:
            inputs = self.model.compute_inputs(
                self.dtype,
                self.cpp_force.getNlistBuffer(),
                self.cpp_force.getPositionsBuffer(),
                self.cpp_force.getBoxBuffer())

            output = self.model(inputs)
            self.model.update(batch_frac, batch_index)
            self.model.compute_outputs(self.dtype, self.cpp_force.getForcesBuffer(), output)
        else:
            inputs = self.model.compute_inputs(
                self.dtype,
                self.cpp_force.getNlistBuffer(),
                self.cpp_force.getPositionsBuffer(),
                self.cpp_force.getBoxBuffer(),
                self.cpp_force.getForcesBuffer())
            self.model.train_on_batch(x=inputs[:-1], y=inputs[-1])


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
