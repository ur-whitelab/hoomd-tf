# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# this simple python interface just actiavates the c++ TensorflowCompute from cppmodule
# Check out any of the python code in lib/hoomd-python-module/hoomd_script for moreexamples

# First, we need to import the C++ module. It has the same name as this module (tensorflow_plugin) but with an underscore
# in front
from hoomd.tensorflow_plugin import _tensorflow_plugin

# Next, since we are extending an Compute, we need to bring in the base class Compute and some other parts from
# hoomd_script
from .tfmanager import main
import sys, math, numpy as np, pickle, multiprocessing, os
import hoomd, hoomd.md.nlist
import tensorflow as tf

## Zeroes all particle velocities
#
# Every \a period time steps, particle velocities are modified so that they are all zero
#
class tensorflow(hoomd.compute._compute):
    ## Initialize the velocity zeroe
    #
    # \param period Velocities will be zeroed every \a period time steps
    #
    # \b tensorflows:
    # \code
    # tensorflow_plugin.update.tensorflow()
    # zeroer = tensorflow_plugin.update.tensorflow(period=10)
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, tf_model_directory, nlist, r_cut, force_mode='overwrite', nneighbor_cutoff = 4, log_filename='tf_manager.log', debug_mode=False):

        #make sure we have number of atoms and know dimensionality, etc.
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error('Cannot create TF before initialization\n')
            raise RuntimeError('Error creating TF')

        #check our parameters
        try:
            with open(os.path.join(tf_model_directory, 'graph_info.p'), 'rb') as f:
                self.graph_info = pickle.load(f)
                if self.graph_info['N'] != len(hoomd.context.current.group_all):
                    hoomd.context.msg.error('Number of atoms must be same in TF model and HOOMD system\n')
                    raise RuntimeError('Error creating TF')
                if self.graph_info['NN'] != nneighbor_cutoff:
                    hoomd.context.msg.error('Number of neighbors to consider must be the same in TF model and HOOMD system\n')
                    raise RuntimeError('Error creating TF')
        except IOError:
            hoomd.context.msg.error('Unable to load model in directory {}'.format(tf_model_directory))
            raise RuntimeError('Error creating TF')

        #I'm not sure if this is necessary following other files
        self.enabled = True
        self.log = True
        self.cpp_force = None
        self.force_name = 'tfcompute'
        self.compute_name = self.force_name
        self.nneighbor_cutoff = nneighbor_cutoff
        self.tf_model_directory = tf_model_directory
        nlist.subscribe(self.rcut)
        self.r_cut = r_cut
        self.debug_mode = debug_mode

        hoomd.util.print_status_line()

        # initialize base class
        hoomd.compute._compute.__init__(self)
        self.tfm = None
        self.log_filename = log_filename

        force_mode_code = _tensorflow_plugin.FORCE_MODE.overwrite
        if force_mode == 'add':
            force_mode_code = _tensorflow_plugin.FORCE_MODE.add
        elif force_mode == 'none' or force_mode == 'ignore' or force_mode is None:
            force_mode_code = _tensorflow_plugin.FORCE_MODE.ignore

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _tensorflow_plugin.TensorflowCompute(self, hoomd.context.current.system_definition, nlist.cpp_nlist, nneighbor_cutoff, force_mode_code)
        else:
            self.cpp_force = _tensorflow_plugin.TensorflowComputeGPU(self, hoomd.context.current.system_definition, nlist.cpp_nlist, nneighbor_cutoff)

        #get double vs single precision
        self.dtype = tf.float32
        if self.cpp_force.is_double_precision():
            self.dtype = tf.double

        #adding to forces causes the computeForces method to be called.
        hoomd.context.current.system.addCompute(self.cpp_force, self.compute_name);
        hoomd.context.current.forces.append(self)

        self.restart_tf()

    def rcut(self):
        #adapted from hoomd/md/pair.py
        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes()
        type_list = []
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i))

        # update the rcut by pair type
        r_cut_dict = hoomd.md.nlist.rcut()
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                # get the r_cut value
                r_cut_dict.set_pair(type_list[i],type_list[j],self.r_cut)
        return r_cut_dict

    def __del__(self):
        if self.tfm and self.tfm.is_alive():
            self.shutdown_tf()

    def shutdown_tf(self):
        #need to terminate orphan
        hoomd.context.msg.notice(2, 'Shutting down TF Session Manager\n')
        self.barrier.abort()
        self.tfm.terminate()

    def restart_tf(self):
        if self.tfm and self.tfm.is_alive():
            self.shutdown_tf()
        if not self.cpp_force:
            return
        #setup locks
        self.lock = multiprocessing.Lock()
        #I can't figure out how to reliably get __del__ to be called,
        #so I set a timeout to clean-up TF manager.
        self.barrier = multiprocessing.Barrier(2, timeout=None if self.debug_mode else 3)
        self.tfm = multiprocessing.Process(target=main,
                                    args=(self.log_filename,
                                          self.graph_info,
                                          self.lock,
                                          self.barrier,
                                          self.cpp_force.get_positions_buffer(),
                                          self.cpp_force.get_nlist_buffer(),
                                          self.cpp_force.get_forces_buffer(),
                                          self.dtype,
                                          self.debug_mode))

        self.tfm.start()
        hoomd.context.msg.notice(2, 'Forked TF Session Manager. Will make tensor of shape {}x4\n'.format(len(hoomd.context.current.group_all)))


    def start_update(self):
        '''Write to output the current sys information'''
        self.lock.acquire()
        if not self.tfm.is_alive():
            hoomd.context.msg.error('TF Session Manager died. See its output log ({})'.format(self.log_filename))
            raise RuntimeError()

    def finish_update(self):
        '''Allow TF to read output and we wait for it to finish.'''
        self.lock.release()
        self.barrier.wait()

    def get_positions_array(self):
        return scalar4_vec_to_np(self.cpp_force.get_positions_array())

    def get_nlist_array(self):
        return scalar4_vec_to_np(self.cpp_force.get_nlist_array())

    def get_forces_array(self):
        return scalar4_vec_to_np(self.cpp_force.get_forces_array())

    def update_coeffs(self):
        pass

def scalar4_vec_to_np(array):
    '''TODO: This must exist somewhere in HOOMD codebase'''
    npa = np.empty((len(array), 4))
    for i, e in enumerate(array):
        npa[i,0] = e.x
        npa[i,1] = e.y
        npa[i,2] = e.z
        npa[i,3] = e.w
    return npa
