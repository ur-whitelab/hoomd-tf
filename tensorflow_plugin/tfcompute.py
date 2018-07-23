# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# this simple python interface just actiavates the c++ TensorflowCompute from cppmodule
# Check out any of the python code in lib/hoomd-python-module/hoomd_script for moreexamples

# First, we need to import the C++ module. It has the same name as this module (tensorflow_plugin) but with an underscore
# in front
from hoomd.tensorflow_plugin import _tensorflow_plugin

# Next, since we are extending an Compute, we need to bring in the base class Compute and some other parts from
# hoomd_script
import hoomd
import multiprocessing
from .tfmanager import main
import sys, math, numpy as np
import tensorflow as tf

## Zeroes all particle velocities
#
# Every \a period time steps, particle velocities are modified so that they are all zero
#
class tensorflow(hoomd.compute._compute):
    ## Initialize the velocity zeroer
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
    def __init__(self, tf_model_directory, nlist, nneighbor_cutoff = 4, log_filename='tf_manager.log'):

        #make sure we have number of atoms and know dimensionality, etc.
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create TF before initialization\n")
            raise RuntimeError('Error creating TF')

        #I'm not sure if this is necessary following other files
        self.enabled = True
        self.log = True
        self.cpp_force = None
        self.force_name = "tfcompute"
        self.compute_name = self.force_name
        self.nneighbor_cutoff = nneighbor_cutoff
        self.tf_model_directory = tf_model_directory

        hoomd.util.print_status_line()

        # initialize base class
        hoomd.compute._compute.__init__(self)
        self.tfm = None
        self.log_filename = log_filename

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _tensorflow_plugin.TensorflowCompute(hoomd.context.current.system_definition, nlist.cpp_nlist, self, nneighbor_cutoff)
        else:
            self.cpp_force = _tensorflow_plugin.TensorflowComputeGPU(hoomd.context.current.system_definition, nlist.cpp_nlist, self, nneighbor_cutoff)

        #adding to forces causes the computeForces method to be called.
        hoomd.context.current.system.addCompute(self.cpp_force, self.compute_name);
        hoomd.context.current.forces.append(self)

        self.restart_tf()

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
        self.barrier = multiprocessing.Barrier(2, timeout=3)
        self.tfm = multiprocessing.Process(target=main,
                                    args=(self.log_filename,
                                          self.tf_model_directory,
                                          self.lock,
                                          self.barrier,
                                          len(hoomd.context.current.group_all),
                                          self.nneighbor_cutoff,
                                          self.cpp_force.get_nlist_buffer(),
                                          self.cpp_force.get_positions_buffer(),
                                          self.cpp_force.get_forces_buffer()))

        self.tfm.start()
        hoomd.context.msg.notice(2, 'Forked TF Session Manager. Will make tensor of shape {}x4\n'.format(len(hoomd.context.current.group_all)))


    def start_update(self):
        '''Write to output the current sys information'''
        self.lock.acquire()
        if not self.tfm.is_alive():
            hoomd.context.msg.error("TF Session Manager died. See its output log ({})".format(self.log_filename))
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
