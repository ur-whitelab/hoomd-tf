# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# this simple python interface just actiavates the c++ TensorflowUpdater from cppmodule
# Check out any of the python code in lib/hoomd-python-module/hoomd_script for moreexamples

# First, we need to import the C++ module. It has the same name as this module (tensorflow_plugin) but with an underscore
# in front
from hoomd.tensorflow_plugin import _tensorflow_plugin

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from
# hoomd_script
import hoomd
import multiprocessing
from .tfmanager import main
import sys, math, numpy as np

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
    def __init__(self, period=1, log_filename='tf_manager.log'):
        hoomd.util.print_status_line()

        # initialize base class
        hoomd.compute._compute.__init__(self)
        self.tfm = None
        self.log_filename = log_filename

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_compute = _tensorflow_plugin.TensorflowUpdater(hoomd.context.current.system_definition, self)
        else:
            self.cpp_compute = _tensorflow_plugin.TensorflowUpdaterGPU(hoomd.context.current.system_definition, self)

        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name);

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
        if not self.cpp_compute:
            return
        #setup locks
        self.lock = multiprocessing.Lock()
        #I can't figure out how to reliably get __del__ to be called,
        #so I set a timeout to clean-up TF manager.
        self.barrier = multiprocessing.Barrier(2, timeout=10)
        self.tfm = multiprocessing.Process(target=main,
                                    args=(self.log_filename,
                                          self.lock,
                                          self.barrier,
                                          len(hoomd.context.current.group_all),
                                          self.cpp_compute.get_input_buffer(),
                                          self.cpp_compute.get_output_buffer()))

        self.tfm.start()
        hoomd.context.msg.notice(2, 'Forked TF Session Manager. Will make tensor of shape {}x4\n'.format(len(hoomd.context.current.group_all)))


    def start_update(self):
        '''Write to output the current sys information'''
        hoomd.context.msg.notice(2, 'FREAFRIJFERIOAFE\n')
        self.lock.acquire()

    def finish_update(self):
        '''Allow TF to read output and we wait for it to finish.'''
        self.lock.release()
        self.barrier.wait()

    def get_input_array(self):
        return scalar4_vec_to_np(self.cpp_compute.get_input_array())
    
    def get_output_array(self):
        return scalar4_vec_to_np(self.cpp_compute.get_output_array())

def scalar4_vec_to_np(array):
    '''TODO: This must exist somewhere in HOOMD codebase'''
    npa = np.empty((len(array), 4))
    for i, e in enumerate(array):
        npa[i,0] = e.x
        npa[i,1] = e.y
        npa[i,2] = e.z
        npa[i,3] = e.w
    return npa
    