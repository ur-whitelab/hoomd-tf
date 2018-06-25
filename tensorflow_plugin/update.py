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
import sys, math

## Zeroes all particle velocities
#
# Every \a period time steps, particle velocities are modified so that they are all zero
#
class tensorflow(hoomd.update._updater):
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
        hoomd.update._updater.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = _tensorflow_plugin.TensorflowUpdater(hoomd.context.current.system_definition, self)
        else:
            self.cpp_updater = _tensorflow_plugin.TensorflowUpdaterGPU(hoomd.context.current.system_definition, self)

        #start tf manager
        self.lock = multiprocessing.Lock()
        #I can't figure out how to reliably get __del__ to be called,
        #so I set a timeout to clean-up TF manager.
        self.barrier = multiprocessing.Barrier(2, timeout=10)
        self.tfm = multiprocessing.Process(target=main,
                                    args=(log_filename,
                                          self.lock,
                                          self.barrier,
                                          len(hoomd.context.current.group_all),
                                          self.cpp_updater.get_input_buffer(),
                                          self.cpp_updater.get_output_buffer()))

        self.tfm.start()
        hoomd.context.msg.notice(2, 'Forked TF Session Manager. Will make tensor of shape {}x4\n'.format(len(hoomd.context.current.group_all)))
        self.setupUpdater(period)

    def __del__(self):
        #need to terminate orphan
        hoomd.context.msg.notice(2, 'Shutting down TF Session Manager\n')
        self.lock.release()
        self.barrier.abort()
        self.tfm.terminate()

    def start_update(self):
        '''Write to output the current sys information'''
        self.lock.acquire()

    def finish_update(self):
        '''Allow TF to read output and we wait for it to finish.'''
        self.lock.release()
        self.barrier.wait()