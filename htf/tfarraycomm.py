# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import hoomd.htf._htf as _htf
import numpy as np
import hoomd

R""" tf_array_comm is used to incorporate some
    native code into pytest framework
"""

class tf_array_comm:
    R""" Set up HOOMD context and interface with C++ memory.

    :param nparray: numpy array to share with the C++ process
    :param hoomd_context: HOOMD execution configuration
        (defaults to ``hoomd.context.exec_conf``)
    """
    def __init__(self, nparray, hoomd_context=hoomd.context.exec_conf):
        # get numpy array address
        ptr_address, _ = nparray.__array_interface__['data']
        self._size = len(nparray)
        assert len(nparray.shape) == 1
        if hoomd_context is None:
            hoomd.context.initialize()
            hoomd_context = hoomd.context.exec_conf
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_ref = _htf.TFArrayCommCPU(
                _htf.int2ptr(ptr_address))
        else:
            raise RuntimeError('Can only build TFArray Comm on CPU')

    ## \internal
    # \brief Send an array
    # See C++ bindings
    def send(self):
        R"""Sends the array to the C++ class instance."""
        self.cpp_ref.send()

    ## \internal
    # \brief Receive an array
    # See C++ bindings
    def receive(self):
        R"""Receives the array from the C++ class instance."""
        self.cpp_ref.receive()

    ## \internal
    # \brief Get this array's data
    # See C++ bindings
    def getArray(self):
        R"""Convert C++ array into a numpy array.

        :return: the converted numpy array"""
        npa = np.empty(self._size)
        array = self.cpp_ref.getArray()
        for i in range(self._size):
            npa[i] = array[i]
        return npa
