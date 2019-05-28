# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

from hoomd.htf import _htf
import numpy as np
import hoomd


class tf_array_comm:
    '''tf_array_comm is used to incorporate some
    native code into pytest framework
    '''
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

    def send(self):
        self.cpp_ref.send()

    def receive(self):
        self.cpp_ref.receive()

    def getArray(self):
        npa = np.empty(self._size)
        array = self.cpp_ref.getArray()
        for i in range(self._size):
            npa[i] = array[i]
        return npa
