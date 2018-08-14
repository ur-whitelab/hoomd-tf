import _ipc_array_comm
import numpy as np
import hoomd

class IPCArrayComm:
    def __init__(self, nparray, hoomd_context = hoomd.context.exec_conf):
        #get numpy array address
        ptr_address, _ = nparray.__array_interface__['data']
        self._size = len(nparray)
        assert len(nparray.shape) == 1
        if hoomd_context is None:
            hoomd.context.initialize()
            hoomd_context = hoomd.context.exec_conf
        self.cpp_ref = _ipc_array_comm.IPCArrayCommCPU(_ipc_array_comm.int2ptr(ptr_address), nparray.dtype.itemsize * len(nparray), hoomd_context)

    def send(self):
        self.cpp_ref.send()

    def receive(self):
        self.cpp_ref.receive()

    def getArray(self):
        npa = np.empty( self._size )
        array = self.cpp_ref.getArray()
        for i in range(self._size):
            npa[i] = array[i]
        return npa


if __name__ == '__main__':
    print('testing...')
    shared = np.zeros(10)
    ipc = IPCArrayComm(shared)

    np.testing.assert_allclose(shared, ipc.getArray())

    shared[4] = 10.0
    ipc.receive()
    np.testing.assert_allclose(shared, ipc.getArray())

    ref = shared[:]
    shared[:] = -1
    ipc.send()
    np.testing.assert_allclose(shared, ref)
    print('all passed')