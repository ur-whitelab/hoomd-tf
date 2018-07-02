import _ipc_tester
import numpy as np

class IpcTester:
    def __init__(self, length):
        self.cpp_ref = _ipc_tester.IpcTester(self, length)

    def get_input_array(self):
        return scalar4_vec_to_np(self.cpp_ref.get_input_array())

    def get_output_array(self):
        return scalar4_vec_to_np(self.cpp_ref.get_output_array())

    def get_input_buffer(self):
        return self.cpp_ref.get_input_buffer()

    def get_output_buffer(self):
        return self.cpp_ref.get_output_buffer()

def scalar4_vec_to_np(array):
    npa = np.empty((len(array), 4))
    for i, e in enumerate(array):
        npa[i,0] = e.x
        npa[i,1] = e.y
        npa[i,2] = e.z
        npa[i,3] = e.w
    return npa