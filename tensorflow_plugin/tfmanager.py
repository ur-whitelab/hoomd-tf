import tensorflow as tf
import numpy as np
import tensorflow as tf
import sys, logging

def main(log_filename, lock, barrier, N, input_buffer, output_buffer):
    tfm = TFManager(lock, barrier, N, input_buffer, output_buffer, log_filename)

    tfm.start_loop()

class TFManager:
    def __init__(self, lock, barrier, N, output_buffer, input_buffer, log_filename):

        self.log = logging.getLogger('tensorflow')
        fh = logging.FileHandler(log_filename)
        self.log.addHandler(fh)
        self.log.setLevel(logging.INFO)

        self.lock = lock
        self.barrier = barrier
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.N = N

        self.log.info('Starting TF Session Manager. MMAP is at {:x}, {:x}'.format(id(input_buffer),id(output_buffer)))

    def _update(self, sess):
        sess.run(self.graph)

    def _build_graph(self):
        ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor
        tensor_to_ipc_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/tensor2ipc/lib_tensor2ipc_op.so')
        tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc
        #need to convert out scalar4 memory address to an integer
        #longlong should be int64
        self.log.info('initializing ipc_to_tensor at address {:x}'.format(self.input_buffer))
        self.log.info('initializing tensor_to_ipc at address {:x}'.format(self.output_buffer))
        input = ipc_to_tensor(address=self.input_buffer, size=self.N, T=tf.float32)
        output = tensor_to_ipc(input, address=self.output_buffer, size=self.N)
        self.graph = output

    def start_loop(self):

        self._build_graph()
        self.log.info('Constructed TF Model graph')
        with tf.Session() as sess:
            self._update(sess) #run once to force initialize 
            while True:
                self.barrier.wait()               
                self.lock.acquire()
                self.log.info('Starting TF update...')
                self._update(sess)
                self.lock.release()




