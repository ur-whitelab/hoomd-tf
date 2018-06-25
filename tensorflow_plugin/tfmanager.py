import tensorflow as tf
import numpy as np
import tensorflow as tf
import sys, logging

def main(log_filename, lock, N, input_buffer, output_buffer):
    tfm = TFManager(lock, N, input_buffer, output_buffer, log_filename)

    tfm.start_loop()

class TFManager:
    def __init__(self, lock, N, output_buffer, input_buffer, log_filename):

        self.log = logging.getLogger('tensorflow')
        fh = logging.FileHandler(log_filename)
        self.log.addHandler(fh)
        self.log.setLevel(logging.INFO)
        print('Switching to logger')

        self.lock = lock
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.N = N

        self.log.info('Starting TF Session Manager. MMAP is at {:x}, {:x}'.format(id(input_buffer),id(output_buffer)))

    def _update(self, sess):
        print(sess.run(self.graph))

    def _build_graph(self):
        ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor
        #need to convert out scalar4 memory address to an integer
        #longlong should be int64
        address = id(self.input_buffer)
        self.log.info('initializing ipc_to_tensor at address {:x}'.format(address))
        self.graph = ipc_to_tensor(address=address, shape=self.N, T=tf.float32)

    def start_loop(self):

        self._build_graph()
        self.log.info('Constructed TF Model graph')
        with tf.Session() as sess:
            while True:                
                self.lock.acquire()
                self.log.info('Starting TF update...')
                self._update(sess)
                self.lock.release()




