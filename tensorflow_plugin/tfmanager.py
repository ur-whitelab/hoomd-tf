import tensorflow as tf
import numpy as np
import tensorflow as tf
import sys, logging

def main(log_filename, graph, lock, barrier, N, NN, positions_buffer, nlist_buffer, forces_buffer):
    tfm = TFManager(lock, graph, barrier, N, NN, positions_buffer, nlist_buffer, forces_buffer, log_filename)

    tfm.start_loop()

class TFManager:
    def __init__(self, graph, lock, barrier, N, NN, positions_buffer, nlist_buffer, forces_buffer, log_filename):

        self.log = logging.getLogger('tensorflow')
        fh = logging.FileHandler(log_filename)
        self.log.addHandler(fh)
        self.log.setLevel(logging.INFO)

        self.lock = lock
        self.barrier = barrier
        self.positions_buffer = positions_buffer
        self.nlist_buffer = nlist_buffer
        self.forces_buffer = forces_buffer
        self.N = N
        self.nneighs = NN

        self.log.info('Starting TF Session Manager. MMAP is at {:x}, {:x}'.format(id(input_buffer),id(output_buffer)))
        self._prepare_graph(graph)

    def _update(self, sess):
        sess.run(self.graph)

    def _prepare_graph(self, graph_def):
        ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor
        tensor_to_ipc_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/tensor2ipc/lib_tensor2ipc_op.so')
        tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc
        #need to convert out scalar4 memory address to an integer
        #longlong should be int64
        self.log.info('initializing ipc_to_tensor at address {:x}'.format(self.positions_buffer))
        self.log.info('initializing ipc_to_tensor at address {:x}'.format(self.nlist_buffer))
        self.log.info('initializing tensor_to_ipc at address {:x}'.format(self.forces_buffer))
        self.positions = ipc_to_tensor(address=self.positions_buffer, size=self.N, T=tf.float32)
        self.nlist = ipc_to_tensor(address=self.nlist_buffer, size=self.nneighs * self.N, T=tf.float32)
        self.forces = tensor_to_ipc(input, address=self.forces_buffer, size=self.N)

        #now insert into graph
        try:
            self.graph = tf.import_graph_def(graph_def, input_map={"Forces:0": self.forces,
                                                               "Nlist:0": self.nlist,
                                                               "Positions:0" : self.positions})
        except ValueError:
            raise ValueError('Your graph must contain the following tensors: Forces:0, Nlist:0, Positions:0')

    def start_loop(self):

        self.log.info('Constructed TF Model graph')
        with tf.Session() as sess:
            self._update(sess) #run once to force initialize
            while True:
                self.barrier.wait()
                self.lock.acquire()
                self.log.info('Starting TF update...')
                self._update(sess)
                self.lock.release()




