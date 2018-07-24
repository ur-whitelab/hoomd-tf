import tensorflow as tf
import numpy as np
import tensorflow as tf
import sys, logging, os

def main(log_filename, model_directory, lock, barrier, N, NN, positions_buffer, nlist_buffer, forces_buffer):
    tfm = TFManager(model_directory, lock,  barrier, N, NN, positions_buffer, nlist_buffer, forces_buffer, log_filename)

    tfm.start_loop()

class TFManager:
    def __init__(self, model_directory, lock, barrier, N, NN, positions_buffer, nlist_buffer, forces_buffer, log_filename):

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
        self.visualize = True
        self.step = 0

        self.log.info('Starting TF Session Manager. MMAP is at {:x}, {:x}'.format(id(positions_buffer),id(forces_buffer)))
        self.model_directory = model_directory
        self._prepare_graph()
        self._prepare_forces()


    def _update(self, sess):
        runs = [self.forces]
        if self.visualize:
            runs += [self.summaries]
        result = sess.run(runs)
        if self.visualize:
            print('writing ', self.step)
            self.tb_writer.add_summary(result[-1], self.step)
        self.step += 1

    def _prepare_graph(self):
        ipc_to_tensor_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/ipc2tensor/lib_ipc2tensor_op.so')
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor
        #need to convert out scalar4 memory address to an integer
        #longlong should be int64
        self.log.info('initializing ipc_to_tensor at address {:x} with size {} x 4'.format(self.positions_buffer, self.N))
        self.log.info('initializing ipc_to_tensor at address {:x} with size {} x 4'.format(self.nlist_buffer, self.nneighs * self.N))
        self.positions = ipc_to_tensor(address=self.positions_buffer, size=self.N, T=tf.float32, name='positions-input')
        self.nlist = ipc_to_tensor(address=self.nlist_buffer, size=self.nneighs * self.N, T=tf.float32, name='nlist-input')
        #now insert into graph
        try:
            self.saver = tf.train.import_meta_graph(os.path.join(self.model_directory,'model.meta'), input_map={'nlist:0': self.nlist,
                                                               'positions:0' : self.positions})
        except ValueError:
            raise ValueError('Your graph must contain the following tensors: forces:0, nlist:0, positions:0')

    def _prepare_forces(self):
        #insert the output forces
        try:
            out = tf.get_default_graph().get_tensor_by_name('forces:0')

        except ValueError:
            raise ValueError('Your graph must contain the following tensors: forces:0, nlist:0, positions:0')
        tensor_to_ipc_module = tf.load_op_library('/srv/hoomd-blue/build/hoomd/tensorflow_plugin/tensor2ipc/lib_tensor2ipc_op.so')
        tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc
        self.forces = tensor_to_ipc(out, address=self.forces_buffer, size=self.N)
        self.log.info('initializing tensor_to_ipc at address {:x}'.format(self.forces_buffer))

    def _attach_tensorboard(self, sess):

        tf.summary.histogram('forces', tf.get_default_graph().get_tensor_by_name('forces:0'))

        self.summaries = tf.summary.merge_all()
        self.tb_writer = tf.summary.FileWriter(self.model_directory + '/tensorboard',
                                      sess.graph)
        self.visualize = True


    def start_loop(self):

        self.log.info('Constructed TF Model graph')
        with tf.Session() as sess:
            #resore model checkpoint
            print(self.model_directory)
            self.saver.restore(sess, tf.train.latest_checkpoint(self.model_directory))
            print(tf.get_default_graph().get_tensor_by_name('nlist:0'))
            self._attach_tensorboard(sess)
            self._update(sess) #run once to force initialize
            while True:
                self.barrier.wait()
                self.lock.acquire()
                self._update(sess)
                self.lock.release()




