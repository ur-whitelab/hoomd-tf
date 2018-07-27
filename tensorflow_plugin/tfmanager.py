import tensorflow as tf
import numpy as np
import sys, logging, os, pickle

def main(log_filename, graph_info, lock, barrier, positions_buffer, nlist_buffer, forces_buffer, dtype, debug=True):
    tfm = TFManager(graph_info, lock,  barrier, positions_buffer, nlist_buffer, forces_buffer, log_filename, dtype, debug)

    tfm.start_loop()

def load_op_library(op):
    import hoomd.tensorflow_plugin
    path = hoomd.tensorflow_plugin.__path__[0]
    try:
        mod = tf.load_op_library(os.path.join(path, op, 'lib_{}_op.so'.format(op)))
    except IOError:
        raise IOError('Unable to load OP {}'.format(op))
    return mod


class TFManager:
    def __init__(self, graph_info, lock, barrier, positions_buffer, nlist_buffer, forces_buffer, log_filename, dtype, debug):

        self.log = logging.getLogger('tensorflow')
        fh = logging.FileHandler(log_filename)
        self.log.addHandler(fh)
        self.log.setLevel(logging.INFO)

        self.lock = lock
        self.barrier = barrier
        self.positions_buffer = positions_buffer
        self.nlist_buffer = nlist_buffer
        self.forces_buffer = forces_buffer
        self.debug = debug
        self.step = 0
        self.graph_info = graph_info
        self.dtype = dtype

        self.log.info('Starting TF Session Manager. MMAP is at {:x}, {:x}. Dtype is {}'.format(id(positions_buffer),id(forces_buffer), dtype))
        self.model_directory = self.graph_info['model_directory']
        self.N = self.graph_info['N']
        self.nneighs = self.graph_info['NN']

        self._prepare_graph()
        if graph_info['output_forces']:
            self.log.info('This TF Graph can modify forces.')
            self._prepare_forces()
        else:
            self.log.info('This TF Graph will not modify forces.')
            self.out_node = tf.get_default_graph().get_tensor_by_name(self.graph_info['out_node'])



    def _update(self, sess):
        runs = [tf.Print(self.out_node, [self.out_node], summarize=44)]
        #for i in tf.get_default_graph().get_operations():
        #    print(i.name)
        #pf = tf.get_default_graph().get_tensor_by_name('force-gradient/nlist-pairwise-force-gradient:0')
        #runs += [tf.Print(pf, [pf], summarize=288)]
        if self.debug:
            runs += [self.summaries]
        result = sess.run(runs)
        if self.debug:
            self.tb_writer.add_summary(result[-1], self.step)
        self.step += 1

    def _prepare_graph(self):
        ipc_to_tensor_module = load_op_library('ipc2tensor')
        ipc_to_tensor = ipc_to_tensor_module.ipc_to_tensor

        self.log.info('initializing  positions ipc_to_tensor at address {:x} with size {} x 4'.format(self.positions_buffer, self.N))
        self.log.info('initializing nlist ipc_to_tensor at address {:x} with size {} x 4'.format(self.nlist_buffer, self.nneighs * self.N))
        self.positions = ipc_to_tensor(address=self.positions_buffer, shape=[self.N, 4], T=self.dtype, name='positions-input')
        self.nlist = ipc_to_tensor(address=self.nlist_buffer, shape=[self.N, self.nneighs, 4], T=self.dtype, name='nlist-input')

        input_map = {self.graph_info['nlist']: self.nlist, self.graph_info['positions'] : self.positions}

        if not self.graph_info['output_forces']:
            #if the graph outputs forces
            self.log.info('initializing nlist ipc_to_tensor at address {:x} with size {} x 4'.format(self.nlist_buffer, self.nneighs * self.N))
            self.forces = ipc_to_tensor(address=self.forces_buffer, shape=[self.N, 4], T=self.dtype, name='forces-input')
            input_map[self.graph_info['forces']] = self.forces

        #now cast if graph dtype are different
        if self.graph_info['dtype'] != self.dtype:
            self.positions = tf.cast(self.positions, self.graph_info['dtype'])
            self.nlist = tf.cast(self.nlist, self.graph_info['dtype'])

        #now insert into graph
        try:
            self.saver = tf.train.import_meta_graph(os.path.join(self.model_directory,'model.meta'), input_map=input_map)
        except ValueError:
            raise ValueError('Your graph must contain the following tensors: forces, nlist, positions')

    def _prepare_forces(self):
        #insert the output forces
        try:
            out = tf.get_default_graph().get_tensor_by_name(self.graph_info['forces'])
            #make sure forces will be output in correct precision to hoomd
            self.forces = tf.cast(out, self.dtype)
        except ValueError:
            raise ValueError('Your graph must contain the following tensors: forces, nlist, positions')
        tensor_to_ipc_module = load_op_library('tensor2ipc')
        tensor_to_ipc = tensor_to_ipc_module.tensor_to_ipc
        self.out_node = tensor_to_ipc(out, address=self.forces_buffer, maxsize=self.N * 4)
        self.log.info('initializing tensor_to_ipc at address {:x}'.format(self.forces_buffer))

    def _attach_tensorboard(self, sess):

        tf.summary.histogram('forces', self.forces)

        self.summaries = tf.summary.merge_all()
        self.tb_writer = tf.summary.FileWriter(os.path.join(self.model_directory, 'tensorboard'),
                                      sess.graph)
        tf.global_variables_initializer()


    def start_loop(self):

        self.log.info('Constructed TF Model graph')
        with tf.Session() as sess:
            #resore model checkpoint
            self.saver.restore(sess, tf.train.latest_checkpoint(self.model_directory))
            if self.debug:
                from tensorflow.python import debug as tf_debug
                #sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
                self._attach_tensorboard(sess)
                self.log.info('You must (first!) attach tensorboard by running '
                            'tensorboard --logdir {} --debugger_port 6064'
                            .format(os.path.join(self.model_directory, 'tensorboard')))
            while True:
                self.barrier.wait()
                self.lock.acquire()
                self._update(sess)
                self.lock.release()




