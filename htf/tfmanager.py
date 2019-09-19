# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import tensorflow as tf
import numpy as np
import sys
import logging
import os
import pickle
import cProfile
import queue
import time

saver_args = {'max_to_keep': 1000}


def main(q, write_tensorboard=False, profile=False):

    tfm_args = q.get()
    tfm = TFManager(q=q,
                    write_tensorboard=write_tensorboard, **tfm_args)
    if(profile):
        cProfile.runctx('tfm.start_loop()', globals(),
                        locals(), filename='tf_profile.out')
    else:
        tfm.start_loop()


def load_op_library(op):
    import hoomd.htf
    path = hoomd.htf.__path__[0]
    try:
        op_path = os.path.join(path, op, 'lib_{}'.format(op))
        if os.path.exists(op_path + '.so'):
            op_path += '.so'
        elif os.path.exists(op_path + '.dylib'):
            op_path += '.dylib'
        else:
            raise OSError()
        mod = tf.load_op_library(op_path)
    except OSError:
        raise OSError('Unable to load OP {}. '
                      'Expected to be in {}'.format(op, path))
    return mod

## \internal
# \brief TensorFlow manager class
# \details
# Manages when TensorFlow performs training and coordinates its
# communication with HOOMD
class TFManager:
    ## \internal
    # \brief Constructs the tfmanager class
    # \details
    # Sets up the TensorFlow graph and sets any placeholder vars.
    # Handles updating and saving of TensorFlow graph, and passing
    # output forces back to HOOMD, as well as writing tensorboards.
    def __init__(self, graph_info, device, q,
                 positions_buffer, nlist_buffer,
                 forces_buffer, box_buffer, virial_buffer, log_filename,
                 dtype, debug, write_tensorboard, use_feed,
                 bootstrap, primary, bootstrap_map,
                 save_period, use_xla):
        R""" Initialize an instance of TFManager.

        Parameters
        ----------
        graph_info
            The structure of the TensorFlow graph, passed as a dict.
            See tfcompute.py
        device
            Which device to run on.
            See tfcompute.py
        q
            Threading queue that is used during execution of TensorFlow.
        positions_buffer
            Buffer where particle positions are stored
        nlist_buffer
            Address of the neighbor list tensor
        box_buffer
            Address of the box tensor
        forces_buffer
            Address of the forces tensor
        virial_buffer
            Address of the virial tensor
        log_filename
            Name of the file to output tensorflow logs
        dtype
            Data type for tensor values, e.g. int32, float32, etc
        debug
            True to run TensorFlow in debug mode
        write_tensorboard
            Whether to output a tensorboard file
        use_feed
            Whether or not to use a feed dictionary of tensor values
        bootstrap
            Location of previously-trained model files to load, otherwise None
        primary
            Whether this is the 'primary' instance of TFManager. Only one instance
            writes logs and saves model files.
        bootstrap_map
            A dictionary to be used when bootstrapping, pairing old models' tensor variable
            names with new ones. Key is new name, value is older model's.
        save_period
            How often to save the TensorFlow data. Period here is measured by
            how many times the TensorFLow model is updated. See tfcompute.py.
        use_xla
            If True, enables the accelerated linear algebra library in TensorFlow, which
            can be useful for large and complicated tensor operations.
        """
        self.primary = primary
        self.log = logging.getLogger('tensorflow')
        if not primary:
            self.log.disabled = True
        else:
            fh = logging.FileHandler(log_filename)
            self.log.addHandler(fh)
            self.log.setLevel(logging.INFO)

        self.device = device
        self.q = q
        self.positions_buffer = positions_buffer
        self.nlist_buffer = nlist_buffer
        self.forces_buffer = forces_buffer
        self.virial_buffer = virial_buffer
        self.box_buffer = box_buffer
        self.debug = debug
        self.step = -1
        self.graph_info = graph_info
        self.dtype = dtype
        self.write_tensorboard = write_tensorboard
        self.use_feed = use_feed
        self.save_period = save_period
        self.bootstrap = bootstrap
        self.bootstrap_map = bootstrap_map
        self.model_directory = self.graph_info['model_directory']
        self.nneighs = self.graph_info['NN']
        self.out_nodes = []
        self.summaries = None
        self.use_xla = use_xla
        self._prepare_graph()
        if graph_info['output_forces']:
            self.log.log(8, 'This TF Graph can modify forces.')
            self._prepare_forces()
        else:
            self.log.log(8, 'This TF Graph will not modify forces.')

        self.log.log(logging.INFO, 'The following quantities will computed:')
        self.log.log(logging.INFO, '\tname period batch')
        for node in self.graph_info['out_nodes']:
            node_attr = [None, 1, None]
            if isinstance(node, list):
                n = node[0]
                node_attr[1:len(node)] = node[1:]
            else:
                n = node
            try:
                name = tf.get_default_graph(
                        ).get_tensor_by_name(n)
            except ValueError:
                name = tf.get_default_graph(
                        ).get_operation_by_name(n)
            node_attr[0] = name
            self.out_nodes.append(node_attr)
            self.log.log(logging.INFO, '\t {} {} {}'.format(node_attr[0].name, 
                node_attr[1], node_attr[2]))

    ## \var primary
    # \internal
    # \brief Whether or not this is the 'primary' thread
    # \details
    # It is only necessary for one thread to save model files and
    # write logs, so the 'primary' thread is the one assigned to  do this.

    ## \var log
    # \internal
    # \brief The logging object to use
    # \details
    # This is set to be the 'tensorflow' logger

    ## \var device
    # \internal
    # \brief Which device Tensorflow is running on
    # \details
    # When GPU execution is enabled and multiple GPU devices are available,
    # this specifies which GPU to use. Otherwise it will default to the only
    # available GPU, or the CPU if GPU execution is disabled.

    ## \var q
    # \internal
    # \brief Threading queue that is used during execution of TensorFlow.

    ## \var positions_buffer
    # \internal
    # \brief The memory address of the particle positions
    # \details
    # This is needed for conversion between HOOMD and TensorFlow memory spaces

    ## \var nlist_buffer
    # \internal
    # \brief The memory address of the neighbor lists (pairwise distances)
    # \details
    # This is needed for conversion between HOOMD and TensorFlow memory spaces

    ## \var box_buffer
    # \internal
    # \brief The memory address of the box dimensions
    # \details
    # This is needed for conversion between HOOMD and TensorFlow memory spaces

    ## \var forces_buffer
    # \internal
    # \brief The memory address of the forces tensor
    # \details
    # This is needed for conversion between HOOMD and TensorFlow memory spaces

    ## \var virial_buffer
    # \internal
    # \brief The memory address of the virial
    # \details
    # This is needed for conversion between HOOMD and TensorFlow memory spaces

    ## \var debug
    # \internal
    # \brief Whether to print debug messages

    ## \var step
    # \internal
    # \brief Which timestep we are currently looking at
    # \details
    # This is tracked in order to print logs and save model files preiodically

    ## \var graph_info
    # \internal
    # \brief Dictionary containing structural information about the graph model
    # \details
    # NN: maximum number of neighbors per particle
    # mol_indices: incides of molecules in the system
    # MN: number of particles per molecule
    # output_forces: whether this model outputs forces to HOOMD
    # model_directory: where this model file was written
    # out_nodes: list of tensors that are output by the model
    # dtype: data type of the model's data
    # nlist: the neighbor list tensor
    # positions: the positoins tensor
    # forces: the forces tensor
    # virial: the virial tensor

    ## \var dtype
    # \internal
    # \brief The data type used by the tensors in this model

    ## \var write_tensorboard
    # \internal
    # \brief Whether to write a tensorboard file or not
    # \details
    # Tensorboard files can be loaded to visualize a trained TensorFlow model.
    # Set this to True to save one in the model directory.

    ## \var use_feed
    # \internal
    # \brief Whether to use a feed dictionary for initial tensor values

    ## \var save_period
    # \internal
    # \brief How often to save the TensorFlow model parameters
    # \details
    # This is how many training periods to wait before saving.
    # e.g. if we train TensorFlow parameters every 100 steps and save_period
    # is set to 200, then parameters are saved every 20000 HOOMD timesteps.

    ## \var bootstrap
    # \internal
    # \brief The name of the bootstrap directory
    # \details
    # The bootstrap directory should contain saved model files from a previously
    # trained TensorFlow model.

    ## \var bootstrap_map
    # \internal
    # \brief Map from one model's tensor names to another
    # \details
    # A dictionary to be used when bootstrapping, pairing old models' tensor variable
    # names with new ones. Key is new name, value is older model's.

    ## \var model_directory
    # \internal
    # \brief Where to save model parameters
    # \details
    # The directory name to which TensorFlow will write saved model files

    ## \var nneighs
    # \internal
    # \brief Max number of neighbors per particle

    ## \var out_nodes
    # \internal
    # \brief List of output tensors of the TensorFlow model
    # \details
    # TensorFlow output consists of tensors, which must be listed here.

    ## \var summaries
    # \internal
    # \brief List of tensors to summarize
    # \details
    # TensorFlow can summarize the change in specified tensor values during training.
    # These tensors are specified here and can be optionally viewed with tensorboard.

    ## \var use_xla
    # \internal
    # \brief Whether to use the accelerated linear algebra library
    # \details
    # XLA can enhance the speed of execution of many TensorFlow models. Set this
    # to True to use it.

    R""" update the TensorFlow model.

    Parameters
    ----------
    sess
        TensorFlow session instance. This is how TensorFlow updates are called.
    feed_dict
        The dictionary keyed by tensor names and filled with corresponding values.
        See feed_dict in tfcompute.__init__.
    batch_index
        Tracks the batch indices used for execution when batching. See graphbuilder.py
    """
    def _update(self, sess, feed_dict, batch_index):
        if batch_index == 0:
            # step starts at -1, so first step is 0
            self.step += 1
        run_nodes = [node[0] for node in self.out_nodes 
            if self.step % node[1] == 0 and (node[2] is None or node[2] == batch_index)]
        result = sess.run(run_nodes, feed_dict=feed_dict)
        # only save on the first batch.
        if self.step % self.save_period == 0 and batch_index == 0:
            self._save_model(sess, result[-1] if self.summaries is not None else None)

        return result
    R""" save model method is called during update
    Saves TensorFlow model parameters."""
    def _save_model(self, sess, summaries=None):

        if not self.primary:
            return

        if self.saver is not None:
            self.log.log(8, 'Writing {} variables at TF step {}'.format(
                    len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES
                                          )), self.step))
            self.saver.save(sess, os.path.join(self.model_directory, 'model'),
                            global_step=self.step)
        if self.write_tensorboard and summaries is not None:
            self.log.log(8, 'Writing tensorboard at TF step {}'.format(
                    self.step))
            # last out_node should be merged summary (set in
            #  _attach_tensorboard)
            self.tb_writer.add_summary(summaries, self.step)
            self.tb_writer.flush()

    R""" The prepare graph method prepares the TensorFlow graph to execute.
    First transfers the HOOMD data to TensorFlow control, then gets the device
    ready to execute and inserts the necessary nodes into the TensorFlow execution
    graph."""
    def _prepare_graph(self):
        hoomd_to_tf_module = load_op_library('hoomd2tf_op')
        hoomd_to_tf = hoomd_to_tf_module.hoomd_to_tf

        with tf.device(self.device):
            self.positions = hoomd_to_tf(
                address=self.positions_buffer, shape=[4],
                T=self.dtype, name='positions-input')
            self.box = hoomd_to_tf(
                address=self.box_buffer, shape=[3],
                T=self.dtype, name='box-input')
            self.nlist = tf.reshape(hoomd_to_tf(address=self.nlist_buffer,
                                                shape=[self.nneighs * 4],
                                                T=self.dtype,
                                                name='nlist-input'),
                                    [-1, self.nneighs, 4])
            self.log.log(10, 'initialized positions hoomd_to_tf at address'
                         ' {:x} with shape {} on {}'
                         .format(self.positions_buffer,
                                 self.positions.shape,
                                 self.device))
            self.log.log(10, 'initialized box hoomd_to_tf at address'
                         ' {:x} with shape {} on {}'
                         .format(self.box_buffer,
                                 self.box.shape,
                                 self.device))
            self.log.log(10, 'initialized nlist hoomd_to_tf at address {:x}'
                         'with shape {} on {}'.format(self.nlist_buffer,
                                                      self.nlist.shape,
                                                      self.device))
        # now cast if graph dtype are different
        if self.graph_info['dtype'] != self.dtype:
            with tf.device(self.device):
                self.positions = tf.cast(self.positions,
                                         self.graph_info['dtype'])
                self.box = tf.cast(self.box, self.graph_info['dtype'])
                self.nlist = tf.cast(self.nlist, self.graph_info['dtype'])

        input_map = {self.graph_info['nlist']: self.nlist,
                     self.graph_info['box']: self.box,
                     self.graph_info['positions']: self.positions}

        if not self.graph_info['output_forces']:
            # if the graph outputs forces, add new node
            with tf.device(self.device):
                self.forces = hoomd_to_tf(address=self.forces_buffer,
                                          shape=[4], T=self.dtype,
                                          name='forces-input')
                self.log.log(10, 'initialized forces hoomd_to_tf at address'
                             ' {:x} with shape {} on {}'
                             .format(self.forces_buffer,
                                     self.forces.shape,
                                     self.device))
            if self.graph_info['dtype'] != self.dtype:
                self.forces = tf.cast(self.forces, self.graph_info['dtype'])
                input_map[self.graph_info['forces']] = self.forces

        # now insert into graph
        try:
            self.graph = tf.train.import_meta_graph(os.path.join(
                    self.model_directory, 'model.meta'), input_map=input_map,
                                                    import_scope='')
        except ValueError:
            raise ValueError('Your graph ({}) must contain the'
                             ' following tensors: forces, nlist, position'
                             's'.format(os.path.join(self.model_directory,
                                                     'model.meta')))

    R""" The prepare forces method readies the force tensor for HOOMD
    Ensures that the forces and virial are the right dtypes, then shares
    the memory of the TensorFlow forces with HOOMD."""
    def _prepare_forces(self):
        # insert the output forces
        try:
            out = tf.get_default_graph().get_tensor_by_name(
                self.graph_info['forces'])
            # make sure forces will be output in correct precision to hoomd
            self.forces = tf.cast(out, self.dtype)
            if self.graph_info['virial'] is not None:
                out = tf.get_default_graph().get_tensor_by_name(
                    self.graph_info['virial'])
                # make sure forces will be output in correct precision to hoomd
                self.virial = tf.cast(out, self.dtype)
            else:
                self.log.warning('No virial computed in graph.'
                                 ' Pressure may be inaccurate!')
        except ValueError:
            raise ValueError('Your graph must contain the following'
                             ' tensors: forces, nlist, positions')
        tf_to_hoomd_module = load_op_library('tf2hoomd_op')
        tf_to_hoomd = tf_to_hoomd_module.tf_to_hoomd
        with tf.device(self.device):
            self.out_nodes.append([tf_to_hoomd(
                    self.forces, address=self.forces_buffer), 1, None])
            self.log.log(10, 'initialized forces tf_to_hoomd at address {:x}'
                         ' with shape {} on {}'.format(self.forces_buffer,
                                                       self.forces.shape,
                                                       self.device))
        if self.graph_info['virial'] is not None:
            # virial is Nx3x3
            with tf.device(self.device):
                self.out_nodes.append([tf_to_hoomd(
                        self.virial, address=self.virial_buffer), 1, None])
                self.log.log(10, 'initialized virial tf_to_hoomd at address'
                             ' {:x} with shape {} on {}'
                             .format(self.virial_buffer,
                                     self.virial.shape,
                                     self.device))

    R""" Attach tensorboard to our TensorFlow session.
    """
    def _attach_tensorboard(self, sess):

        self.summaries = tf.summary.merge_all()
        self.out_nodes.append([self.summaries, self.save_period, 0])
        self.tb_writer = tf.summary.FileWriter(os.path.join(
                self.model_directory, 'tensorboard'),
                                               sess.graph)

    R""" start_loop method of tfmanager.
    Prepares GPU for execution, gathers trainable variables,
    sets up model saving, loads pre-trained variables, sets
    up tensorboard if requested and parses feed_dicts.
    """
    def start_loop(self):

        self.log.log(10, 'Constructed TF Model graph')
        # make it grow as memory is needed instead of consuming all
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        if self.use_xla:
            config.graph_options.optimizer_options.global_jit_level = \
                tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess:
            # restore model checkpoint if there are variables
            if len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) > 0:
                # first initialize
                self.log.log(10, 'Found trainable variables...')
                sess.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))
                self.log.log(10, 'Trainable vars initialized')
                self.saver = tf.train.Saver(**saver_args)
                if self.bootstrap is not None:
                    checkpoint = tf.train.latest_checkpoint(self.bootstrap)
                    if checkpoint is None:
                        raise ValueError(
                            'Could not find '
                            'bootstrap checkpoint'
                            ' {}'.format(self.bootstrap))
                    self.log.log(8, 'Using bootstrap checkpoint'
                                 ' {}'.format(self.bootstrap))
                    # only load vars in the checkpoint and the graph!
                    cp = tf.train.NewCheckpointReader(checkpoint)
                    var_to_shape_map = cp.get_variable_to_shape_map()
                    var_list = var_to_shape_map.keys()
                    # convert bootstrap map values into actual variables
                    variable_map = None
                    if self.bootstrap_map is not None:
                        variables = var_list
                        variable_map = dict()
                        for vname, k in self.bootstrap_map.items():
                            value = None
                            for v in variables:
                                print(k, v, vname + ':0')
                                if v == vname:
                                    value = tf.get_default_graph().get_tensor_by_name(k + ':0')
                            if value is None:
                                raise ValueError(
                                    'Could not find variable'
                                    ' {} in graph while'
                                    ' processing'
                                    ' bootstrap_map'.format(vname))
                            variable_map[vname] = value
                        bootstrap_saver = tf.train.Saver(variable_map,
                                                     **saver_args)
                    else:
                        # remove vars that aren't in our graph
                        filtered_varlist = []
                        for v in var_to_shape_map.keys():
                            try:
                                t = tf.get_default_graph().get_tensor_by_name(v + ':0')
                                filtered_varlist.append(t)
                            except KeyError:
                                pass
                        bootstrap_saver = tf.train.Saver(filtered_varlist, **saver_args)
                    bootstrap_saver.restore(sess, checkpoint)
                else:
                    checkpoint = tf.train.latest_checkpoint(
                        self.model_directory)
                    if checkpoint is not None:
                        self.saver.restore(sess, checkpoint)
            else:
                self.saver = None
                if self.bootstrap is not None:
                    raise ValueError('Passed in a bootstrap'
                                     ' file to a non-trainable graph')
            if self.debug:
                from tensorflow.python import debug as tf_debug
                sess = tf_debug.TensorBoardDebugWrapperSession(
                    sess, 'localhost:6064')
                self.log.log(6, 'You must (first!) attach tensorboard by'
                             ' running'
                             ' tensorboard --logdir {} --debugger_port 6064'
                             .format(os.path.join(self.model_directory,
                                                  'tensorboard')))
            if self.write_tensorboard:
                self._attach_tensorboard(sess)
            # indicating we are ready to begin
            self.log.log(10, 'Completed TF Set-up')
            self.q.task_done()
            cumtime = 0
            processing_cumtime = 0
            result = None
            feed_dict = None
            while True:
                try:
                    raw_feed_dict = self.q.get()
                    if raw_feed_dict is None:
                        self.log.info('Empty Queue')
                        raise queue.Empty()
                except queue.Empty:
                    self.log.info('Received exit. Leaving TF Update Loop. ')
                    self.log.info('TF Update running'
                                  'time is {}'.format(cumtime))
                    self.log.info('TF Feed Processing'
                                  'time is {}'.format(processing_cumtime))
                    self.log.info('TF Total Time'
                                  '(excluding communication)'
                                  ' is {}'
                                  .format(processing_cumtime + cumtime))
                    self._save_model(sess)
                    break

                try:
                    # convert name keys to actual
                    # tensor keys if we're using a feed_dict
                    # from user
                    last_clock = time.perf_counter()
                    feed_dict = dict()
                    bi = raw_feed_dict['htf-batch-index:0']
                    for k, v in raw_feed_dict.items():
                        tensor = tf.get_default_graph().get_tensor_by_name(k)
                        feed_dict[tensor] = v
                    processing_cumtime += (time.perf_counter() - last_clock)
                    last_clock = time.perf_counter()
                    result = self._update(sess, feed_dict, bi)
                finally:
                    cumtime += (time.perf_counter() - last_clock)
                    self.q.task_done()
