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


def main(q, tasklock, write_tensorboard=False, profile=False):

    tfm_args = q.get()
    tfm = TFManager(q=q, tasklock=tasklock,
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
        mod = tf.load_op_library(os.path.join(path,
                                              'lib_{}_op.so'.format(op)))
    except:
        raise IOError('Unable to load OP {}. Expected to be in {}'.format(op, path))
    return mod


class TFManager:
    def __init__(self, graph_info, device, q, tasklock,
                 positions_buffer, nlist_buffer,
                 forces_buffer, virial_buffer, log_filename,
                 dtype, debug, write_tensorboard, use_feed,
                 bootstrap, primary, bootstrap_map,
                 save_period, use_xla):
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
        self.tasklock = tasklock
        self.positions_buffer = positions_buffer
        self.nlist_buffer = nlist_buffer
        self.forces_buffer = forces_buffer
        self.virial_buffer = virial_buffer
        self.debug = debug
        self.step = 0
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

        for n in self.graph_info['out_nodes']:
            try:
                self.out_nodes.append(tf.get_default_graph(
                        ).get_tensor_by_name(n))
            except ValueError:
                self.out_nodes.append(tf.get_default_graph(
                        ).get_operation_by_name(n))

    def _update(self, sess, feed_dict=None):

        if self.step % self.save_period == 0:
            if self.summaries is not None:
                result = sess.run(self.out_nodes + [self.summaries],
                                  feed_dict=feed_dict)
            else:
                result = sess.run(self.out_nodes, feed_dict=feed_dict)
            self._save_model(sess, result[-1])
        else:
            result = sess.run(self.out_nodes, feed_dict=feed_dict)
        self.step += 1

        return result

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

    def _prepare_graph(self):
        hoomd_to_tf_module = load_op_library('hoomd2tf')
        hoomd_to_tf = hoomd_to_tf_module.hoomd_to_tf

        with tf.device(self.device):
            self.positions = hoomd_to_tf(
                address=self.positions_buffer, shape=[4],
                T=self.dtype, name='positions-input')
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
            self.log.log(10, 'initialized nlist hoomd_to_tf at address {:x}'
                         'with shape {} on {}'.format(self.nlist_buffer,
                                                      self.nlist.shape,
                                                      self.device))
        # now cast if graph dtype are different
        if self.graph_info['dtype'] != self.dtype:
            with tf.device(self.device):
                self.positions = tf.cast(self.positions,
                                         self.graph_info['dtype'])
                self.nlist = tf.cast(self.nlist, self.graph_info['dtype'])

        input_map = {self.graph_info['nlist']: self.nlist,
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
        tf_to_hoomd_module = load_op_library('tf2hoomd')
        tf_to_hoomd = tf_to_hoomd_module.tf_to_hoomd
        with tf.device(self.device):
            self.out_nodes.append(tf_to_hoomd(
                    self.forces, address=self.forces_buffer))
            self.log.log(10, 'initialized forces tf_to_hoomd at address {:x}'
                         ' with shape {} on {}'.format(self.forces_buffer,
                                                       self.forces.shape,
                                                       self.device))
        if self.graph_info['virial'] is not None:
            # virial is Nx3x3
            with tf.device(self.device):
                self.out_nodes.append(tf_to_hoomd(
                        self.virial, address=self.virial_buffer))
                self.log.log(10, 'initialized virial tf_to_hoomd at address'
                             ' {:x} with shape {} on {}'
                             .format(self.virial_buffer,
                                     self.virial.shape,
                                     self.device))

    def _attach_tensorboard(self, sess):

        self.summaries = tf.summary.merge_all()
        self.tb_writer = tf.summary.FileWriter(os.path.join(
                self.model_directory, 'tensorboard'),
                                               sess.graph)

    def start_loop(self):

        self.log.log(10, 'Constructed TF Model graph')
        # make it grow as memory is needed instead of consuming all
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        if self.use_xla:
            config.graph_options.optimizer_options.global_jit_level = \
            tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess:
            # resore model checkpoint if there are variables
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
                    # convert bootstrap map values into actual variables
                    variable_map = None
                    if self.bootstrap_map is not None:
                        variables = tf.get_collection(
                            tf.GraphKeys.GLOBAL_VARIABLES)
                        variable_map = dict()
                        for k, vname in self.bootstrap_map.items():
                            value = None
                            for v in variables:
                                if v.name == vname + ':0':
                                    value = v
                            if value is None:
                                raise ValueError(
                                    'Could not find variable'
                                    ' {} in graph while'
                                    ' processing'
                                    ' bootstrap_map'.format(vname))
                            variable_map[k] = value
                    bootstrap_saver = tf.train.Saver(variable_map,
                                                     **saver_args)
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
            result = None

            if self.use_feed:
                feed_dict = None
                while True:
                    try:
                        feed_name_dict = self.q.get()
                        if feed_name_dict is None:
                            self.log.exception('Empty')
                            raise queue.Empty()
                    except queue.Empty:
                        self.log.log(2, 'Received exit. Leaving TF Update'
                                     'Loop. \n')
                        self.log.log(2, 'TF Update time (excluding '
                                     'communication) is {}\n'.format(cumtime))
                        self._save_model(sess)
                        break
                    # convert name keys to actual tensor keys
                    try:
                        feed_dict = dict()
                        for k, v in feed_name_dict.items():
                            tensor = tf.get_default_graph(
                                ).get_tensor_by_name(k)
                            feed_dict[tensor] = v
                        last_clock = time.perf_counter()
                        result = self._update(sess, feed_dict=feed_dict)
                    finally:
                        cumtime += (time.perf_counter() - last_clock)
                        self.q.task_done()
            else:
                while True:
                    if not self.tasklock.start():
                        self.log.log(2, 'Received exit. Leaving TF Update'
                                     ' Loop.')
                        self.log.log(2, 'TF Update time (excluding'
                                     ' communication) is {:.3f}'
                                     ' seconds'.format(cumtime))
                        self._save_model(sess)
                        break
                    last_clock = time.perf_counter()
                    try:
                        result = self._update(sess)
                    except Exception as e:
                        self.tasklock.end()
                        self.tasklock.exit()
                        raise e
                    cumtime += (time.perf_counter() - last_clock)
                    self.tasklock.end()
