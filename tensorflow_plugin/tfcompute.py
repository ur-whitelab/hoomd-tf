# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

from hoomd.tensorflow_plugin import _tensorflow_plugin
from .tfmanager import main
import sys, math, numpy as np, pickle, queue, threading, os, time
import hoomd, hoomd.md.nlist, hoomd.comm
import tensorflow as tf

## Integrates tensorflow
#
# TODO
#
class tfcompute(hoomd.compute._compute):
    def __init__(self,tf_model_directory, log_filename='tf_manager.log', device=None,
                  bootstrap=None, bootstrap_map=None,
                  _debug_mode=False, _mock_mode=False, write_tensorboard=False):

        #so delete won't fail
        self.tfm = None

        #if hoomd.init.is_initialized():
        #    raise RuntimeError('Must create TF before hoomd initialization')


        self.debug_mode = _debug_mode
        self.tf_model_directory = tf_model_directory
        self.log_filename = log_filename

        try:
            with open(os.path.join(tf_model_directory, 'graph_info.p'), 'rb') as f:
                self.graph_info = pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError('Unable to load model in directory {}'.format(tf_model_directory))

        self.tasklock = _tensorflow_plugin.make_tasklock()
        self.mock_mode = _mock_mode
        self.device = device
        self.write_tensorboard = write_tensorboard
        self.bootstrap = bootstrap
        self.bootstrap_map = bootstrap_map
        self.feed_dict = None

    def __enter__(self):
        if not self.mock_mode:
            self._init_tf()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #trigger end in task lock
        if not self.mock_mode and self.tfm.is_alive():
            hoomd.context.msg.notice(2, 'Sending exit signal.\n')
            self.tasklock.exit()
            time.sleep(1)
            if self.tfm and self.tfm.is_alive():
                hoomd.context.msg.notice(2, 'Shutting down TF Manually.\n')
                self.shutdown_tf()

    ##
    # feed_dict = takes in tfcompute (which gives access to forces/positions/nlist)
    # feed_dict should return a dictionary where the key is the tensor name (can be set during graph build stage)
    # and the value is the result to be fed into the named tensor. Note that if you name a tensor, typically you must
    # append :0 to it. For example, if your name is 'my-tesnor', then the actual tensor is named 'my-tensor:0'.
    #
    def attach(self, nlist = None, r_cut = 0, save_period=1000, period=1, feed_dict=None, batch_size=None):

        #make sure we have number of atoms and know dimensionality, etc.
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error('Must attach TF after initialization\n')
            raise RuntimeError('Error creating TF')
        if self.tfm is None and not self.mock_mode:
            raise Exception('You must use the with statement to construct and attach a tfcompute')

        #I'm not sure if this is necessary following other files
        self.enabled = True
        self.log = True
        self.cpp_force = None
        self.feed_dict = feed_dict
        self.save_period = save_period
        self.force_name = 'tfcompute'
        self.compute_name = self.force_name
        self.nneighbor_cutoff = self.graph_info['NN']
        self.atom_number = len(hoomd.context.current.group_all)
        r_cut = float(r_cut)
        self.r_cut = r_cut
        self.batch_size = 0 if batch_size is None else batch_size

        if nlist is not None:
            nlist.subscribe(self.rcut)
            #activate neighbor list
            nlist.update_rcut()
        elif self.nneighbor_cutoff != 0:
            raise ValueError('Must provide an nlist if you have nneighbor_cutoff > 0')


        hoomd.util.print_status_line()

        # initialize base class
        hoomd.compute._compute.__init__(self)

        self.force_mode_code = _tensorflow_plugin.FORCE_MODE.tf2hoomd if self.graph_info['output_forces'] else _tensorflow_plugin.FORCE_MODE.hoomd2tf
        hoomd.context.msg.notice(2, 'Force mode is {} \n'.format(self.force_mode_code))
        #if graph is not outputting (input) then tfcompute should be outputting them
        if not self.graph_info['output_forces'] and not _tensorflow_plugin.FORCE_MODE.hoomd2tf:
            raise ValueError('Your graph takes forces as input but you are not sending them from tfcompute')

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _tensorflow_plugin.TensorflowCompute(self,
            hoomd.context.current.system_definition, nlist.cpp_nlist if nlist is not None else None,
            r_cut, self.nneighbor_cutoff, self.force_mode_code, period, self.batch_size)
            # TODO: This is not correct
            if self.device is None:
                self.device = '/cpu:0'
        else:
            self.cpp_force = _tensorflow_plugin.TensorflowComputeGPU(self,
            hoomd.context.current.system_definition,  nlist.cpp_nlist if nlist is not None else None,
            r_cut, self.nneighbor_cutoff, self.force_mode_code, period, self.batch_size)
            if self.device is None:
                self.device = '/gpu:0'

        # get double vs single precision
        self.dtype = tf.float32
        if self.cpp_force.isDoublePrecision():
            self.dtype = tf.float64

        # adding to forces causes the computeForces method to be called.
        hoomd.context.current.system.addCompute(self.cpp_force, self.compute_name)
        if self.force_mode_code == _tensorflow_plugin.FORCE_MODE.tf2hoomd:
            hoomd.context.current.forces.append(self)
        else:
            integrator = hoomd.context.current.integrator
            if integrator is None:
                raise ValueError('Must have integrator set to receive forces')
            integrator.cpp_integrator.setHalfStepHook(self.cpp_force.hook())

        if not self.mock_mode:
            self._start_tf()

    def set_reference_forces(self, *forces):
        if self.force_mode_code == _tensorflow_plugin.FORCE_MODE.tf2hoomd:
            raise ValueError('Only valid to set reference forces if mode is hoomd2tf')
        for f in forces:
            if not hasattr(f, 'cpp_force'):
                raise ValueError('given force does not seem like a hoomd force')
            self.cpp_force.addReferenceForce(f.cpp_force)
            hoomd.context.msg.notice(2, 'Will use given force for TFCompute {} \n'.format(f.name))

    def rcut(self):
        # adapted from hoomd/md/pair.py
        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes()
        type_list = []
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i))

        # update the rcut by pair type
        r_cut_dict = hoomd.md.nlist.rcut()
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                # get the r_cut value
                r_cut_dict.set_pair(type_list[i],type_list[j],self.r_cut)
        return r_cut_dict

    def __del__(self):
        if self.tfm and self.tfm.is_alive():
            self.shutdown_tf()

    def shutdown_tf(self):
        # need to terminate orphan
        if not self.q.full():
            hoomd.context.msg.notice(2, 'TF Queue is waiting, sending None\n')
            self.q.put(None)
        self.tfm.join(1)

    def _init_tf(self):
        self.q = queue.Queue(maxsize=1)
        self.tfm = threading.Thread(target=main, args=(self.q, self.tasklock,self.write_tensorboard, self.device))
        self.tfm.start()
        hoomd.context.msg.notice(2, 'Forked TF Session Manager.\n')

    def _start_tf(self):
        if not self.cpp_force:
            return
        args = {'log_filename': self.log_filename,
                'graph_info': self.graph_info,
                'positions_buffer': self.cpp_force.getPositionsBuffer(),
                'nlist_buffer': self.cpp_force.getNlistBuffer(),
                'forces_buffer': self.cpp_force.getForcesBuffer(),
                'virial_buffer': self.cpp_force.getVirialBuffer(),
                'dtype': self.dtype,
                'use_feed': self.feed_dict is not None,
                'bootstrap': self.bootstrap,
                'bootstrap_map': self.bootstrap_map,
                'save_period': self.save_period,
                'debug': self.debug_mode,
                'primary': hoomd.comm.get_rank() == 0,
                'device': self.device}
        self.q.put(args)
        message =  ['Starting TF Manager with:']
        for k,v in args.items():
            if k == 'graph_info':
                continue
            else:
                message.append('\t{: <20}: {: >20}'.format(str(k), str(v)))
        message.append('\t{: <20}:'.format('graph_info'))
        for k,v in args['graph_info'].items():
            message.append('\t  {: <18}: {: >20}'.format(str(k), str(v)))
        for m in message:
            hoomd.context.msg.notice(2, m + '\n')
        self.q.join()
        if not self.tfm.isAlive():
            exit()
        hoomd.context.msg.notice(2,'TF Session Manager has released control. Starting HOOMD updates\n')

    def finish_update(self, batch_index, batch_frac):
        '''Allow TF to read output and we wait for it to finish.'''
        if self.mock_mode:
            return
        fd = {'htf-batch-index:0': batch_index, 'htf-batch-frac:0': batch_frac}
        if self.feed_dict is not None:
            if type(self.feed_dict) == dict:
                value = self.feed_dict
            else:
                value = self.feed_dict(self)
                assert value is not None, 'feed_dict callable failed to provide value'
            self.q.put({**value, **fd}, block=False)
        else:
            self.q.put(fd, block=False)
        self.q.join()
        if not self.tfm.isAlive():
            hoomd.context.msg.error('TF Session Manager has unexpectedly stopped\n')
            raise RuntimeError('TF Session Manager has unexpectedly stopped\n')

    def get_positions_array(self):
        return self.scalar4_vec_to_np(self.cpp_force.getPositionsArray())

    def get_nlist_array(self):
        return self.scalar4_vec_to_np(self.cpp_force.getNlistArray())

    def get_forces_array(self):
        return self.scalar4_vec_to_np(self.cpp_force.getForcesArray())

    def get_virial_array(self):
        array = np.array(self.cpp_force.getVirialArray())
        return array.reshape((-1, 9))

    def update_coeffs(self):
        pass

    def scalar4_vec_to_np(self,array):
        npa = np.empty((len(array), 4))
        for i, e in enumerate(array):
            npa[i,0] = e.x
            npa[i,1] = e.y
            npa[i,2] = e.z
            npa[i,3] = e.w
        return npa
