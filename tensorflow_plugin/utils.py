# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import tensorflow as tf
import numpy as np


def compute_pairwise_potential(model_directory, r, potential_tensor_name, checkpoint = -1, feed_dict = {}):
    ''' Compute the pairwise potential at r for the given model.

    Parameters
    ----------
    model_directory
        The model directory
    r
        A 1D grid of points at which to compute the potential.
    potential_tensor_name
        The tensor containing potential energy.
    checkpoint
        Which checkpoint to load. Default is -1, which loads latest checkpoint. An integer indicates loading
        from the model directory. If you pass a string, it is interpreted as a path.
    feed_dict
        Allows you to add any other placeholder values that need to be added to compute potential in your model
    Returns
    -------
    A 1D array of potentials corresponding the pairwise distances in r.
    '''
    # just in case
    tf.reset_default_graph()
    # load graph
    tf.train.import_meta_graph(os.path.join('{}/'.format(model_directory),'model.meta'), import_scope='')
    with open('{}/graph_info.p'.format(model_directory), 'rb') as f:
        model_params = pickle.load(f)
    if not ':' in potential_tensor_name:
        potential_tensor_name += ':0'
    potential_tensor = tf.get_default_graph().get_tensor_by_name('calculated_energies:0')
    nlist_tensor = tf.get_default_graph().get_tensor_by_name(model_params['nlist'])

    #build nlist
    NN = model_params['NN']
    np_nlist = np.zeros( (2, NN, 4) )
    potential = np.empty(len(r))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if(checkpoint == -1):
            #get latest
            checkpoint_str = model_directory
            checkpoint = tf.train.latest_checkpoint(checkpoint_str)
            saver.restore(sess, checkpoint)
            checkpoint = 'latest'
        elif type(checkpoint) == int:
            #get specific checkpoint number
            checkpoint_str = '{}/model-{}'.format(model_directory, checkpoint)
            checkpoint = tf.train.load_checkpoint(checkpoint_str)
            saver.restore(sess, checkpoint_str)
        else:
            checkpoint_str = checkpoint
            checkpoint = tf.train.load_checkpoint(checkpoint_str)
            saver.restore(sess, checkpoint_str)
        for i,ri in enumerate(r):
            np_nlist[0,0,1] = r / 2
            np_nlist[1,0,1] = r / 2
            # run including passed in feed_dict
            result = sess.run(energy_tensor, feed_dict = {**feed_dict, nlist_tensor: np_nlist} )
            potential[i] = result
    return potential