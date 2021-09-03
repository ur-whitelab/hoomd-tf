# Copyright (c) 2020 HOOMD-TF Developers

''' HOOMD-TF

Use TensorFlow to do arbitrary collective variable calculations and
machine learning on-the-fly in HOOMD-blue simulations.
'''
import hoomd.md
from hoomd.htf.tensorflowcompute import tfcompute
from hoomd.htf.simmodel import *
from hoomd.htf.version import __version__
from hoomd.htf.utils import *
from hoomd.htf.layers import *
import tensorflow as tf

_tf_on_gpu = False
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        _tf_on_gpu = True
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

try:
    custom_things = [SimModel, MolSimModel, RBFExpansion, WCARepulsion,
                     EDSLayer]
    custom_objects = {o.__name__: o for o in custom_things}
    custom_objects.update({o.__name__: o for o in custom_things})
    del custom_things
except NameError:
    # somehow we can get here on doc build?
    pass
