import hoomd
import hoomd.md
import hoomd.data
import hoomd.init
import hoomd.dump
import hoomd.group
from hoomd.htf import tfcompute
import tensorflow as tf
from sys import argv as argv
from math import sqrt
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


if(len(argv) != 3):
    print('Usage: plot_learned_potential.py [checkpoint_directory]'
          ' [checkpoint_number](-1 for latest)')
    exit()

training_dir = argv[1]  # e.g. '/scratch/rbarret8/ann-training'
checkpoint_num = int(argv[2])

tf.train.import_meta_graph(os.path.join('{}/'.format(training_dir),
                                        'model.meta'), import_scope='')
with open('{}/graph_info.p'.format(training_dir), 'rb') as f:
    var_dict = pickle.load(f)
print(var_dict)

# fake_var = tf.get_variable('bias_b1', shape=[6])
energy_tensor = tf.get_default_graph(
    ).get_tensor_by_name('calculated_energies:0')
r_inv_tensor = tf.get_default_graph().get_tensor_by_name('r_inv:0')
nlist_tensor = tf.get_default_graph(
    ).get_tensor_by_name(var_dict['nlist'])
NN = var_dict['NN']
energy_arr = []


with tf.Session() as sess:
    saver = tf.train.Saver()
    if(checkpoint_num == -1):
        checkpoint_str = training_dir  # '/scratch/rbarret8/ann-training/'
        checkpoint = tf.train.latest_checkpoint(checkpoint_str)
        saver.restore(sess, checkpoint)
        checkpoint_num = 'latest'
    else:
        checkpoint_str = '{}model-{}'.format(training_dir, checkpoint_num)
        checkpoint = tf.train.load_checkpoint(checkpoint_str)
        print(checkpoint)
        saver.restore(sess, checkpoint_str)
    pos_arr = np.linspace(0., 3.0, 300)
    long_pos_arr = np.linspace(0., 3.0, 300)
    np_nlist = np.zeros((2, NN, 4))
    nlist = {}
    for i in range(1, 300):  # don't forget the distance is double
        np_nlist[0, 0, 1] = pos_arr[i]
        np_nlist[1, 0, 1] = -pos_arr[i]
        nlist[nlist_tensor] = np_nlist
        nlist['keep_prob:0'] = 1.0
        output = sess.run({'energy': energy_tensor}, feed_dict=nlist)
        print('pairwise energy with radius {} is :{}'.format(
                pos_arr[i], output['energy'][0]))
        energy_arr.append(output['energy'][0] + output['energy'][1])


energy_arr = np.array(energy_arr)


def lj_energy(r):
    return(4 * ((r)**(-12) - (r)**(-6)))


plt.figure()
plt.plot(pos_arr[1:], energy_arr - energy_arr[-1],
         label='Neural Network Potential')
plt.plot(long_pos_arr[1:], lj_energy(long_pos_arr[1:]),
         label='Lennard-Jones Potential')

NN_min_idx = np.argmin(energy_arr)
# if pos_arr[NN_min_idx] <= 0.2:#pointed the wrong way
#     energy_arr = -energy_arr #flip it upside-down
#     NN_min_idx = np.argmin(energy_arr)
lj_min_idx = np.argmin(lj_energy(pos_arr[1:]))
lj_min = pos_arr[lj_min_idx]
NN_min = pos_arr[NN_min_idx]

# plt.scatter(2*NN_min, energy_arr[NN_min_idx] - energy_arr[-1],
# label = 'NN minimum: {:.5}'.format(NN_min))
# plt.scatter(lj_min, lj_energy(lj_min), label =
# 'LJ minimum:{:.5}'.format(lj_min))

print('X value at min of calculated LJ: {}'.format(lj_min))
print('X value at min of Neural Net LJ: {}'.format(NN_min))
SIZE = 14
plt.legend(loc='best', fontsize=SIZE)
plt.xlabel(r'$(r\sigma)^{-1}$', fontsize=SIZE)
plt.ylabel(r'$U(r) / \epsilon$', fontsize=SIZE)
plt.xticks(np.arange(0., np.max(pos_arr)+0.5, 0.5), fontsize=SIZE)
plt.yticks(range(-2, 11, 1), fontsize=SIZE)
plt.ylim(-2, 20)
plt.savefig('step_{}_ann_potential_zoomout.png'.format(checkpoint_num))
plt.savefig('step_{}_ann_potential_zoomout.pdf'.format(checkpoint_num))
plt.savefig('step_{}_ann_potential_zoomout.svg'.format(checkpoint_num))
plt.ylim(-2, 10)
plt.savefig('step_{}_ann_potential.png'.format(checkpoint_num))
plt.savefig('step_{}_ann_potential.pdf'.format(checkpoint_num))
plt.savefig('step_{}_ann_potential.svg'.format(checkpoint_num))
