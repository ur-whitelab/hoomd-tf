import hoomd.htf as htf
from MDAnalysis import Universe
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import numpy as np


def convert_trajectory(universe, output, r_cut, weight_vector, NN=128, selection='all'):
    box = universe.dimensions 
    system = type('',
                    (object, ),
                    {'box': type('', (object,),
                    {'Lx': box[0], 'Ly': box[1], 'Lz':box[2]})})
    hoomd_box = [[box[0], 0, 0], [0, box[1], 0], [0, 0, box[2]]]
    # make type array
    ag = universe.select_atoms(selection)
    types = list(np.unique(ag.atoms.types))
    type_array = np.array([types.index(i) for i in ag.atoms.types]).reshape(-1, 1)
    graph = htf.graph_builder(0, False)
    nlist = htf.compute_nlist(graph.positions, r_cut, NN, system)
    with tf.Session() as sess:
        for ts in universe.trajectory:
            sess.run(nlist, feed_dict={graph.positions: np.concatenate((ag.positions, type_array), axis=1),
                                       graph.box: hoomd_box,
                                       graph.batch_index: 0,
                                       graph.batch_frac: 1})

       
        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert trajectory into tfrecords')
    parser.add_argument('topology_file')
    parser.add_argument('trajectory_file')
    parser.add_argument('output_file')
    parser.add_argument('--rcut', default=10)
    parser.add_argument('--weight_file')
    parser.add_argument('--selection', default=None, nargs='?')
    args = parser.parse_args()
    

    # load weights if passed
    # TODO
    u = Universe(args.topology_file, args.trajectory_file)

    convert_trajectory(u, args.output_file, args.rcut, None, selection=args.selection)
