import hoomd.htf as htf
from MDAnalysis import Universe
import tensorflow as tf
import numpy as np
import gsd.hoomd

def convert_trajectory(universe, output, rcut, NN=128, selection='all'):
    
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
    N = (np.shape(type_array))[0]
    # Make graph
    graph = htf.graph_builder(NN, output_forces=False)
	# specify nlist operation
    nlist = htf.compute_nlist(graph.positions, rcut, NN, system)
    t = gsd.hoomd.open(name = output, mode = 'wb')
    with tf.Session() as sess:
        # sess.run() evaluates the nlist at every frame of universe.trajectory. 
        for ts,i in zip(universe.trajectory, range(len(universe.trajectory))):
            nlist_values = sess.run(nlist, feed_dict={graph.positions: np.concatenate((ag.positions, type_array), axis=1), 
                                                      graph.box: hoomd_box, 
                                                      graph.batch_index: 0, 
                                                      graph.batch_frac: 1})
            t.append(create_frame(i, N, types, type_array, ag.positions, box, nlist_values))
            

def create_frame(frame_number, N, types, type_array, positions, box, nlist):
    s = gsd.hoomd.Snapshot()
    s.configuration.step = frame_number
    s.configuration.box = box
    s.particles.N = len(type_array)
    s.particles.types = types
    s.particles.typeid = type_array
    s.particles.position = positions
    return s



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert trajectory into tfrecords')
    parser.add_argument('--topology_file')
    parser.add_argument('--trajectory_file')
    parser.add_argument('--output_file', type=str, default='new.gsd')
    parser.add_argument('--rcut', type=float, default=10.)
    parser.add_argument('--weight_file', default=None)
    parser.add_argument('--selection', default='all')
    args = parser.parse_args()
    print(args)

    # load weights if passed
    # TODO
    u = Universe(args.topology_file, args.trajectory_file)

    convert_trajectory(universe = u, output = args.output_file, rcut = args.rcut, selection=args.selection)
