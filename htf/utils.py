# Copyright (c) 2020 HOOMD-TF Developers

import tensorflow as tf
import numpy as np
from os import path
import pickle
import hoomd
from pkg_resources import parse_version


def center_of_mass(positions, mapping, box_size, name='center-of-mass'):
    ''' Computes mapped positions given positions and system-level mapping
    by considering PBC.

    :param positions: The tensor of particle positions
    :type positions: N x 3 tensor
    :param mapping: The coarse-grain system-level mapping used to produce
        the particles in system
    :type mapping: M x N tensor
    :param box_size: A list that contains the size of the box ``[Lx, Ly, Lz]``
    :type box_size:  list
    :param name: The name of the op to add to the TF graph
    :type name: string

    :return: An [M x 3] mapped particles
    '''

    try:
        sorting = hoomd.context.current.sorter.enabled
        if sorting:
            raise ValueError(
                'You must disable hoomd sorting to use center_of_mass!')
    except AttributeError:
        pass

    # slice to avoid accidents
    positions = positions[:, :3]
    # https://en.wikipedia.org/wiki/
    # /Center_of_mass#Systems_with_periodic_boundary_conditions
    # Adapted for -L to L boundary conditions
    # box dim in hoomd is 2 * L
    box_dim = box_size
    theta = positions / box_dim * 2 * np.pi
    xi = tf.math.cos(theta)
    zeta = tf.math.sin(theta)
    ximean = tf.sparse.sparse_dense_matmul(mapping, xi)
    zetamean = tf.sparse.sparse_dense_matmul(mapping, zeta)
    thetamean = tf.math.atan2(zetamean, ximean)
    return tf.identity(thetamean / np.pi / 2 * box_dim, name=name)


def compute_ohe_bead_type_interactions(pos_btype, nlist_btype, n_btypes):
    ''' Computes bead type interactions as a one-hot encoding.

    :param pos_btype: type of the beads based on the mapped positions[...,-1]
    :type pos_btype:  N tensor with dtype tf.int32
    :param nlist_btype: type of the beads based on the mapped neighborlist[...,-1]
    :type nlist_btype: N x M tensor with dtype tf.int32
    :param n_btypes: number of unique bead types in the CG molecule
    :type n_btypes: int

    :return: An [N x M x I] array, where M is the total number of beads in the system,
        N is the size of CG neighborlist and I is the total number of possible interactions
        between two beads
    '''
    m, n = tf.math.minimum(pos_btype[..., tf.newaxis], nlist_btype), tf.math.maximum(
        pos_btype[..., tf.newaxis], nlist_btype)
    one_hot_indices = m * (2 * n_btypes - m + 1) // 2 + n - m
    # Finding the total number of possible interactions between different bead
    # types
    total_interactions = n_btypes * (n_btypes - 1) // 2 + n_btypes
    return tf.one_hot(one_hot_indices, depth=total_interactions)


def compute_nlist(
        positions,
        r_cut,
        NN,
        box_size,
        sorted=False,
        return_types=False,
        exclusion_matrix=None):
    ''' Computes particle pairwise neighbor lists.

    :param positions: Positions of the particles
    :type positions: N x 4 or N x 3 tensor
    :param r_cut: Cutoff radius (Hoomd units)
    :type r_cut: float
    :param NN: Maximum number of neighbors per particle
    :type NN: int
    :param box_size: A list contain the size of the box [Lx, Ly, Lz]
    :type box_size: list or shape 3 tensor
    :param sorted: Whether to sort neighbor lists by distance
    :type sorted: bool
    :param return_types: If true, requires N x 4 positions array and
        last element of nlist is type. Otherwise last element is index of neighbor
    :type return_types: bool
    :param exclusion_matrix: Matrix (True = exclude) with size B x B (where B is the total number
        of bead particles in the system) indicating which pairs
        should be excluded from nlist
    :type exclusion_matrix: Tensor of dtype bools


    :return: An [N x NN x 4] tensor containing neighbor lists of all
        particles and index
    '''

    if return_types and positions.shape[1] == 3:
        raise ValueError(
            'Cannot return type if positions does not have type. Make sure positions is N x 4')
    # get number of input positions
    M = tf.shape(positions)[0]

    # Make sure positions is only xyz
    positions3 = positions[:, :3]
    # subtract them to get distance vector matrix
    dist_mat = positions3[tf.newaxis, ...] - positions3[:, tf.newaxis, :]
    # apply minimum image
    box = tf.reshape(tf.convert_to_tensor(value=box_size), [1, 1, 3])
    dist_mat -= tf.math.round(dist_mat / box) * box
    dist = tf.norm(tensor=dist_mat, axis=2)
    # mask distance matrix to remove things beyond cutoff and zeros
    # mask value = 1: valid
    mask = (dist <= r_cut) & (dist >= 5e-4)
    if exclusion_matrix is not None:
        # negate it since mask = 1 is valid
        # also make sure it's symmetric
        nem = tf.logical_not(exclusion_matrix)
        mask &= nem & tf.transpose(nem)
    mask_cast = tf.cast(mask, dtype=dist.dtype)
    if sorted:
        # replace these masked elements with really large numbers
        # that will be very negative (therefore not part of "top")
        dist_mat_r = dist * mask_cast + (1 - mask_cast) * 1e20
        topk = tf.math.top_k(-dist_mat_r, k=NN, sorted=True)
    else:
        # all the 0s will disappear as we grab topk
        dist_mat_r = dist * mask_cast
        topk = tf.math.top_k(dist_mat_r, k=NN, sorted=False)

    # we have the topk, but now we need to remove others
    idx = tf.tile(tf.reshape(tf.range(M), [-1, 1]), [1, NN])
    idx = tf.reshape(idx, [-1, 1])
    flat_idx = tf.concat([idx, tf.reshape(topk.indices, [-1, 1])], -1)
    # mask is reapplied here, so those huge numbers won't still be in there.
    nlist_pos = tf.reshape(tf.gather_nd(dist_mat, flat_idx), [-1, NN, 3])
    nlist_mask = tf.reshape(tf.gather_nd(mask_cast, flat_idx), [-1, NN, 1])

    if return_types:
        nlist_type = tf.reshape(
            tf.gather(positions[:, 3], flat_idx[:, 0]), [-1, NN, 1])
        return tf.concat([
            nlist_pos,
            nlist_type
        ], axis=-1) * nlist_mask
    else:
        return tf.concat([
            nlist_pos,
            tf.cast(tf.reshape(topk.indices, [-1, NN, 1]),
                    tf.float32)], axis=-1) * nlist_mask


def compute_pairwise(model, r, type_i=0, type_j=0):
    ''' Computes model output for a 2 particle system of type_i and type_j at
    distances set by ``r``.
    If the model outputs two tensors of shape ``L x M`` and ``K``, then
    the output  will be a tuple of numpy arrays of size ``N x L x M`` and
    ``N x K``, where ``N`` is number of points in ``r``.

    :param model: The model
    :type model: :py:class:`.SimModel`
    :param r: A 1D grid of points at which to compute the potential.
    :type r: numpy array
    :param type_i: First particle (bead) type
    :type type_i: int
    :param type_j: Second particle (bead) type
    :type type_j: int

    :return: A tuple of numpy arrays as output from ``model``
    '''
    NN = model.nneighbor_cutoff
    nlist = np.zeros((2, NN, 4))
    nlist[0, :, -1] = type_j
    nlist[1, :, -1] = type_i
    output = None
    positions = np.zeros((2, 4))
    positions[0, -1] = type_i
    positions[1, -1] = type_j
    box = tf.constant([[0., 0, 0], [1e10, 1e10, 1e10], [0, 0, 0]])

    for i, ri in enumerate(r):
        nlist[0, 0, 1] = ri
        nlist[1, 0, 1] = -ri
        result = model([nlist, positions, box, 1.0])
        if output is None:
            output = [r.numpy()[np.newaxis, ...] for r in result]
        else:
            output = [np.append(o, r[np.newaxis, ...], axis=0)
                      for o, r in zip(output, result)]
    return output


def create_frame(frame_number, N, types, typeids, positions, box):
    ''' Creates snapshots of a system state.

    :param frame_number: Frame number in a trajectory
    :type frame_number: int
    :param N: Number of CG beads
    :type N: int
    :param types: Names of particle types
    :type types: List of strings (N,)
    :param typeids: CG bead type id
    :type typeids: Numpy array (N,)
    :param positions: CG beads positions
    :type positions: Numpy array (N,3)
    :param box: System box dimensions
    :type box: Numpy array (6,)

    :return: Snapshot of a system state
    '''
    import gsd
    import gsd.hoomd

    s = gsd.hoomd.Snapshot()
    s.configuration.step = frame_number
    s.configuration.box = box
    s.particles.N = N
    s.particles.types = types
    s.particles.typeid = typeids
    s.particles.position = positions

    return s


def find_molecules(system):
    ''' Given a hoomd system, return a mapping from molecule index to particle index.
    This is a slow function and should only be called once.

    :param system: The molecular system in Hoomd.

    :return: A list of length L (number of molecules) whose elements are lists of atom indices
    '''
    mapping = []
    mapped = set()
    N = len(system.particles)
    unmapped = set(range(N))
    pi = 0

    # copy over bonds for speed
    bonds = [[b.a, b.b] for b in system.bonds]

    print('Finding molecules...', end='')
    while len(mapped) != N:
        print('\rFinding molecules...{:.2%}'.format(len(mapped) / N), end='')
        pi = unmapped.pop()
        mapped.add(pi)
        mapping.append([pi])
        # traverse bond group
        # until no more found
        # Have to keep track of "to consider" for branching molecules
        to_consider = [pi]
        while len(to_consider) > 0:
            pi = to_consider[-1]
            found_bond = False
            for bi, bond in enumerate(bonds):
                # see if bond contains pi and an unseen atom
                if (pi == bond[0] and bond[1] in unmapped) or \
                        (pi == bond[1] and bond[0] in unmapped):
                    new_pi = bond[0] if pi == bond[1] else bond[1]
                    unmapped.remove(new_pi)
                    mapped.add(new_pi)
                    mapping[-1].append(new_pi)
                    to_consider.append(new_pi)
                    found_bond = True
                    break
            if not found_bond:
                to_consider.remove(pi)
    # sort it to be ascending in min atom index in molecule
    print('')
    for m in mapping:
        m.sort()
    mapping.sort(key=lambda x: min(x))
    return mapping


def find_molecules_from_topology(
        universe,
        atoms_in_molecule_list,
        selection='all'):
    ''' Given a universe from MDAnaylis and list of atoms in every molecule type
    in the system, return a mapping from molecule index to particle index.
    Depending on the size of your system, this fuction might be slow to run.

    :param universe: Use MDAnalysis universe to read the tpr topology file from GROMACS.
    :type universe: MDAnalysis Universe object
    :param selection: The atom groups to extract from universe
    :param atoms_in_molecule_list: This is a list of atoms lists in every molecule type
        in the system.

    :return: A list of length L (number of molecules) whose elements are lists of atom indices.

    Here's an example:

    .. code:: python

                TPR = 'nvt_prod.tpr'
                TRAJECTORY = 'Molecules_CG_Mapping/traj.trr'
                u = mda.Universe(TPR, TRAJECTORY)
                atoms_in_molecule_list = [
                    u.select_atoms("resname PHE and resid 0:1").names]
                find_molecules_from_topology(
                    u, atoms_in_molecule_list, selection = "resname PHE")
    '''

    # Getting total number of atoms in selection from topology
    total_number_of_atoms = universe.select_atoms(selection).n_atoms

    # Getting the list of every redidue type name in topology
    _, idx = np.unique(universe.atoms.resnames, return_index=True)
    resname_list = universe.atoms.resnames[np.sort(idx)].tolist()

    molecule_list_indexed = []
    molecule_to_be_added = []
    for i in range(total_number_of_atoms):
        resname_type_index = resname_list.index(universe.atoms.resnames[i])
        molecule_length = len(atoms_in_molecule_list[resname_type_index])
        if len(molecule_to_be_added) < molecule_length:
            molecule_to_be_added.append(i)
        if len(molecule_to_be_added) == molecule_length:
            molecule_list_indexed.append(molecule_to_be_added)
            molecule_to_be_added = []
    if molecule_list_indexed[-1][-1] != total_number_of_atoms - 1:
        raise Exception(
            "Mismatch found between the number of atoms in the system and the final index value.\
             Check your atoms_in_molecule_list input.")
    return molecule_list_indexed


def find_cgnode_id(atm_id, cg):
    ''' Computes the CG bead index. Supports only
    outputs formats from DSGPM model. Called by compute_adj_mat function.

    :param atm_id: index of the atom to find its CG node id
    :type atm_id: int
    :param cg: array of cg beads
    :type cg: numpy array

    :return: CG bead index of the given atom
    '''
    for num_index, num_val in enumerate(cg):
        for j_idx, j_value in enumerate(num_val):
            if j_value == atm_id:
                return num_index


def gen_mapped_exclusion_list(
        universe,
        atoms_in_molecule,
        beads_mappings,
        selection='all'):
    ''' Generates mapped exclusion list to compute mapped_nlist for non-bonded bead-type
    interactions.

    :param universe: MDAnalysis Universe that contains bond information
    :type universe: MDAnalysis Universe object
    :param atoms_in_molecule: Selection of atoms in the molecule from MDAnalysis universe
    :type atoms_in_molecule: ``MDAnalysis.core.groups.AtomGroup``
    :param beads_mappings: List of lists of beads mappings. Note that each list should
                contain atoms as strings just like how they appear in the topology file.
    :type beads_mappings: Array
    :param selection: The atom groups to extract from universe
    :type selection: string

    :return: A [B x B] array of dtype bools, indicating which pairs
        should be excluded from nlist (True = exclude). B is the total number of
        bead particles in the system
    '''
    # Get the number of atoms
    N = len(universe.select_atoms(selection))
    bonds = universe.select_atoms(selection).bonds.to_indices()
    aa_exclusion_list = np.zeros((N, N), dtype=bool)
    for b in bonds:
        aa_exclusion_list[tuple(b)] = 1
        aa_exclusion_list[tuple(np.roll(b, 1))] = 1
    matrix_mapping_molecule = hoomd.htf.matrix_mapping(
        atoms_in_molecule, beads_mappings, mass_weighted=False)[1]
    # Get the number of molecules
    M = N // matrix_mapping_molecule.shape[1]
    # repeat matrix_mapping_molecule along diag
    matrix_mapping_system = np.kron(
        np.eye(M, dtype=int), matrix_mapping_molecule).astype(bool)
    mapped_exclusion = matrix_mapping_system @ aa_exclusion_list @ (
        matrix_mapping_system.T)
    np.fill_diagonal(mapped_exclusion, False)
    return mapped_exclusion


def gen_bonds_group(mapped_exclusion_list):
    ''' Generates HOOMD snapshot bonds group based on
    bead particles' exclusion list.

    :param mapped_exclusion_list: A [B x B] array of dtype bools,
        indicating which pairs should be excluded from nlist (True = exclude)
    :type obj: numpy array

    :return: :py:meth:`hoomd.data.make_snapshot.bonds.group`
    '''
    rows, cols = np.where(mapped_exclusion_list)
    bonds_group = np.array([[rows[i], cols[i]]
                            for i in range(rows.shape[0]) if rows[i] <= cols[i]])
    return bonds_group


def compute_adj_mat(obj):
    ''' Given a CG mapping file in json format, outputs the
    adjacency matrix. See :py:meth:`.utils.compute_cg_graph`.

    :param obj: mapping output from DSGPM
    :type obj: dict

    :return: Adjacency matrix of the mapping
    '''
    cg = obj['cgnodes']
    cg_num = len(cg)
    adj = np.zeros((cg_num, cg_num))
    for edges in obj['edges']:
        source_id = int(edges['source'])
        target_id = int(edges['target'])
        source_cg = find_cgnode_id(source_id, cg)
        target_cg = find_cgnode_id(target_id, cg)
        if source_cg != target_cg:
            adj[source_cg, target_cg] = adj[target_cg, source_cg] = 1
    return adj


def compute_cg_graph(
        DSGPM=True,
        infile=None,
        adj_mat=None,
        cg_beads=None,
        group_atoms=False,
        u_no_H=None,
        u_H=None):
    ''' Given a CG mapping in JSON format(from DSGPM model) OR adjacency matrix,
    outputs indices of connected CG beads to compute CG bond distances, CG angles
    and CG dihedrals. If DSGPM is True, path to jsonfiles must be specified. If DSGPM
    is False, adjacency matrix and the number of CG beads must be specified.
    If ``group_atoms`` is given as True outputs CG coordinates as well.
    If ``group_atoms`` flag is set to True, two MDAnalysis universes with Hydrogens
    and without Hydrogens must be given as arguments.

    Optional dependencies: MDAnalysis, networkx

    :param DSGPM: flag to identify if mapping in json format is used or not
    :type DSGPM: bool
    :param infile: path to the CG mapping in JSON format
    :type infile: str
    :param adj_matrix: adjacency matrix (if DSGPM=False)
    :type adj_matrix: numpy array
    :param cg_beads: number of CG beads per molecule
    :type cg_beads: int
    :param group_atoms: flag to output CG coordinates
    :type group_atoms: bool
    :param u_no_H: All atom structure without hydrogens
    :type u_no_H: MDAnalysis universe
    :param u_H: All atom structure with hydrogens
    :type u_H: MDAnalysis universe

    :return: List of indices bonded CG bead pairs, list of indices of CG beads making angles,
             list of indices of CG beads making dihedrals, and/or CG coordinates
     '''
    import MDAnalysis as mda
    import networkx as nx
    import json

    dist_idx = []
    ang_idx = []
    dihe_idx = []
    dist_list = []

    if DSGPM is True and infile is not None:
        obj = json.load(open(infile, 'r'))
        cg = obj['cgnodes']
        cg_num = len(cg)
        adj = compute_adj_mat(obj)

    elif DSGPM is False and adj_mat is not None:
        adj = adj_mat
        cg_num = cg_beads

    else:
        print('correct inputs/flags are not given')

    if adj is not None and cg_num is not None:
        cg_grph = nx.Graph(adj)

        length = dict(nx.all_pairs_shortest_path_length(cg_grph))

        # find node connectivities from the CG graph
        for i in range(cg_num):
            for j in range(i + 1, cg_num):
                cg_l = length[i][j]
                if cg_l == 1:
                    dist_idx.append((i, j))
                elif cg_l == 2:
                    ang_idx.append((i, j))

                elif cg_l == 3:
                    dihe_idx.append((i, j))

        # find indices of bonded pairs
        for x in range(len(dist_idx)):
            r_source = dist_idx[x][0]
            r_target = dist_idx[x][1]
            dist_list.append(
                list(
                    nx.all_shortest_paths(
                        cg_grph,
                        source=r_source,
                        target=r_target)))

        rs = np.asarray(dist_list).squeeze(axis=(1,))

        # find indices of angles-making nodes
        ang_list = []
        for x in range(len(ang_idx)):
            a_source = ang_idx[x][0]
            a_target = ang_idx[x][1]
            ang_list.append(
                list(
                    nx.all_shortest_paths(
                        cg_grph,
                        source=a_source,
                        target=a_target)))
        angs = np.asarray(ang_list).squeeze(axis=(1,))

        # find indices of dihedral-making nodes
        dih_list = []
        for x in range(len(dihe_idx)):
            d_source = dihe_idx[x][0]
            d_target = dihe_idx[x][1]
            dih_list.append(
                list(
                    nx.all_shortest_paths(
                        cg_grph,
                        source=d_source,
                        target=d_target)))
        dihs = np.asarray(dih_list).squeeze(axis=(1,))

        if group_atoms is True:
            if u_no_H is None or u_H is None:
                print('One or both MDAnalysis universe not specified')

            if u_H is not None and u_no_H is not None:
                cg_positions = []
                for i in range(cg_num):
                    atm_group = 0
                    for j in range(len(cg[i])):
                        atm_id = cg[i][j]
                        atom = u_no_H.atoms[atm_id]
                        a_name = str(atom.name)
                        a_resid = str(atom.resid)
                        heavy_atom = u_H.select_atoms(
                            'name ' + a_name + ' and resid ' + a_resid)
                        h = u_H.select_atoms(
                            'type H and bonded name ' + a_name + ' and resid ' + a_resid)
                        if len(list(h)) == 0:
                            atm_group += heavy_atom
                        else:
                            ah = heavy_atom + h
                            atm_group += ah

                    com = atm_group.center_of_mass()
                    cg_positions.append(com)

                return rs, angs, dihs, np.asarray(cg_positions)

        else:
            print(
                'CG coordinates are not caculated. Only connectivities are calculated')

            return rs, angs, dihs


def mol_features_multiple(bnd_indices=None, ang_indices=None,
                          dih_indices=None, molecules=None, beads=None):
    '''Use to apply the result of compute_cg_graph to multiple
    molecules in a system. Computes the atom/cg bead indices of bonds,angles
    and dihedrals for multiple molecules.

    :param bnd_indices: indices of atoms making bonds
    :type bnd_indices: numpy array of type int
    :param ang_indices: indices of atoms making angles
    :type ang_indices: numpy array of type int
    :param dih_indices: indices of atoms making dihedrals
    :type dih_indices: numpy array of type int
    :param molecules: number of molecules in the system
    :type molecules: int
    :param beads: number of atoms/CG beads in a molecule
    :type beads: int

    :return: arrays of indices of atoms/cg beads making bonds,
    angles and dihedrals in a system of molecules.
    '''

    bnd_ids = []
    ang_ids = []
    dih_ids = []

    for n in range(molecules):
        if bnd_indices is not None:
            bnd_ids.append(bnd_indices + n * beads)

        if ang_indices is not None:
            ang_ids.append(ang_indices + n * beads)

        if dih_indices is not None:
            dih_ids.append(dih_indices + n * beads)

    bnd_ids = np.asarray(bnd_ids).reshape((-1, 2))
    ang_ids = np.asarray(ang_ids).reshape((-1, 3))
    dih_ids = np.asarray(dih_ids).reshape((-1, 4))

    return bnd_ids, ang_ids, dih_ids


def iter_from_trajectory(
        nneighbor_cutoff,
        universe,
        selection='all',
        r_cut=10.,
        period=1,
        start=0.,
        end=None):
    ''' This generator will process information from a trajectory and
    yield a tuple of  ``[nlist, positions, box]`` and ``MDAnalysis.TimeStep`` object.
    The first list can be directly used to call a :py:class:`.SimModel` (e.g., ``model(inputs)``).
    See :py:meth:`.SimModel.compute` for details of these terms.

    Here's an example:

    .. code:: python

        model = MyModel(16)
        for inputs, ts in iter_from_trajectory(16, universe):
            result = model(inputs)

    :param nneighbor_cutoff: The maximum size of neighbor list
    :type nneighbor_cutoff: int
    :param universe: The MDAnalysis universe
    :param selection: The atom groups to extract from universe
    :type selection: string
    :param r_cut: The cutoff raduis to use in neighbor list
        calculations
    :type r_cut: float
    :param period: Period of reading the trajectory frames
    :type period: int
    :param start: Starting frame for reading the trajectory frames
    :type period: int
    :param period: Ending frame for reading the trajectory frames
    :type period: int
    '''
    import MDAnalysis
    from tqdm import tqdm
    # Modifying the universe for none 'all' atom selections.
    if selection != 'all':
        from MDAnalysis.analysis.base import AnalysisFromFunction
        p = universe.select_atoms(selection)
        dt = universe.trajectory[0].dt
        dimensions = universe.trajectory[0].dimensions
        if universe.trajectory[0].has_forces is False:
            # Only include positions if traj does not have forces
            x = AnalysisFromFunction(
                lambda ag: [ag.positions.copy()], p).run().results
            # Construct new_trajectory from the MemoryReader explicitly:
            new_traj = MDAnalysis.coordinates.memory.MemoryReader(
                x[:, 0], dimensions=dimensions, dt=dt)
        else:
            # Include positions, velocities and forces:
            xvf = AnalysisFromFunction(lambda ag: [ag.positions.copy(
            ), ag.velocities.copy(), ag.forces.copy()], p).run().results
            new_traj = MDAnalysis.coordinates.memory.MemoryReader(
                xvf[:, 0], velocities=xvf[:, 1], forces=xvf[:, 2], dimensions=dimensions, dt=dt)
        universe.trajectory = new_traj
        print(
            f'The universe was redefined based on the atom group {selection}.')
    # read trajectory
    # Modifying the universe for non 'all' atom selections.
    box = universe.dimensions
    # convert lattice angles to tilt factors!
    a = box[0]
    b = 1  # box[1]
    c = 1  # box[2]
    alpha = np.deg2rad(box[3])
    beta = np.deg2rad(box[4])
    gamma = np.deg2rad(box[5])
    lx = a
    xy = 1. / np.tan(gamma)
    xz = c * np.cos(beta)
    yz = (b * c * np.cos(alpha) - xy * xz)
    # define the system
    hoomd_box = np.array([[0, 0, 0], [box[0], box[1], box[2]], [xy, xz, yz]])
    # make type array
    # Select atom group to use in the system
    atom_group = universe.select_atoms(selection)
    # get unique atom types in the selected atom group
    try:
        types = list(np.unique(atom_group.atoms.types))
        # associate atoms types with individual atoms
        type_array = np.array([types.index(i)
                               for i in atom_group.atoms.types]).reshape(-1, 1)
    except MDAnalysis.exceptions.NoDataError:
        type_array = np.zeros(len(atom_group)).reshape(-1, 1)

    # define nlist operation
    # box_size = [box[0], box[1], box[2]]
    nlist = compute_nlist(
        atom_group.positions,
        r_cut=r_cut,
        NN=nneighbor_cutoff,
        box_size=box[:3])
    if end is None:
        end = universe.trajectory.totaltime
    # Need this for silly TF 2.4
    box = hoomd_box
    vtf = parse_version(tf.__version__)
    if vtf >= parse_version('2.4'):
        box = tf.SparseTensor(
            indices=[[0, 0, 0],
                     [0, 0, 1],
                     [0, 0, 2],
                     [0, 1, 0],
                     [0, 1, 1],
                     [0, 1, 2],
                     [0, 2, 0],
                     [0, 2, 1],
                     [0, 2, 2]],
            values=tf.reshape(box, (-1,)),
            dense_shape=(len(atom_group), 3, 3)
        )
    # Run the model at every nth frame where time is in range [start,end] and
    # n = period
    for i, ts in enumerate(tqdm(universe.trajectory)):
        if ts.frame >= start and ts.frame <= end:
            if i % period == 0:
                yield [nlist, np.concatenate(
                    (atom_group.positions,
                     type_array),
                    axis=1), box], ts


def matrix_mapping(molecule, beads_mappings, mass_weighted=True):
    '''Creates a ``M x N`` mass weighted mapping matrix where ``M`` is the number
    of atoms in the molecule and ``N`` is the number of mapping beads.

    :param molecule: This is atom selection in the molecule.
    :type molecule: MDAnalysis Atoms object
    :param beads_mappings: List of lists of beads mapping. Note that each list should
                contain atoms as strings just like how they appear in the topology file.
    :type beads_mappings: Array
    :param mass_weighted: Returns mass weighted mapping matrix (if `True`)
                     or both mass weighted and non-mass weighted matrices (if `False`)
    :type no_mass_mat: Boolean

    :return: Mappying operator at the molecule level. (Array/arrays) of shape M x N.
        Use :py:meth:`.utils.sparse_mapping` to get mapping operator at the system level
    '''
    Mws_dict = dict(zip(molecule.names, molecule.masses))
    M, N = len(beads_mappings), len(molecule)
    CG_matrix = np.zeros((M, N))
    index = 0
    for s in range(M):
        for i, atom in enumerate(beads_mappings[s]):
            CG_matrix[s, i + index] = [v for k,
                                       v in Mws_dict.items() if atom in k][0]
        index += np.count_nonzero(CG_matrix[s])
        CG_matrix[s] = CG_matrix[s] / np.sum(CG_matrix[s])
    # Cheking that all atoms in the topology are included in the beads mapping
    # list:
    assert index == molecule.n_atoms, 'Number of atoms in the beads mapping list does \
     not match the number of atoms in topology.'
    if mass_weighted is True:
        return CG_matrix
    else:
        no_mass_mapping = np.where(CG_matrix == 0, CG_matrix, 1)
        return CG_matrix, no_mass_mapping


def mol_angle(
        mol_positions=None,
        type_i=None,
        type_j=None,
        type_k=None,
        CG=False,
        cg_positions=None,
        b1=None,
        b2=None,
        b3=None,
        box=None):
    ''' This method calculates the bond angle given three atoms batched by molecule.
    Or to output CG angles input CG=True and indices of the CG beads making the angles.
    cg_positions and bead indices can be computed by calling :py:meth:`.generate_cg_graph()`

    :param  mol_positions: Positions tensor of atoms batched by molecules.
            Can be created by calling :py:meth:`.build_mol_rep()` method in simmodel
    :type mol_positions: float
    :param type_i: Index of the first atom
    :type type_i: int
    :param type_j: Index of the second atom
    :type type_j: int
    :param type_k: Index of the third atom
    :type type_k: int
    :param CG: flag to compute CG angles must be given with b1, b2 and b3
    :type CG: bool
    :param cg_positions: array of CG coordinates
    :type cg_positions: float
    :param b1: index of first CG bead
    :type b1: int
    :param b2: index of second CG bead
    :type b2: int
    :param b3: index of third CG bead
    :type b3: int
    :param box: low, high coordinates and tilt vectors of the box
    :type box: [3,3] array

    :returns: angles:Tensor containing all atom angles (CG=False)
              or
              cg_angles: list containing CG angles (CG=True)
    '''
    if mol_positions is None and CG is False:
        raise ValueError('mol_positions not found. Call build_mol_rep()')

    if mol_positions is not None and CG is False:
        v_ij = mol_positions[:, type_i, :3] - mol_positions[:, type_j, :3]
        v_jk = mol_positions[:, type_k, :3] - mol_positions[:, type_j, :3]
        wrap_vij = hoomd.htf.wrap_vector(v_ij, box)
        wrap_vjk = hoomd.htf.wrap_vector(v_jk, box)
        cos_a = tf.einsum('ij,ij->i', wrap_vij, wrap_vjk)
        cos_a = tf.math.divide(
            cos_a,
            (tf.norm(
                tensor=wrap_vij,
                axis=1) *
                tf.norm(
                tensor=wrap_vjk,
                axis=1)))
        angles = tf.math.acos(cos_a)
        return angles

    if CG is True and cg_positions is None:
        raise ValueError('cg_positions not found.')

    if CG is True and cg_positions is not None:
        v_ij = cg_positions[b2] - cg_positions[b1]
        v_jk = cg_positions[b3] - cg_positions[b2]
        wrap_vij = hoomd.htf.wrap_vector(v_ij, box)
        wrap_vjk = hoomd.htf.wrap_vector(v_jk, box)
        cos_a = np.dot(wrap_vij, wrap_vjk)
        cos_a = np.divide(
            cos_a, (np.linalg.norm(wrap_vij) * np.linalg.norm(wrap_vjk)))

        cg_angles = np.arccos(cos_a)
        return cg_angles


def mol_bond_distance(
        mol_positions=None,
        type_i=None,
        type_j=None,
        CG=False,
        cg_positions=None,
        b1=None,
        b2=None,
        box=None):
    ''' This method calculates the bond distance given two atoms batched by molecule.
    Or to output CG bond distances, input CG=True and indices of the CG bead pairs
    cg_positions and bead indices can be computed by calling :py:meth:`.generate_cg_graph()`

    :param mol_positions: Positions tensor of atoms batched by molecules.
           Can be created by calling build_mol_rep() method in simmodel
    :type mol_positions: float
    :param type_i: Index of the first atom
    :type type_i: int
    :param type_j: Index of the second atom
    :type type_j: int type
    :param CG: flag to compute CG bond distances (must be given with b1 and b2)
    :type CG: bool
    :param cg_positions: array of CG coordinates
    :type cg_positions: float
    :param b1: index of first CG bead
    :type b1: int
    :param b2: index of second CG bead
    :type b2: int
    :param box: low, high coordinates and tilt vectors of the box
    :type box: [3,3] array

    :returns: v_ij: Tensor containing bond distances(CG=False)
              or
              u_ij: Array containig CG bond distances(CG=True)
    '''

    if CG is False and mol_positions is None:
        raise ValueError('mol_positions not found. Call build_mol_rep()')

    if CG is False and mol_positions is not None:
        v_ij = mol_positions[:, type_j, :3] - mol_positions[:, type_i, :3]
        wrap_vij = hoomd.htf.wrap_vector(v_ij, box)
        wrap_vij = tf.norm(tensor=wrap_vij, axis=1)
        return wrap_vij

    if CG is True and cg_positions is None:
        raise ValueError('cg_positions not found')

    if CG is True and cg_positions is not None:
        u_ij = cg_positions[b2] - cg_positions[b1]
        wrap_uij = hoomd.htf.wrap_vector(u_ij, box)
        wrap_uij = np.linalg.norm(wrap_uij)
        return wrap_uij


def mol_dihedral(
        mol_positions=None,
        type_i=None,
        type_j=None,
        type_k=None,
        type_l=None,
        CG=False,
        cg_positions=None,
        b1=None,
        b2=None,
        b3=None,
        b4=None,
        box=None):
    ''' This method calculates the dihedral angles given three atoms batched by molecule.
    Or to output CG dihedral angles input CG=True and indices of the CG beads making the angles.
    cg_positions and bead indices can be computed by calling :py:meth:`.generate_cg_graph()`

    :param  mol_positions: Positions tensor of atoms batched by molecules.
            Can be created by calling build_mol_rep() method in simmodel
    :type mol_positions: float
    :param type_i: Index of the first atom
    :type type_i: int
    :param type_j: Index of the second atom
    :type type_j: int
    :param type_k: Index of the third atom
    :type type_k: int
    :param type_l: Index of the fourth atom
    :type type_k: int
    :param CG: flag to compute CG dihedral angles must be given with b1,b2,b3 and b4
    :type CG: bool
    :param cg_positions: array of CG coordinates
    :type cg_positions: float
    :param b1: index of first CG bead
    :type b1: int
    :param b2: index of second CG bead
    :type b2: int
    :param b3: index of third CG bead
    :type b3: int
    :param b4: index of fourth CG bead
    :type b4: int
    :param box: low, high coordinates and tilt vectors of the box
    :type box: [3,3] array

    :returns: dihedrals:Tensor containing all atom dihedral angles (CG=False)
              or
              cg_dihedrals: list containing CG dihedral angles (CG=True)
    '''

    if mol_positions is None and CG is False:
        raise ValueError('mol_positions not found. Call build_mol_rep()')

    if mol_positions is not None and CG is False:
        v_ij = mol_positions[:, type_j, :3] - mol_positions[:, type_i, :3]
        v_jk = mol_positions[:, type_k, :3] - mol_positions[:, type_j, :3]
        v_kl = mol_positions[:, type_l, :3] - mol_positions[:, type_k, :3]
        wrap_vij = hoomd.htf.wrap_vector(v_ij, box)
        wrap_vjk = hoomd.htf.wrap_vector(v_jk, box)
        wrap_vkl = hoomd.htf.wrap_vector(v_kl, box)

        # calculation of normal vectors
        n1 = tf.linalg.cross(wrap_vij, wrap_vjk)
        n2 = tf.linalg.cross(wrap_vjk, wrap_vkl)
        n1_norm = tf.norm(tensor=n1)
        n2_norm = tf.norm(tensor=n2)
        if n1_norm == 0.0 or n2_norm == 0.0:
            tf.print(n1_norm, n2_norm)
            raise ValueError('Vectors are linear')
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        cos_d = tf.einsum('ij,ij->i', n1, n2)
        dihedrals = tf.math.acos(cos_d)
        return dihedrals

    if CG is True and cg_positions is None:
        raise ValueError('cg_positions not found.')

    if CG is True and cg_positions is not None:
        v_ij = cg_positions[b2] - cg_positions[b1]
        v_jk = cg_positions[b3] - cg_positions[b2]
        v_kl = cg_positions[b4] - cg_positions[b3]
        wrap_vij = hoomd.htf.wrap_vector(v_ij, box)
        wrap_vjk = hoomd.htf.wrap_vector(v_jk, box)
        wrap_vkl = hoomd.htf.wrap_vector(v_kl, box)

        n1 = np.cross(wrap_vij, wrap_vjk)
        n2 = np.cross(wrap_vjk, wrap_vkl)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)

        if n1_norm == 0.0 or n2_norm == 0.0:
            print(n1_norm, n2_norm)
            raise ValueError('Vectors are linear')

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        cos_d = np.dot(n1, n2)
        cg_dihedrals = np.arccos(cos_d)
        return cg_dihedrals


def sparse_mapping(molecule_mapping, molecule_mapping_index,
                   system=None):
    ''' This will create the necessary indices and values for
    defining a sparse tensor in
    tensorflow that is a mass-weighted ``B x N`` mapping operator.
    where ``B`` is the number of coarse-grained beads.
    This is a slow function and should not be called frequently.

    :param molecule_mapping: This is a list of L x M matrices, where M is the number
        of atoms in the molecule and L is the number of coarse-grained
        beads. These dimensions can be different for different molecules.
        There should be one matrix per molecule.
        The ordering of the atoms should follow what is
        passed in from ``molecule_mapping_index``
    :type molecule_mapping: list of numpy arrays
    :param molecule_mapping_index: This is the output from find_molecules.
         A list of the same length as ``molecule_mapping`` whose elements are lists of atom indices
    :type molecule_mapping_index: list of lists
    :param system: The hoomd system. This is used to get mass values
        for the mapping, if you would like to
        weight by mass of the atoms.

    :return: A sparse tensorflow tensor of dimension M x N,
        where M is the number of molecules and N is number of atoms
    '''
    if type(molecule_mapping[0]) != np.ndarray:
        raise TypeError('molecule_mapping should be list of numpy arrays')
    # get system size
    N = sum([len(m) for m in molecule_mapping_index])
    # get number of output CG bead sites
    B = sum([m.shape[0] for m in molecule_mapping])
    # create indices
    indices = []
    values = []
    total_i = 0

    if len(molecule_mapping_index) != len(molecule_mapping):
        raise ValueError(
            'Length of molecule_mapping_index and molecule_mapping must match')

    for i, (mmi, mm) in enumerate(
            zip(molecule_mapping_index, molecule_mapping)):
        # check dimensions are valid
        if len(mmi) != mm.shape[1]:
            raise ValueError(
                f'Mismatch in shapes of molecule_mapping_index and molecule_mapping at index {i}. '
                f'shape {len(mmi)} is incompatible with {mm.shape}')
        idx = []
        vs = []
        masses = [0 for _ in range(mm.shape[0])]
        # iterate over CG particles
        for i in range(mm.shape[0]):
            # iterate over atoms
            for j in range(mm.shape[1]):
                # check if non-zero
                if mm[i, j] > 0:
                    # index -> CG particle, atom index
                    idx.append([i + total_i, mmi[j]])
                    if system is not None:
                        vs.append(system.particles[mmi[j]].mass)
                    else:
                        vs.append(mm[i, j])
        # now scale values by mases
        if system is not None:
            # now add up masses
            for i in range(len(idx)):
                # get masses from previous values
                masses[idx[i][0] - total_i] += vs[i]
            # make sure things are valid
            assert sum([m == 0 for m in masses]) == 0

            for i in range(len(idx)):
                vs[i] /= masses[idx[i][0] - total_i]
        # all done
        indices.extend(idx)
        values.extend(vs)
        total_i += len(masses)
    assert total_i == B, 'Indices failed!'
    return tf.SparseTensor(
        indices=indices,
        values=np.array(
            values,
            dtype=np.float32),
        dense_shape=[
            B,
            N])
