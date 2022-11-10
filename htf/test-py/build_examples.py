# Copyright (c) 2020 HOOMD-TF Developers

import tensorflow as tf
import os
import hoomd
import gsd.hoomd
import hoomd.htf as htf
import pickle

def generic_square_lattice(lattice_constant, n_replicas, device):
    '''Helper function to make a 2D square lattice of generic "A" particles.
    :param lattice_constant: lattice vector in HOOMD distance units.
    :type lattice_constant: float

    :param n_replicas: number of times to replicate the lattice in both directions.
    :type n_replicas: list of (2 or 3) integers

    :param device: HOOMD device that will be used for system configuration
    :type device: HOOMD device (e.g. from 'hoomd.device.CPU()')

    :return: simulation object with initial lattice positions of generic particles'''
    snap = gsd.hoomd.Snapshot()
    snap.particles.N = 1
    snap.particles.types = ['A']
    snap.particles.typeid = [0]
    snap.particles.position = [[lattice_constant/2, lattice_constant/2, lattice_constant/2]]
    snap.configuration.box = [lattice_constant, lattice_constant, lattice_constant, 0, 0, 0]
    sim = hoomd.Simulation(device)
    sim.create_state_from_snapshot(snap)
    sim.state.replicate(*n_replicas)
    return sim

class SimplePotential(htf.SimModel):
    def compute(self, nlist, positions):
        nlist = nlist[:, :, :3]
        neighs_rs = tf.norm(tensor=nlist, axis=2, keepdims=True)
        # no need to use netwon's law because nlist should be double counted
        fr = tf.multiply(-1.0,
                         tf.multiply(tf.math.reciprocal(neighs_rs),
                                     nlist),
                         name='nan-pairwise-forces')
        zeros = tf.zeros_like(nlist)
        real_fr = tf.where(tf.math.is_finite(fr), fr, zeros,
                           name='pairwise-forces')
        forces = tf.reduce_sum(input_tensor=real_fr, axis=1, name='forces')
        return forces


class BenchmarkPotential(htf.SimModel):
    def compute(self, nlist):
        rinv = htf.nlist_rinv(nlist)
        energy = rinv
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class NoForceModel(htf.SimModel):
    def compute(self, nlist, positions):
        neighs_rs = tf.norm(tensor=nlist[:, :, :3], axis=2)
        energy = tf.math.divide_no_nan(tf.ones_like(
            neighs_rs, dtype=neighs_rs.dtype),
            neighs_rs, name='energy')
        pos_norm = tf.norm(tensor=positions, axis=1)
        return energy, pos_norm


class TensorSaveModel(htf.SimModel):
    def compute(self, nlist, positions):
        pos_norm = tf.norm(tensor=positions, axis=1)
        return pos_norm


class WrapModel(htf.SimModel):
    def compute(self, nlist, positions, box):
        p1 = positions[0, :3]
        p2 = positions[-1, :3]
        r = p1 - p2
        rwrap = htf.wrap_vector(r, box)
        # TODO: Smoke test. Think of a better test.
        return rwrap


class BenchmarkNonlistModel(htf.SimModel):
    def compute(self, nlist, positions, box):
        ps = tf.norm(tensor=positions, axis=1)
        energy = tf.math.divide_no_nan(1., ps)
        forces = htf.compute_positions_forces(positions, energy)
        return forces


class LJModel(htf.SimModel):
    def compute(self, nlist, positions, box):
        # get r
        rinv = htf.nlist_rinv(nlist)
        inv_r6 = rinv**6
        # pairwise energy. Double count -> divide by 2
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class LJTypedModel(htf.SimModel):
    def setup(self):
        self.avg_rdfa = tf.keras.metrics.MeanTensor()
        self.avg_rdfb = tf.keras.metrics.MeanTensor()

    def compute(self, nlist, positions, box):
        # get r
        rinv = htf.nlist_rinv(nlist)
        inv_r6 = rinv**6
        # pairwise energy. Double count -> divide by 2
        p_energy = 1e-10 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces = htf.compute_nlist_forces(nlist, energy)
        rdfa, rs = htf.compute_rdf(
            nlist, [0, 10], positions[:, 3], type_i=0, type_j=1)
        rdfb, rs = htf.compute_rdf(
            nlist, [0, 10], positions[:, 3], type_i=1, type_j=0)
        # compute running mean
        self.avg_rdfa.update_state(rdfa)
        self.avg_rdfb.update_state(rdfb)
        return forces


class LJVirialModel(htf.SimModel):
    def compute(self, nlist, positions, box):
        # get r
        rinv = htf.nlist_rinv(nlist)
        inv_r6 = rinv**6
        # pairwise energy. Double count -> divide by 2
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces_and_virial = htf.compute_nlist_forces(
            nlist, energy, virial=True)
        return forces_and_virial


class EDSModel(htf.SimModel):
    def setup(self, set_point):
        self.cv_avg = tf.keras.metrics.Mean()
        self.eds_bias = htf.EDSLayer(set_point, 5, 1 / 5)

    def compute(self, nlist, positions, box):
        # get distance from center
        rvec = htf.wrap_vector(positions[0, :3], box)
        cv = tf.norm(tensor=rvec)
        self.cv_avg.update_state(cv)
        alpha = self.eds_bias(cv)
        # eds + harmonic bond
        energy = (cv - 5) ** 2 + cv * alpha
        # energy  = cv^2 - 6cv + cv * alpha + C
        # energy = (cv - (3 + alpha / 2))^2 + C
        # alpha needs to be = 4
        forces = htf.compute_positions_forces(positions, energy)
        return forces, alpha


class MolFeatureModel(htf.MolSimModel):

    def mol_compute(self, nlist, positions, mol_nlist, mol_pos, box):
        r = htf.mol_bond_distance(mol_pos, 2, 1, box=box)
        a = htf.mol_angle(mol_pos, 1, 2, 3, box=box)
        d = htf.mol_dihedral(mol_pos, 1, 2, 3, 4, box=box)
        avg_r = tf.reduce_mean(input_tensor=r)
        avg_a = tf.reduce_mean(input_tensor=a)
        avg_d = tf.reduce_mean(input_tensor=d)
        return avg_r, avg_a, avg_d


class CGModel(htf.SimModel):

    def compute(self):

        import MDAnalysis as mda

        jfile = os.path.join(os.path.dirname(__file__), 'test_cgmap.json')

        u2 = mda.Universe(os.path.join(
            os.path.dirname(__file__), 'test_segA_xH.pdb'))
        u1 = mda.Universe(os.path.join(
            os.path.dirname(__file__), 'test_segA.pdb'))

        cg_feats = htf.compute_cg_graph(
            DSGPM=True,
            infile=jfile,
            group_atoms=True,
            u_no_H=u2,
            u_H=u1)

        return cg_feats


class CustomNlist(htf.SimModel):
    def compute(self, nlist, positions, box):
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)

        # compute nlist
        cnlist = htf.compute_nlist(
            positions[:, :3], self.r_cut, self.nneighbor_cutoff, htf.box_size(box))
        cr = tf.norm(tensor=cnlist[:, :, :3], axis=2)
        return r, cr


class MappedNlist(htf.SimModel):
    def my_map(pos, box):
        x = tf.reduce_mean(pos[:, :3], axis=0, keepdims=True)
        cg1 = tf.concat((x, tf.zeros((1, 1), dtype=x.dtype)), -1)
        cg2 = tf.convert_to_tensor([[0, 0, 0.1, 1]], dtype=x.dtype)
        return tf.concat((cg1, cg2), axis=0)

    def compute(self, nlist, positions, box):
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)

        # compute nlist
        nlist, cnlist = self.mapped_nlist(nlist)
        return positions, nlist, cnlist


class NlistNN(htf.SimModel):
    def setup(self, dim, top_neighs):
        self.dense1 = tf.keras.layers.Dense(dim)
        self.dense2 = tf.keras.layers.Dense(dim)
        self.last = tf.keras.layers.Dense(1)
        self.top_neighs = top_neighs

    def compute(self, nlist, positions, box):
        rinv = htf.nlist_rinv(nlist)
        # closest neighbors have largest value in 1/r, take top
        top_n = tf.sort(rinv, axis=1, direction='DESCENDING')[
            :, :self.top_neighs]
        # run through NN
        # make sure shape is definite
        top_n = tf.reshape(top_n, (-1, self.top_neighs))
        x = self.dense1(top_n)
        x = self.dense2(x)
        energy = self.last(x)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class WCA(htf.SimModel):
    def setup(self):
        self.wca = htf.WCARepulsion(0.5)

    def compute(self, nlist):
        energy = self.wca(nlist)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class RBF(htf.SimModel):
    def setup(self, low, high, count):
        self.rbf = htf.RBFExpansion(low, high, count)
        self.dense = tf.keras.layers.Dense(1)

    def compute(self, nlist):
        r = htf.safe_norm(nlist[:, :3], axis=2)
        rbf = self.rbf(r)
        energy = tf.reduce_sum(self.dense(rbf))
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class TrainModel(htf.SimModel):
    def setup(self, dim, top_neighs):
        self.dense1 = tf.keras.layers.Dense(dim)
        self.dense2 = tf.keras.layers.Dense(dim)
        self.last = tf.keras.layers.Dense(1)
        self.top_neighs = top_neighs
        self.output_zero = False

    def compute(self, nlist, positions, training):
        rinv = htf.nlist_rinv(nlist)
        # closest neighbors have largest value in 1/r, take top
        top_n = tf.sort(rinv, axis=1, direction='DESCENDING')[
            :, :self.top_neighs]
        # run through NN
        x = self.dense1(top_n)
        x = self.dense2(x)
        energy = self.last(x)
        if training:
            energy *= 2

        forces = htf.compute_nlist_forces(nlist, energy)
        if self.output_zero:
            energy *= 0.
        return forces, tf.reduce_sum(energy)


class LJRunningMeanModel(htf.SimModel):
    def setup(self):
        self.avg_energy = tf.keras.metrics.Mean()

    def compute(self, nlist, positions, box):
        # get r
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)
        # compute 1 / r while safely treating r = 0.
        # pairwise energy. Double count -> divide by 2
        inv_r6 = tf.math.divide_no_nan(1., r**6)
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        # compute running mean
        self.avg_energy.update_state(energy)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class LJRDF(htf.SimModel):
    def setup(self):
        self.avg_rdf = tf.keras.metrics.MeanTensor()

    def compute(self, nlist, positions, box):
        # get r
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)
        # compute 1 / r while safely treating r = 0.
        # pairwise energy. Double count -> divide by 2
        inv_r6 = tf.math.divide_no_nan(1., r**6)
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # get rdf
        rdf, rs = htf.compute_rdf(nlist, [3, 5], positions[:, 3])
        # also compute without types
        _, _ = htf.compute_rdf(nlist, [3, 5])
        # compute running mean
        self.avg_rdf.update_state(rdf)
        forces = htf.compute_nlist_forces(nlist, p_energy)
        return forces


class LJMolModel(htf.MolSimModel):
    def mol_compute(self, nlist, positions, mol_nlist, mol_positions, box):
        # assume particle (w) is 0
        r = tf.norm(mol_nlist, axis=3)
        rinv = tf.math.divide_no_nan(1.0, r)
        mol_p_energy = 4.0 / 2.0 * (rinv**12 - rinv**6)
        total_e = tf.reduce_sum(input_tensor=mol_p_energy)
        forces = htf.compute_nlist_forces(nlist, total_e)
        return forces


class PrintModel(htf.SimModel):
    def compute(self, nlist, positions, box):
        # get r
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)
        # compute 1 / r while safely treating r = 0.
        # pairwise energy. Double count -> divide by 2
        inv_r6 = tf.math.divide_no_nan(1., r**6)
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        tf.print(energy)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class LJLayer(tf.keras.layers.Layer):
    def __init__(self, sig, eps):
        super().__init__(self, name='lj')
        self.start = [sig, eps]
        self.w = self.add_weight(
            shape=[2],
            initializer=tf.constant_initializer([sig, eps]),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True,
            name='lj-params'

        )

    def call(self, r):
        r6 = tf.math.divide_no_nan(self.w[1]**6, r**6)
        energy = self.w[0] * 4.0 * (r6**2 - r6)
        # divide by 2 to remove double count
        return energy / 2.

    def get_config(self):
        c = {}
        c['sig'] = self.start[0]
        c['eps'] = self.start[1]
        return c


class TrainableGraph(htf.SimModel):
    def setup(self):
        self.lj = LJLayer(1.0, 1.0)

    def compute(self, nlist, positions, box):
        # get r
        r = htf.safe_norm(tensor=nlist[:, :, :3], axis=2)
        p_energy = self.lj(r)
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces
