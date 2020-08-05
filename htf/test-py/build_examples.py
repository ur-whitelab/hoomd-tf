# Copyright (c) 2018 Andrew White at the University of Rochester
# This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

import tensorflow as tf
import os
import hoomd.htf as htf
import pickle


class SimplePotential(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        nlist = nlist[:, :, :3]
        neighs_rs = tf.norm(tensor=nlist, axis=2, keepdims=True)
        # no need to use netwon's law because nlist should be double counted
        fr = tf.multiply(-1.0, tf.multiply(tf.math.reciprocal(neighs_rs), nlist),
                         name='nan-pairwise-forces')
        zeros = tf.zeros_like(nlist)
        real_fr = tf.where(tf.math.is_finite(fr), fr, zeros,
                           name='pairwise-forces')
        forces = tf.reduce_sum(input_tensor=real_fr, axis=1, name='forces')
        return forces


def benchmark_gradient_potential(directory='/tmp/benchmark-gradient-potential-model'):
    graph = htf.SimModel(1024, 64)
    nlist = graph.nlist[:, :, :3]
    # get r
    r = tf.norm(tensor=nlist, axis=2)
    # compute 1 / r while safely treating r = 0.
    energy = tf.reduce_sum(input_tensor=tf.math.divide_no_nan(1., r), axis=1)
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces,
               model_directory=directory)


def gradient_potential(directory='/tmp/test-gradient-potential-model'):
    graph = htf.SimModel(9 - 1)
    with tf.compat.v1.name_scope('force-calc') as scope:
        nlist = graph.nlist[:, :, :3]
        neighs_rs = tf.norm(tensor=nlist, axis=2)
        energy = 0.5 * tf.math.divide_no_nan(numerator=tf.ones_like(
            neighs_rs, dtype=neighs_rs.dtype), denominator=neighs_rs,
            name='energy')
    forces = graph.compute_forces(energy)
    graph.save(force_tensor=forces,
               model_directory=directory,
               out_nodes=[energy])


class NoForceModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        neighs_rs = tf.norm(tensor=nlist[:, :, :3], axis=2)
        energy = tf.math.divide_no_nan(tf.ones_like(
            neighs_rs, dtype=neighs_rs.dtype),
            neighs_rs, name='energy')
        pos_norm = tf.norm(tensor=positions, axis=1)
        return energy, pos_norm


class TensorSaveModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        pos_norm = tf.norm(tensor=positions, axis=1)
        return pos_norm


class WrapModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        p1 = positions[0, :3]
        p2 = positions[-1, :3]
        r = p1 - p2
        rwrap = htf.wrap_vector(r, box)
        # TODO: Smoke test. Think of a better test.
        return rwrap


def mol_force(directory='/tmp/test-mol-force-model'):
    graph = htf.SimModel(0, output_forces=False)
    graph.build_mol_rep(3)
    f = tf.norm(tensor=graph.mol_forces, axis=0)
    graph.save(directory, out_nodes=[f])
    return directory


class BenchmarkNonlistGraph(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        ps = tf.norm(tensor=positions, axis=1)
        energy = tf.math.divide_no_nan(1., ps)
        forces = htf.compute_positions_forces(positions, energy)
        return forces


class LJModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        # get r
        rinv = htf.nlist_rinv(nlist)
        inv_r6 = rinv**6
        # pairwise energy. Double count -> divide by 2
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class LJVirialModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        # get r
        rinv = htf.nlist_rinv(nlist)
        inv_r6 = rinv**6
        # pairwise energy. Double count -> divide by 2
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces = htf.compute_nlist_forces(nlist, energy, virial=True)
        return forces


def eds_graph(directory='/tmp/test-lj-eds'):
    graph = htf.SimModel(0)
    # get distance from center
    rvec = graph.wrap_vector(graph.positions[0, :3])
    cv = tf.norm(tensor=rvec)
    cv_mean = graph.running_mean(cv, name='cv-mean')
    alpha = htf.eds_bias(cv, 4, 5, cv_scale=1 / 5, name='eds')
    alpha_mean = graph.running_mean(alpha, name='alpha-mean')
    # eds + harmonic bond
    energy = (cv - 5) ** 2 + cv * alpha
    # energy  = cv^2 - 6cv + cv * alpha + C
    # energy = (cv - (3 + alpha / 2))^2 + C
    # alpha needs to be = 4
    forces = graph.compute_forces(energy, positions=True)
    graph.save(
        force_tensor=forces,
        model_directory=directory,
        out_nodes=[
            cv_mean,
            alpha_mean])
    return directory


def mol_features_graph(directory='/tmp/test-mol-features'):
    graph = htf.SimModel(50, output_forces=False)
    graph.build_mol_rep(6)
    mol_pos = graph.mol_positions
    r = htf.mol_bond_distance(mol_pos, 2, 1)
    a = htf.mol_angle(mol_pos, 1, 2, 3)
    d = htf.mol_dihedral(mol_pos, 1, 2, 3, 4)
    avg_r = tf.reduce_mean(input_tensor=r)
    avg_a = tf.reduce_mean(input_tensor=a)
    avg_d = tf.reduce_mean(input_tensor=d)
    graph.save_tensor(avg_r, 'avg_r')
    graph.save_tensor(avg_a, 'avg_a')
    graph.save_tensor(avg_d, 'avg_d')
    graph.save(model_directory=directory)
    return directory


def run_traj_graph(directory='/tmp/test-run-traj'):
    graph = htf.SimModel(128)
    nlist = graph.nlist[:, :, :3]
    r = tf.norm(tensor=nlist, axis=2)
    # compute 1 / r while safely treating r = 0.
    # pairwise energy. Double count -> divide by 2
    inv_r6 = tf.math.divide_no_nan(1., r**6)
    p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
    # sum over pairwise energy
    energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
    forces = graph.compute_forces(energy)
    avg_energy = graph.running_mean(tf.reduce_sum(input_tensor=energy, axis=0),
                                    'average-energy')
    graph.save(force_tensor=forces, model_directory=directory,
               out_nodes=[avg_energy])
    return directory


class CustomNlist(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)

        # compute nlist
        cnlist = htf.compute_nlist(
            positions[:, :3], self.r_cut, self.nneighbor_cutoff, htf.box_size(box))
        cr = tf.norm(tensor=cnlist[:, :, :3], axis=2)
        return r, cr


class LJRunningMeanModel(htf.SimModel):
    def setup(self):
        self.avg_energy = tf.keras.metrics.Mean()

    def compute(self, nlist, positions, box, sample_weight):
        # get r
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)
        # compute 1 / r while safely treating r = 0.
        # pairwise energy. Double count -> divide by 2
        inv_r6 = tf.math.divide_no_nan(1., r**6)
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # sum over pairwise energy
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        # compute running mean
        self.avg_energy.update_state(energy, sample_weight=sample_weight)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces


class LJRDF(htf.SimModel):
    def setup(self):
        self.avg_rdf = tf.keras.metrics.Mean()

    def compute(self, nlist, positions, box, sample_weight):
        # get r
        r = tf.norm(tensor=nlist[:, :, :3], axis=2)
        # compute 1 / r while safely treating r = 0.
        # pairwise energy. Double count -> divide by 2
        inv_r6 = tf.math.divide_no_nan(1., r**6)
        p_energy = 4.0 / 2.0 * (inv_r6 * inv_r6 - inv_r6)
        # get rdf
        rdf, rs = htf.compute_rdf(nlist, positions, [3, 5])
        # compute running mean
        self.avg_rdf.update_state(rdf, sample_weight=sample_weight)
        forces = htf.compute_nlist_forces(nlist, p_energy)
        return forces


def lj_mol(NN, MN, directory='/tmp/test-lj-mol'):
    graph = htf.SimModel(NN)
    graph.build_mol_rep(MN)
    # assume particle (w) is 0
    r = graph.safe_norm(graph.mol_nlist, axis=3)
    rinv = tf.math.divide_no_nan(1.0, r)
    mol_p_energy = 4.0 / 2.0 * (rinv**12 - rinv**6)
    total_e = tf.reduce_sum(input_tensor=mol_p_energy)
    forces = graph.compute_forces(total_e)
    graph.save(force_tensor=forces, model_directory=directory, out_nodes=[])
    return directory


class PrintModel(htf.SimModel):
    def compute(self, nlist, positions, box, sample_weight):
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
    def __init__(self, NN, **kwargs):
        super().__init__(NN, **kwargs)
        self.lj = LJLayer(1.0, 1.0)

    def compute(self, nlist, positions, box, sample_weight):
        # get r
        r = htf.safe_norm(tensor=nlist[:, :, :3], axis=2)
        p_energy = self.lj(r)
        energy = tf.reduce_sum(input_tensor=p_energy, axis=1)
        forces = htf.compute_nlist_forces(nlist, energy)
        return forces
