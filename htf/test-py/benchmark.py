def test_lj_benchmark(benchmark):

    import hoomd.htf as htf
    from hoomd.htf import tfcompute
    import hoomd
    import hoomd.md
    import hoomd.dump
    import hoomd.group
    import numpy as np
    import tensorflow as tf

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


    N = 256
    NN = 64
    model = LJModel(NN)
    tfcompute = htf.tfcompute(model)
    hoomd.context.initialize('')
    rcut = 3.0
    sqrt_N = int(np.sqrt(N))

    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[sqrt_N, sqrt_N])
    nlist = hoomd.md.nlist.cell(check_period=1)
    # basic LJ forces from Hoomd
    lj = hoomd.md.pair.lj(rcut, nlist)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.0, seed=42)
    # equilibrate for 4k steps first
    hoomd.run(4000)
    # now attach the LJ force model
    tfcompute.attach(nlist,
                     r_cut=rcut)

    # train on 1k timesteps
    benchmark.pedantic(hoomd.run, (1000,), rounds=5, iterations=1)
