import hoomd, hoomd.md, hoomd.tensorflow_plugin

with hoomd.tensorflow_plugin.tfcompute('/tmp/eds') as tfcompute:
    hoomd.context.initialize()
    N = 64
    NN = 16
    rcut = 3.0
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[8,8])
    nlist = hoomd.md.nlist.cell(check_period = 1)
    lj = hoomd.md.pair.lj(rcut, nlist)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.nve(
        group=hoomd.group.all()).randomize_velocities(kT=0.2, seed=42)

    tfcompute.attach(nlist, r_cut=rcut)
    hoomd.run(100)