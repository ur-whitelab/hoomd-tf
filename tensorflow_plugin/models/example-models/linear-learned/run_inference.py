import hoomd, hoomd.md, hoomd.tensorflow_plugin

with hoomd.tensorflow_plugin.tfcompute('/tmp/inference',
        bootstrap='/tmp/training') as tfcompute:
    hoomd.context.initialize()
    N = 128
    NN = 32
    rcut = 3.0
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[8,8])
    nlist = hoomd.md.nlist.cell(check_period = 1)
    #notice we no longer compute forces with hoomd
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.nve(
        group=hoomd.group.all()).randomize_velocities(kT=0.2, seed=42)

    tfcompute.attach(nlist, r_cut=rcut)
    hoomd.run(100)