import hoomd, hoomd.md, hoomd.tensorflow_plugin
import tensorflow as tf
with hoomd.tensorflow_plugin.tfcompute('/tmp/eds') as tfcompute:
    hoomd.context.initialize()
    N = 64
    NN = 30
    rcut = 3.0
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[8,8])
    hoomd.md.update.enforce2d()
    nlist = hoomd.md.nlist.cell(check_period = 1)

    lj = hoomd.md.pair.lj(rcut, nlist)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.nvt(kT=1.0,tau=0.5,
        group=hoomd.group.all())
    hoomd.analyze.log(filename='eds.log',
                      quantities = ['momentum', 'temperature', 'time'],
                      period=100,overwrite=True)
    tfcompute.attach(nlist, r_cut=rcut)
    hoomd.dump.gsd(filename='saveall_running.gsd', overwrite=True, period=100, group=hoomd.group.all(), dynamic=['attribute', 'momentum', 'topology'])
    hoomd.run(200000)
