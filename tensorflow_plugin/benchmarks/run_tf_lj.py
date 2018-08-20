import hoomd, hoomd.md, math
import hoomd.tensorflow_plugin

model_dir = '/tmp/benchmark-lj-potential-model'
tfcompute = hoomd.tensorflow_plugin.tensorflow(model_dir, _debug_mode=False)
N = 1024
rcut = 3.0
hoomd.context.initialize()
system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=1.0),
                                    n=int(math.ceil(N**(1/2))))
nlist = hoomd.md.nlist.cell()
hoomd.md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.nve(group=hoomd.group.all()).randomize_velocities(seed=1, kT=1)
#lj = hoomd.md.pair.lj(r_cut=3.0, nlist=nlist)
#lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
tfcompute.attach(nlist, r_cut=rcut)
log = hoomd.analyze.log(filename='thermo', quantities=['potential_energy', 'temperature'], period=1)
#for i in range(1000):
#    hoomd.run(10)
#    print(i * 10, log.query('potential_energy'), log.query('temperature'))
hoomd.run(1000, profile=True)
