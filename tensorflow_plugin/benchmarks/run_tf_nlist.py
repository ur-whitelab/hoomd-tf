import hoomd, hoomd.md, math
import hoomd.tensorflow_plugin

model_dir = '/tmp/benchmark-gradient-potential-model'
tfcompute = hoomd.tensorflow_plugin.tfcompute(model_dir)
N = 1024
rcut = 5.0

hoomd.context.initialize()
system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                    n=int(math.ceil(N**(1/2))))
nlist = hoomd.md.nlist.cell()
hoomd.md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.nve(group=hoomd.group.all())
tfcompute.attach(nlist, r_cut=rcut)
hoomd.run(1000, profile=True)
