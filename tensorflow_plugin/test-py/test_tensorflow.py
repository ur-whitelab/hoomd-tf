# -*- coding: iso-8859-1 -*-
# Maintainer: Andrew White

import hoomd;
hoomd.context.initialize();
import hoomd.tensorflow_plugin;
import unittest;
import os;

class test_simple(unittest.TestCase):
    def test_constructor(self):
        sysdef = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[1,2])

        updater = hoomd.tensorflow_plugin.update.tensorflow(4)

if __name__ == '__main__':
    unittest.main(argv = ['test_tensorflow.py', '-v'])
