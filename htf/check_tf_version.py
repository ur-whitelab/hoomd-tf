''' Check to ensure that we have a compatible version of TensorFlow.
1;5202;0c'''

from __future__ import print_function
import sys
try:
    import tensorflow
except ImportError:
    print('-- Error: Could not find tensorflow')
    exit(1)
from pkg_resources import parse_version

# Ensure copmiler versions match
if len(sys.argv) == 3 and len(sys.argv[2]) > 1:
    vcomp = parse_version(tensorflow.__compiler_version__)
    vcomp_req = parse_version(sys.argv[2])

    if vcomp.base_version.split('.')[0] != vcomp_req.base_version.split('.')[0]:
        print('-- Error: tensorflow was compiled with compiler version {} '
              'but you are trying to build with {} (major mismatch). To force'
              'compilation add IGNORE_HTF_COMPILER flag'.format(vcomp, vcomp_req))
        exit(1)
    elif vcomp != vcomp_req:
        print('-- Warning: tensorflow was compiled with compiler version {} '
              'but you are trying to build with {} (minor mismatch)'.format(vcomp, vcomp_req))

# get and parse the version of the detected TF version
vtf = parse_version(tensorflow.__version__)
vreq = parse_version(sys.argv[1])

# check if major version match
if vtf.base_version.split('.')[0] == vreq.base_version.split('.')[0]:
    if vtf >= vreq:
        exit()
print('-- Error: tensorflow version incompatible. Found', vtf, 'but need', vreq)
exit(1)
