R""" Check to ensure that we have a compatible version of TensorFlow.
"""

from __future__ import print_function
import sys
try:
    import tensorflow
except ImportError:
    print('Error: Could not find tensorflow')
    exit(1)
from pkg_resources import parse_version

# get and parse the version of the detected TF version
vtf = parse_version(tensorflow.__version__)
vreq = parse_version(sys.argv[1])

# check if major version match
if vtf.base_version.split('.')[0] == vreq.base_version.split('.')[0]:
    if vtf >= vreq:
        exit()
print('Error: tensorflow version incompatible. Found', vtf, 'but need', vreq)
exit(1)
