.. _unit_tests:

Unit Tests
----------


You can run the unit tests directly via ``python htf/test-py/test_tensorflow.py``,
``python htf/test-py/test_utils.py``, etc. Be careful using automated tools like `pytest` on the whole directory. It can automatically start multiple processes
which doesn't work well with tensorflow. It works fine on individual files.

