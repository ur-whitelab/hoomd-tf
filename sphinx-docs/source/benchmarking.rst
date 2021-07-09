.. _benchmarking:

Benchmarking
------------

Benchmarking is automatically done in pytest. To compute the reported results,
run the following command from the `test-py` directory. You may need to install `ansi2html`

New benchmarks should be committed with any major changes to track
potential regressions.

.. code:: bash

    pytest-benchmark compare | ansi2html  > ../../sphinx-docs/source/benchmarks.html

:download:`benchmark results<benchmarks.html>`