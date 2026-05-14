nemopy
======

A column-vector-first NumPy wrapper. Vectors are always ``(n, 1)``;
matrices are constructed column-by-column; arithmetic raises a clear
``ShapeError`` when shapes disagree instead of silently broadcasting.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   examples
