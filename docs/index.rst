nemopy
======

A column-vector-first NumPy wrapper. Vectors are always ``(n, 1)``;
matrices are constructed column-by-column; arithmetic raises a clear
``ShapeError`` when shapes disagree instead of silently broadcasting.

Installation
------------

nemopy is not on PyPI yet; install it from git. It requires Python ``>=3.10``
(the repo pins 3.14 via ``.python-version``) and is developed with `uv
<https://docs.astral.sh/uv/>`_. There are two modes.

**Mode A — pure-Python** (core types, shape-guarded arithmetic, Tier-2
decompositions ``svd``/``qr``/``lu``/``cholesky``/``eigh``, and stats)::

   uv add "git+https://github.com/nster101/nemopy"

**Mode B — Rust-enabled** unlocks the Tier-3 surface (advanced decompositions,
elimination, and future LP / network / Markov features). Build the optional
``nemopy._rust_core`` extension after cloning::

   uv sync
   scripts/build_rust.sh   # canonical build command (wraps maturin)

Without the extension, Tier-3 methods raise a clear ``ImportError`` instead of
falling back. Check which mode is active::

   python -c "import nemopy._core as c; print('rust active:', c._RUST is not None)"

See the project ``README`` for the full install guide.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   examples
