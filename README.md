Python implementation of Histogrammar
=====================================

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.61418.svg)](http://dx.doi.org/10.5281/zenodo.61418)

See [histogrammar.org](http://histogrammar.org) for a complete introduction to Histogrammar.

This is a pure Python implementation for Python versions 2.7, and 3.4, 3.5, 3.6, 3.7

Latest Python release: v1.0.10 (Sep 2019).

Support of Histogrammar is provided on a best-effort basis only.


Installation
============

Histogrammar has a standard Python install script. Run

```bash
python setup.py install
```

with `sudo` if superuser permissions are needed or `--home=~` to install in a home directory. The latter option requires an appropriate `PYTHONPATH`.

This package has no explicit dependencies, though it can use back-ends like Numpy and front-ends like Matplotlib and Bokeh if they are on your system. If not, you'll get an `ImportError` at the time you try to call these methods.

Status
======

![Build status](https://travis-ci.org/histogrammar/histogrammar-python.svg)

The tests are thorough.

   * `basic.py` were written during development to test all features as they were added.
   * `testnumpy.py` tests numerical agreement between the conventional implementation and the Numpy implementation, which are very different. Also tests much larger datasets and infinity/NaN handling.
     * contrary to its name, `testnumpy.py` also compares its implementation with the literal code given in [the specification](http://histogrammar.org/docs/specification/) as well.
   * `testrootcling.py` applies all of the Numpy tests to the ROOT/Cling implementation.
   * `testgpu.py` applies all of the Numpy tests to the CUDA GPU implementation.

Primitive implementation is mature. CUDA implementation has begun.

| Primitive         | Pure Python | Numpy | ROOT JIT | CUDA GPU   |
|:------------------|:------------|:------|:---------|:-----------|
| Count             | done        | done  | done     | done       |
| Sum               | done        | done  | done     | done       |
| Average           | done        | done  | done     | done       |
| Deviate           | done        | done  | done     | done       |
| Minimize          | done        | done  | done     | done       |
| Maximize          | done        | done  | done     | done       |
| Bag               | done        | done  | done     | impossible |
| Bin               | done        | done  | done     | done       |
| SparselyBin       | done        | done  | done     | impossible |
| CentrallyBin      | done        | done  | done     | done       |
| IrregularlyBin    | done        | done  | done     | done       |
| Categorize        | done        | done  | done     | impossible |
| Fraction          | done        | done  | done     | done       |
| Stack             | done        | done  | done     | done       |
| Select            | done        | done  | done     | done       |
| Label             | done        | done  | done     | done       |
| UntypedLabel      | done        | done  | done     | done       |
| Index             | done        | done  | done     | done       |
| Branch            | done        | done  | done     | done       |

* "impossible" for CUDA GPU means that the primitive requires a non-constant memory allocation (hashmaps in all three cases). There must be ways of doing this by preallocating more space than is needed, but I'm not going to get into that for this round.
