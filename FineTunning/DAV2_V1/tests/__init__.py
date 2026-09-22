"""Marks `tests/` as a real package.

Eight test modules do `from tests.stub_dav2 import use_stub`.  Without this
file `tests` is only a PEP-420 namespace package, and the import then depends on
nothing else on `sys.path` owning that very common name.  Locally nothing does,
so the suite passed; on the Kaggle image something ships a regular `tests`
package, which wins the resolution outright (a directory with `__init__.py`
stops the scan, while namespace portions do not) and collection died with
`ModuleNotFoundError: No module named 'tests.stub_dav2'` on all eight.

With this file `tests` is a regular package anchored at the repo's own path
entry, so it resolves the same way everywhere.
"""
