# Test qualification

Run from the repository root with a dedicated virtual environment:

```bash
python -m pip install -e '.[dev,api]'
PYTHONPATH=src python -m pytest --import-mode=importlib -q \
  tests/access_auditor tests/blast tests/config tests/conflicts \
  tests/consistency tests/models \
  tests/http_api/test_new_endpoints.py tests/registry/test_store.py
```

855 tests passed in the 2026-09-13 Python 3.13 qualification. This includes
six component contract/Goodhart suites whose APIs exist in `arbiter.*`, plus
the endpoint and registry regressions. Imports address the shipped package;
consistency tests get an isolated store per case.

The remaining generated CLI, HTTP API, OTLP subscriber, and human-gate suites
still import old top-level module names and require interfaces not exported by
the current package. For example, the CLI suite expects `arbiter_group` and
`cmd_trust_show`; the shipped CLI uses Click's `main` and subcommands. The HTTP
suite expects standalone route functions; the shipped API registers handlers
inside `create_app`. `human_gate` has no corresponding shipped module.

These artifacts were retained in commit `5060551` (Pact pipeline artifacts).
They remain visible to ordinary full-tree pytest collection, which is **not
green**. The historical claim of 1,335 passing tests is not verified by the
current tree. Porting their intent to the current architecture remains work;
import aliases alone cannot supply absent interfaces.

To inspect every retained artifact, run:

```bash
PYTHONPATH=src python -m pytest --import-mode=importlib -q tests
```
