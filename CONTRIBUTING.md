# Contributing to Hypergrid

Thank you for taking the time to contribute! This document covers everything you need to get started.

---

## Table of contents

- [Getting started](#getting-started)
- [Project layout](#project-layout)
- [Running the tests](#running-the-tests)
- [Code style](#code-style)
- [Adding a new feature](#adding-a-new-feature)
- [Building the documentation](#building-the-documentation)
- [Submitting a pull request](#submitting-a-pull-request)
- [Commit message conventions](#commit-message-conventions)
- [Reporting a bug](#reporting-a-bug)

---

## Getting started

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (recommended) — or pip

### Fork and clone

```bash
git clone https://github.com/Clems6323/hypergrid.git
cd hypergrid
```

### Install in development mode

```bash
# Install the package and all dev dependencies (pytest, ruff, pandas)
uv sync --extra dev

# Activate the virtual environment
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

If you prefer pip:

```bash
pip install -e ".[dev]"
```

---

## Project layout

```
hypergrid/
├── base/               # Concrete grid classes and base ABCs
│   ├── _base_tensor.py       # BaseTensorHypergrid — shared fit/update logic + mixin chain
│   ├── dense_tensor_hypergrid.py
│   ├── sparse_tensor_hypergrid.py
│   ├── static_hypergrid.py
│   ├── adaptive_hypergrid.py
│   └── temporal_hypergrid.py
├── mixin/              # Opt-in capabilities (rebin, compare, embed, visualize, stats)
├── storage/            # DictStorage, DenseTensorStorage, SparseTensorStorage
└── utils/              # Edge computation (compute_edges, compute_bins_1d)

tests/                  # pytest test suite (mirrors the source tree)
docs/                   # MkDocs source (Markdown + example notebooks)
```

All public grid classes inherit from `BaseTensorHypergrid`, which wires together the mixins. New capabilities should be added as mixins rather than modifying the base classes directly.

---

## Running the tests

```bash
# Run the full test suite
uv run pytest

# Run a specific file or test
uv run pytest tests/test_describe.py
uv run pytest tests/test_describe.py::TestDescribeBasic::test_default_rows

# With verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x
```

The test suite must pass on Python 3.10–3.13 before a PR is merged. The CI matrix runs all four versions automatically on every push.

---

## Code style

Hypergrid uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (line length: 100).

```bash
# Check for lint errors
uv run ruff check hypergrid/

# Auto-fix what can be fixed
uv run ruff check --fix hypergrid/

# Format the code
uv run ruff format hypergrid/
```

A few conventions to follow:

- **No comments by default.** Only add a comment when the *why* is non-obvious (a hidden constraint, a subtle invariant, a workaround for a specific bug). Do not describe what the code does.
- **Docstrings use NumPy style** and are required on all public classes and methods. Keep them concise; full detail belongs in the narrative documentation.
- **No backwards-compatibility shims.** If something is unused, delete it.

---

## Adding a new feature

1. **Open an issue first** for anything beyond a small bug fix. This avoids wasted effort if the direction isn't right.
2. **Add a mixin** if the feature is a new capability (e.g. a new statistical method). Wire it into `BaseTensorHypergrid` in `hypergrid/base/_base_tensor.py` and export it from `hypergrid/mixin/__init__.py`.
3. **Add tests** in `tests/`. Every new public method needs at least: a basic correctness test, an edge-case test (empty grid, single bin, 1-D input), and a cross-backend test (DenseHypergrid + SparseHypergrid).
4. **Update the docstring** on the method and, if relevant, the narrative documentation under `docs/`.
5. **Update the example notebook** `docs/examples/01_basic.ipynb` if the change affects the core workflow.

---

## Building the documentation

```bash
# Install docs dependencies
uv sync --extra docs

# Serve locally with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

The docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and [mkdocstrings](https://mkdocstrings.github.io/). API documentation is generated automatically from NumPy-style docstrings — keep them up to date.

---

## Submitting a pull request

1. Create a branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
2. Make your changes, add tests, and confirm the suite passes:
   ```bash
   uv run pytest
   uv run ruff check hypergrid/
   ```
3. Push and open a pull request against `main`. Fill in the PR template — in particular the **test plan** section.
4. At least one approval is required before merging.

---

## Commit message conventions

Commits follow the [Gitmoji](https://gitmoji.dev/) convention. Use the emoji that best describes the intent:

| Emoji | Code | Use for |
|---|---|---|
| ✨ | `:sparkles:` | New feature |
| 🐛 | `:bug:` | Bug fix |
| ♻️ | `:recycle:` | Refactor (no behaviour change) |
| ✅ | `:white_check_mark:` | Adding or updating tests |
| 📝 | `:memo:` | Documentation only |
| 🔧 | `:wrench:` | Configuration / tooling |
| 👷 | `:construction_worker:` | CI/CD changes |
| ⚡️ | `:zap:` | Performance improvement |
| 🗑️ | `:wastebasket:` | Removing code or files |

Example:

```
:sparkles: add entropy() method to StatsMixin
```

---

## Reporting a bug

Use the [bug report template](https://github.com/Clems6323/hypergrid/issues/new?template=bug_report.md). Please include a **minimal reproducible example** and your environment details (Python version, OS, `pyHypergrid` version). Issues without a reproduction case may be closed.
