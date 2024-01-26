# Contributing to pymc_bart
This document outlines only the most common contributions. More information coming soon.
We welcome a wide range of contributions, not only code!

## Reporting issues
If you encounter any bug or incorrect behaviour while using pymc_bart,
please report an issue to our [issue tracker](https://github.com/pymc-devs/pymc-bart/issues).
Please include any supporting information, in particular the version of
pymc_bart that you are using.
The issue tracker has several templates available to help in writing the issue
and including useful supporting information.

## Contributing code
Thanks for your interest in contributing code to pymc_bart!

**If this is your first time contributing to a project on GitHub, please read through our step by step guide to contributing to pymc_bart**

### Local Development

0. Create a virtual environment (optional, but strongly recommended)

1. Install the library in editable mode

```bash
pip install -e .
```

### Feature Branch

1. From the fork of the pymc_bart repository, create a new branch for your feature.

```bash
git checkout -b feature_branch_name
```

2. Make your changes to the code base.

3.Add and commit your changes.

```bash
git add my_modified_file.py
git commit -m "Added a new feature"
```

4. Push your changes to your fork of the pymc_bart repository.

```bash
git push origin feature_branch_name
```

### Code Style

The repository has some code style checks in place. This will happen on every commit of a pull request. If you want to run the checks locally, you can do so by running the following command from the root of the repository:

1. Install pre-commit

```bash
pip install pre-commit
```

2. Set up pre-commit

```bash
pre-commit install
```

3. Run the complete pre-commit hook to check specific files:

```bash
pre-commit run --files pymc_bart/tree.py
```

or all files:

```bash
pre-commit run --all-files
```

**Once you commit something the pre-commit hook will run all the checks**!

In particular, if the commited changed have linting errors, the commit will try to fix them. If successful,you need to add the changes again (for example, `git add -u`) and commit again. If not successful, you need to fix the errors manually and commit again.

You can skip this (for example when is WIP) by adding a flag (`-n` means no-verify)

```bash
git commit -m"my message" -n
```

### Pre-Commit Components

One can, of course, install `ruff` in the Python environment to enable auto-format (for example in VS Code), but this is not strictly necessary. The specific versions of` ruff` and `mypy` must be only specified in [`.pre-commit-config.yaml`](.pre-commit-config.yaml). It should be the only source of truth! Hence, if you want to install them locally make sure you use the same versions (revisions `rev` in the config file) as in the config file.

#### Ruff

Once installed locally as

```
pip install ruff==<VERSION>
```

You can check the lint as

```bash
ruff . --no-fix
```

You can allow `ruff` to fix the code by using the flag:

```bash
ruff . --fix
```

#### MyPy

We use `mypy` to check the type annotations. Once installed locally as

```bash
pip install mypy==<VERSION>
```

You also need the `pandas-stubs` library with the version specified in the [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file.

```bash
pip install pandas-stubs==<VERSION>
```

Now, you can check the type annotations as

```bash
mypy pymc_bart/.
```

### Adding new features
If you are interested in adding a new feature to pymc_bart,
first submit an issue using the "Feature Request" label for the community
to discuss its place and implementation within pymc_bart.
