# Data science Functions for myra-core

[![CircleCI](https://circleci.com/gh/dextra/dna-core.svg?style=shield&circle-token=416684ff946282695aa8a56d1cf9cbae51fe7b96)](https://app.circleci.com/pipelines/github/myra-ink)
[![Documentation](https://img.shields.io/badge/docs-0.4.5-orange.svg?style=flat-square)](https://github.com/myra-ink/myra-core)
[![Python required version: 3.7](https://img.shields.io/badge/python-3.7-blue.svg?style=flat-square)](https://www.python.org/downloads/release/python-370)

---

A very lightweight (python) almost-dependency-free util for DnA.
Used to bootstrap projects and infrastructure.


### Contributing

This project is maintained by the ink team members during mild development cycles and contribution is always welcomed. Bugs can be reported and suggestions can be made on the issue of our github can be pushed as well.

## Installation

```shell
pip install git+ssh://git@github.com:myra-ink/myra-core.git
```

## Usage

```shell
dna create project PROJECT
cd PROJECT
dna build
dna start  # A local jupyter server is now
           # available at localhost:8086
```
