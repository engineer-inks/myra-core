# Data science Functions for myra-core

[![CircleCI](https://circleci.com/gh/dextra/dna-core.svg?style=shield&circle-token=416684ff946282695aa8a56d1cf9cbae51fe7b96)](https://app.circleci.com/pipelines/github/myra-ink)
[![Documentation](https://img.shields.io/badge/docs-0.5.0-orange.svg?style=flat-square)](https://app.circleci.com/pipelines/github/myra-ink)
[![Python required version: 3.7](https://img.shields.io/badge/python-3.7-blue.svg?style=flat-square)](https://www.python.org/downloads/release/python-370)

---

A very lightweight (python) almost-dependency-free util for DnA.
Used to bootstrap projects and infrastructure.

### Contributing

This project is maintained by the ink team members during mild development cycles and contribution is always welcomed. Bugs can be reported and suggestions can be made on the issue of our github can be pushed as well.

## Installation

```shell
pip install git+https://github.com/myra-ink/myra-core.git
```

## Usage

```shell
import ink.core.templates as T

class Rawing(T.processors.Rawing):
    def call(self, x):
        x = self.hash_sensible_info(x, ('email', 'name'))
        x = self.exclude_sensible_info(x, 'review')

        return x

class Refining(T.processors.Refining):
    def call(self, x):
        ws = T.datasets.HIGH_FRICTION_WORDS
        x = self.infer_from_keywords(x, 'review', 'alarming', ws)
        x = x.withColumn('sentiment', x.score / 5)

        return x
```
