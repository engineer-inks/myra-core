# Data science Functions for myra-core




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

core create project PROJECT
cd PROJECT
core build
core start  # A local jupyter server is now

```
```

```
```

```
```

available at localhost:8086

### Import Lib core

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
