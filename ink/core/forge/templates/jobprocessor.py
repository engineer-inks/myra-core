import os
import logging
from argparse import ArgumentParser

import ink.core.forge.joins.archetype as ARCHETYPE_INITIAL


def parse_args():
    p = ArgumentParser('NAME_SIMPLE job')
    p.add_argument('--inputs', default=os.path.join(ARCHETYPE_INITIAL.config.lakes.transient, 'NAME.parquet'))
    p.add_argument('--outputs', default=os.path.join(ARCHETYPE_INITIAL.config.lakes.raw, 'NAME.parquet'))

    return p.parse_args()


def run(inputs, outputs):
    (ARCHETYPE_INITIAL.processors.NAME_SIMPLE(inputs=inputs, outputs=outputs)
     .setup(ARCHETYPE_INITIAL.config)
     .perform()
     .describe()
     .save()
     .teardown())



if __name__ == '__main__':
    logging.basicConfig(**ARCHETYPE_INITIAL.config.logging.default.asDict())
    args = parse_args()

    run(args.inputs, args.outputs)
