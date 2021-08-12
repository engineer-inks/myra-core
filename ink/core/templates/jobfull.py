import os
import logging
from argparse import ArgumentParser

import ink.core.templates.core as C
import ink.core.templates.archetype as ARCHETYPE_INITIAL


def parse_args():
    p = ArgumentParser('NAME_SIMPLE job')
    p.add_argument('--inputs', default=os.path.join(ARCHETYPE_INITIAL.config.lakes.transient, 'NAME.parquet'))
    p.add_argument('--outputs', default=os.path.join(ARCHETYPE_INITIAL.config.lakes.refined, 'NAME.parquet'))

    return p.parse_args()


def run(inputs, outputs):
    raw = os.path.join(ARCHETYPE_INITIAL.config.lakes.raw, 'NAME.parquet')
    tru = os.path.join(ARCHETYPE_INITIAL.config.lakes.trusted, 'NAME.parquet')

    (C.Job(
        ARCHETYPE_INITIAL.processors.NAME_DOTTED.Rawing(inputs, raw),
        ARCHETYPE_INITIAL.processors.NAME_DOTTED.Trusting(raw, tru),
        ARCHETYPE_INITIAL.processors.NAME_DOTTED.Refining(tru, outputs))
     .setup(ARCHETYPE_INITIAL.config)
     .perform()
     .describe()
     .teardown())


if __name__ == '__main__':
    logging.basicConfig(**ARCHETYPE_INITIAL.config.logging.default.asDict())
    args = parse_args()
    
    run(args.inputs, args.outputs)
