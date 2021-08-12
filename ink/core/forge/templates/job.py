import logging
from argparse import ArgumentParser

import ink.core.templates.archetype as ARCHETYPE_INITIAL


def parse_args():
    p = ArgumentParser('NAME_SIMPLE job')
    p.add_argument('name')
    
    return p.parse_args()


def run(name):
    logging.info(f'Hello {name}!')
    logging.info(f'Lakes: {ARCHETYPE_INITIAL.config.lakes}')


if __name__ == '__main__':
    logging.basicConfig(**ARCHETYPE_INITIAL.config.logging.default.asDict())
    args = parse_args()

    run(args.name)
