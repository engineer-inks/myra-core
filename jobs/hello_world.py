import logging
from argparse import ArgumentParser

import ink.core.forge.joins.archetype as A
import ink.core.forge.joins.core as C

arg_parser = ArgumentParser('Hello World Job')
arg_parser.add_argument('name')


def run(name):
    logging.info(f'Hello {name}!')
    logging.debug(f'Listing transient bucket `{A.config.lakes.transient}`:')
    logging.info(C.io.storage.listdir(A.config.lakes.transient))


if __name__ == '__main__':
    args = arg_parser.parse_args()
    logging.basicConfig(**A.config.logging.default.asDict())

    run(args.name)
