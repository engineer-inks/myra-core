import argparse
import logging
from ink.core import main

from . import generators, backend, consts

DESCRIPTION = 'Core projects management tool.'
ENV_PARAMS = {'nargs': '?', 'default': 'local', 'help': 'environment configuration used'}
VERBOSITY_CHOICES = [v.lower() for v in logging._nameToLevel]


def core_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('--env', default='local', help='environment configuration used')

    e = p.add_mutually_exclusive_group()
    e.add_argument('--verbosity', '-v', type=str, choices=VERBOSITY_CHOICES, default=None)
    e.add_argument('--quiet', '-q', action='store_true', default=False)

    s = p.add_subparsers()

    g = s.add_parser('create', help='create core artifacts')
    g.add_argument('artifact', help='artifact to be created', choices=generators.GENERATORS.keys())
    g.add_argument('name', help='name of the creating artifact')
    g.add_argument('-t', '--to', default=None, help='location of the artifact')
    g.add_argument('-f', '--force', action='store_true', help='overwrite when already exists')
    g.add_argument('--arch-version', default='master', help='archetype version used (tag or branch)')
    g.set_defaults(operation=lambda args, _: generators.adapter(args).create())

    u = s.add_parser('build', help='build a core project')
    u.add_argument('env', **ENV_PARAMS)
    u.set_defaults(operation=lambda args, _: backend.adapter(args.env).build())

    u = s.add_parser('clean', help='clean project build')
    u.add_argument('env', **ENV_PARAMS)
    u.set_defaults(operation=lambda args, _: backend.adapter(args.env).clean())

    u = s.add_parser('start', help='start docker or cluster')
    u.add_argument('env', **ENV_PARAMS)
    u.set_defaults(operation=lambda args, _: backend.adapter(args.env).start())

    u = s.add_parser('explore', help='explore a core environment with jupyter')
    u.add_argument('env', **ENV_PARAMS)
    u.set_defaults(operation=lambda args, _: backend.adapter(args.env).explore())

    u = s.add_parser('stop', help='stop docker or cluster')
    u.add_argument('env', **ENV_PARAMS)
    u.set_defaults(operation=lambda args, _: backend.adapter(args.env).stop())

    u = s.add_parser('run', help='run project job')
    u.add_argument('job', help='path to job')
    u.set_defaults(operation=lambda args, remaining: backend.adapter(args.env)
                   .run(args.job, remaining))

    u = s.add_parser('sync', help='sync local notebooks with the ones in a cluster')
    u.add_argument('artifact', help='artifact to be synced', choices=consts.ARTIFACTS)
    u.add_argument('source', help='env containing the source artifacts')
    u.add_argument('target', help='env in which the artifacts will be updated')
    u.set_defaults(operation=lambda args, _: backend.adapter(args.source)
                   .sync(args.artifact, args.target))

    u = s.add_parser('test', help='run tests')
    u.set_defaults(env='test',
                   operation=lambda args, remaining:
                       backend.adapter(args.env).test(remaining))

    p.set_defaults(operation=lambda args, _: p.print_help())

    return p


def main():
    args, unknown = core_parser().parse_known_args()

    level = (getattr(logging, args.verbosity.upper(), None) if args.verbosity else
             logging.ERROR if args.quiet else
             logging.WARNING)

    logging.basicConfig(format=consts.LOGGING_FORMAT, level=level)

    try:
        args.operation(args, unknown)

    except KeyboardInterrupt:
        logging.info('Interrupted')

    except RuntimeError as e:
        logging.error(e)
        exit(-1)


if __name__ == '__main__':
    main()
