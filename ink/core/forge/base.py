import abc
import logging
import os
from typing import Dict, Optional

from . import consts

C = Dict[str, str]


def default_env_path(target: str, env: str) -> str:
    return os.path.join(target, 'config', env)


def default_env_file(target: str, env: str) -> str:
    return os.path.join(default_env_path(target, env), '.env')


def load_config(env_file: str) -> C:
    if not os.path.exists(env_file):
        raise RuntimeError(f'Cannot load env file {env_file}.\nMake sure '
                           f'your current directory is a valid project.')

    with open(env_file) as f:
        ls = f.readlines()

    ls = [l.strip() for l in ls]
    ls = [l for l in ls if l and not l.startswith('#')]
    parts = [l.split('=') for l in ls]
    config = {k: '='.join(v) for k, *v in parts}

    return config


def assert_is_dna_project(target):
    if not os.path.exists(os.path.join(target, 'myra')):
        raise RuntimeError('DnA project not found.\nMake sure your '
                           'current directory is a valid project.')


def context(env) -> C:
    """Retrieve the project's current context, based on an environment.

    """
    target = os.getcwd()

    env_path = default_env_path(target, env)
    env_file = default_env_file(target, env)

    c = load_config(env_file)
    c.update(_ENV_PATH=env_path,
             _ENV_FILE=env_file,
             _TARGET=target,
             _CORE_REPO=consts.CORE_REPO,
             _ARCH_REPO=consts.ARCH_REPO)

    return c


class Runner(metaclass=abc.ABCMeta):
    def __init__(self, config: C):
        self.config = config

    def execute(self,
                command: str,
                log_contextualized: bool = False,
                **context):
        command_c = self.contextualized(command, context)
        logging.info(f'Running `{command_c if log_contextualized else command}`')

        if command is NotImplemented:
            raise NotImplementedError(f'Command was not implemented for '
                                      f'`{type(self).__name__}` backend.')

        return os.system(command_c)

    def contextualized(self, command, context=None):
        c = {**self.config, **(context or {})}
        return command.format(**c)


# region Artifacts related tasks

def safe_name(name):
    return name.replace('-', '_')


def as_path(name: str,
            directory: str,
            suffix: Optional[str] = '.py'):
    name = safe_name(name)
    locations = (directory, f'./{directory}')

    path, file = os.path.split(name)
    return os.path.join(
        path if path.startswith(locations) else os.path.join(directory, path),
        file if not suffix or file.endswith(suffix) else file + suffix)

# endregion
