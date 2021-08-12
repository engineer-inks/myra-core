import os

from ink.core.forge.templates.configs import Config, Test

HOME_DIR = os.path.expanduser('~')
CONFIG_DIR = os.getenv('CONFIG_DIR', os.path.join(HOME_DIR, 'ink'))
ENV = os.getenv('ENV', 'local')

config = Config(env=ENV, config_dir=CONFIG_DIR)

print(f'Using {config.env} environment.')

__all__ = ['Config', 'Test', 'config']
