import abc
import logging
import os
import shutil

import yaml

from . import consts
from .base import Runner, context, assert_is_dna_project, safe_name, as_path, C


class Generator(Runner, metaclass=abc.ABCMeta):
    def __init__(self,
                 config: C,
                 force: bool):
        super().__init__(config)
        self.name = self.config['NAME']
        self.target = self.config['_TARGET']
        self.arch_version = self.config['ARCH_VERSION']
        self.force = force

        self.simple_name = os.path.basename(self.name)


class Project(Generator):
    """Archetype Adapter

    Generate a new project using archetype as base.
    """
    def __init__(self, config: C, force: bool):
        super().__init__(config, force)
        self.simple_name = safe_name(self.simple_name)

    def create(self):
        if os.path.exists(self.target):
            if not self.force:
                raise ValueError(f'Project "{self.target}" already exists.')

            shutil.rmtree(self.target)

        logging.info(f'Creating project `{self.simple_name}`')

        (self
         .clone()
         .replace_with_actual_name())

        return self

    def clone(self):
        logging.info(f'Cloning archetype from `{consts.ARCH_REPO}`')
        os.system(f'git clone -q {consts.ARCH_REPO} {self.target} --depth 1')
        shutil.rmtree(os.path.join(self.target, '.git'))

        return self

    def replace_with_actual_name(self):
        logging.info(f'Replacing `archetype` occurrences by `{safe_name(self.simple_name)}`')

        roots = []

        for root, dirs, files in os.walk(self.target):
            target_root = self._rename_arch(root)

            if root != target_root:
                logging.debug(f'  mk {target_root}')
                os.makedirs(target_root, exist_ok=True)

            for f in files:
                source, target = os.path.join(
                    root, f), os.path.join(target_root, f)

                logging.debug(f'  mk {target}')

                with open(source) as s:
                    content = s.read()

                with open(target, 'w') as t:
                    t.write(self._rename_arch(content))

                if source != target:
                    logging.debug(f'  rm {source}')
                    os.chmod(target, os.stat(source).st_mode)
                    os.remove(source)

            roots.append((root, target_root))

        logging.debug(f'Deleting dangling archetype folders')
        for original_root, target_root in roots:
            if original_root != target_root and os.path.exists(original_root):
                logging.debug(f'  rm {original_root}')
                shutil.rmtree(original_root)

        return self

    def _rename_arch(self, content):
        return content.replace('archetype', self.simple_name)


class TemplateGenerator(Generator):
    DIRS: str
    SUFFIX: str = '.py'
    TEMPLATE_NAME: str = None

    @property
    def template_name(self):
        return self.TEMPLATE_NAME or self.__class__.__name__.lower()

    def create(self):
        assert_is_dna_project('.')

        dirs = self.DIRS.format(**self.config)
        path = as_path(self.name, dirs, self.SUFFIX)

        if os.path.exists(path) and not self.force:
            return logging.warning(f'{self.template_name} {self.name} already exists at `{path}`')

        logging.info(f'Creating job {self.name} at `{path}`')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        t = self.load_template()

        for key, value in self.default_context():
            t = t.replace(key, value)

        with open(path, 'w') as f:
            logging.debug(f' mk {path}')
            f.write(t)

    def load_template(self):
        name, ext = self.template_name, self.SUFFIX

        name = name.replace(ext, '')
        file_path = os.path.join(consts.TEMPLATES_DIR, f'{name}{ext}')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Cannot find template {name} at `{file_path}`.')

        with open(file_path) as f:
            return f.read()

    def default_context(self):
        return (('NAME_SIMPLE', self.simple_name),
                ('NAME_DOTTED', self.name.replace('/', '.')),
                ('ARCHETYPE_INITIAL', self.config['SERVICE'][:1].upper()),
                ('ARCHETYPE', self.config['SERVICE']),
                ('NAME', self.name))


class Notebook(TemplateGenerator):
    DIRS = 'notebooks'
    SUFFIX = '.ipynb'


class Job(TemplateGenerator):
    DIRS = 'jobs'


class JobProcessor(Job):
    ...


class JobFull(Job):
    ...


class Processor(TemplateGenerator):
    DIRS = 'ink/core/forge/{SERVICE}/processors'


class Test(TemplateGenerator):
    DIRS = 'tests/unit/{SERVICE}'
    SUFFIX = '_test.py'


GENERATORS = {
    'project': Project,
    'notebook': Notebook,
    'job': Job,
    'job-processor': JobProcessor,
    'job-full': JobFull,
    'processor': Processor,
    'test': Test,
}


def adapter(args, verbose: bool = True):
    global GENERATORS

    if verbose: logging.info(f'Using {args.env} environment.')

    c = {} if args.artifact == 'project' else context(args.env)

    c.update(ARTIFACT=args.artifact,
             NAME=args.name,
             ARCH_VERSION=args.arch_version,
             _TARGET=args.to or args.name,
             _CORE_REPO=consts.CORE_REPO,
             _ARCH_REPO=consts.ARCH_REPO)

    logging.debug(yaml.dump({'config': c}))
    return GENERATORS[args.artifact](c, force=args.force)
