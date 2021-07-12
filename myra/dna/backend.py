import abc
import logging
import os

import yaml

from . import consts
from .base import Runner, assert_is_dna_project, context, as_path


class Backend(Runner, metaclass=abc.ABCMeta):
    """Contains basic interaction logic with a environment backend.

    This is a base-class. Look at its sub-classes for implementation details.
    """

    artifacts = {a: NotImplemented for a in consts.ARTIFACTS}

    transport = {
        'local': NotImplemented,
        'gcloud': NotImplemented
    }

    class Command:
        SYNC: str = '{TRANSPORT} {ARTIFACT_A} {ARTIFACT_B}'
        BUILD: str = NotImplemented
        CLEAN: str = NotImplemented
        START: str = NotImplemented
        EXPLORE: str = NotImplemented
        STOP: str = NotImplemented
        RUN: str = NotImplemented
        TEST: str = NotImplemented

    def build(self):
        assert_is_dna_project(self.config['_TARGET'])
        self.build_dependencies()
        return self.execute(self.Command.BUILD)

    def clean(self):
        return self.execute(self.Command.CLEAN)

    def start(self):
        return self.execute(self.Command.START)

    def explore(self):
        return self.execute(self.Command.EXPLORE)

    def stop(self):
        return self.execute(self.Command.STOP)

    def run(self, job, job_args):
        assert_is_dna_project(self.config['_TARGET'])
        return self.execute(self.Command.RUN,
                            JOB=as_path(job, 'jobs'),
                            JOB_ARGS=' '.join(job_args))

    def test(self, job_args):
        assert_is_dna_project(self.config['_TARGET'])
        return self.execute(self.Command.TEST, JOB_ARGS=' '.join(job_args))

    def sync(self, artifact, target):
        assert_is_dna_project(self.config['_TARGET'])
        t = adapter(target, verbose=False)

        return self.execute(self.Command.SYNC,
                            TRANSPORT=self.transport[t.config['BACKEND']],
                            ARTIFACT_A=self.contextualized(self.artifacts[artifact]),
                            ARTIFACT_B=t.contextualized(t.artifacts[artifact]),
                            log_contextualized=True)

    def build_dependencies(self):
        for dep in consts.DEPS:
            name, repo, v_field = dep

            if v_field in self.config:
                version = self.config[v_field]

                wheel = self.contextualized('{_TARGET}/lib/wheels/%s-%s-py3-none-any.whl' % (name, version))
                source = self.contextualized('{_TARGET}/lib/wheels/%s_%s' % (name, version))

                if not os.path.exists(wheel):
                    self.clone_and_build_py_dep(
                        repo=repo,
                        branch=version,
                        source_dst=source)

    def clone_and_build_py_dep(self, repo, branch, source_dst):
        self.execute(
            f"""
    rm -rf {source_dst}
    mkdir -p {source_dst}
    git -c advice.detachedHead=false clone -b {branch} {repo} -q --depth 1 {source_dst}
    cd {source_dst}
    python setup.py -q sdist bdist_wheel --dist-dir ../
    cd ../../../..
    rm -rf {source_dst}""")


class Local(Backend):
    artifacts = {
        'notebooks': 'notebooks',
        'jobs': 'jobs'
    }

    transport = {
        'local': 'rsync -ru',
        'gcloud': 'gsutil -mq rsync -ru'
    }

    class Command(Backend.Command):
        BUILD = 'docker-compose --env-file {_ENV_FILE} build'
        CLEAN = 'rm -rf {_TARGET}/lib/wheels'
        START = 'docker-compose --env-file {_ENV_FILE} up -d'
        STOP = 'docker-compose --env-file {_ENV_FILE} down'
        RUN = 'docker-compose --env-file {_ENV_FILE} exec {SERVICE} python {JOB} {JOB_ARGS}'
        TEST = 'docker-compose --env-file {_ENV_FILE} run --rm {SERVICE} tests {JOB_ARGS}'
        EXPLORE = 'firefox http://localhost:{JUPYTER_PORT}'


class GCloud(Backend):
    artifacts = {
        'notebooks': 'gs://{BUCKET}/notebooks/jupyter',
        'jobs': 'gs://{BUCKET}/jobs',
    }

    transport = {
        'local': 'gsutil -mq rsync -ru',
        'gcloud': 'gsutil -mq rsync -ru'
    }

    class Command(Backend.Command):
        START = 'gcloud --project {PROJECT} dataproc clusters import {CLUSTER_NAME} --source {_ENV_PATH}/{CLUSTER_DEF} --region {REGION}'
        STOP = 'gcloud --project {PROJECT} dataproc clusters delete {CLUSTER_NAME} -q --region {REGION}'
        RUN = 'gcloud --project {PROJECT} dataproc jobs submit pyspark {JOB} --cluster {CLUSTER_NAME} --region {REGION} -- {JOB_ARGS}'
        EXPLORE = 'firefox $(gcloud --project {PROJECT} dataproc clusters describe {CLUSTER_NAME} --region {REGION} | grep -Pom1 "Jupyter: \K.+")'
        CLEAN = 'gsutil -mq rm -rf {_TARGET}/lib/wheels'
        BUILD = """
  python setup.py -q sdist bdist_wheel --dist-dir {_TARGET}/lib/wheels/
  gsutil mb -p {PROJECT} -l {REGION} gs://{BUCKET}
  gsutil -mq rsync -rd config/{ENV} gs://{BUCKET}/config/
  gsutil -mq cp -r lib/* gs://{BUCKET}/config/
  rm -rf build dist
  """


BACKENDS = {
    'local': Local,
    'gcloud': GCloud
}


def adapter(env, verbose: bool = True) -> Backend:
    """Retrieve the appropriate adapter for an environment.

    """
    global BACKENDS

    if verbose: logging.info(f'Using {env} environment.')

    c = context(env)
    k = c.get('BACKEND', 'local')

    if verbose: logging.debug(yaml.dump({'config': c}))

    try:
        _cls = BACKENDS[k] if isinstance(k, str) else k
        return _cls(c)

    except KeyError:
        raise RuntimeError(f'Sorry. We are not sure how to build an environment '
                           f'within `{k}` backend.\nMake sure a valid value for BACKEND '
                           f'{list(BACKENDS.keys())} is set at "config/{env}/.env"')
