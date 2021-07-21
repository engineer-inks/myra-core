import os

CORE_REPO = 'git@github.com:myra-ink/myra-core.git'
ARCH_REPO = 'git@github.com:myra-ink/myra-core.git'

DEPS = (
    ('myra_core', CORE_REPO, 'CORE_VERSION'),
)

LOGGING_FORMAT = '%(message)s'

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')

ARTIFACTS = ('notebooks', 'jobs')
