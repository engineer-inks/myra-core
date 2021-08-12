import os

ARCH_REPO = 'git@github.com:myra-ink/myra-core.git'
CORE_REPO = 'git@github.com:myra-ink/myra-core.git'
CORE_TEXT_REPO = 'git@github.com:myra-ink/myra-core.git'
CORE_IMAGE_REPO = 'git@github.com:myra-ink/myra-core.git'
CORE_SPEECH_REPO = 'git@github.com:myra-ink/myra-core.git'

DEPS = (
    ('myra_core_core', CORE_REPO, 'CORE_VERSION'),
    ('myra_core_text', CORE_TEXT_REPO, 'CORE_TEXT_VERSION'),
    ('myra_core_image', CORE_IMAGE_REPO, 'CORE_IMAGE_VERSION'),
    ('myra_core_speech', CORE_SPEECH_REPO, 'CORE_SPEECH_VERSION'),    
)

LOGGING_FORMAT = '%(message)s'

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')

ARTIFACTS = ('notebooks', 'jobs')
