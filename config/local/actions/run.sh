#!/bin/bash

python setup.py -q develop
/bin/bash

if [ -n "$JUPYTER_THEME" ]; then
  jt -t $JUPYTER_THEME
fi

jupyter notebook --ip=0.0.0.0 \
                 --port="$JUPYTER_PORT" --allow-root \
                 --NotebookApp.notebook_dir='./notebooks' \
                 --NotebookApp.token='' \
                 --NotebookApp.password=''
