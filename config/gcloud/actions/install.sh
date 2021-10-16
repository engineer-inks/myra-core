#!/bin/bash

ROLE="$(/usr/share/google/get_metadata_value attributes/dataproc-role)"
BUCKET="$(/usr/share/google/get_metadata_value attributes/dataproc-bucket)"
USE_JUPYTER="$(/usr/share/google/get_metadata_value attributes/jupyter_server || true)"

CONFIG_DIR="/root/dna"

setup_core() {
  echo "Setting up project"

  mkdir -p "$CONFIG_DIR"
  gsutil -mq rsync -rd gs://$BUCKET/config "$CONFIG_DIR"

  pip install $CONFIG_DIR/wheels/*

  source $CONFIG_DIR/.env
  echo "export CONFIG_DIR=$CONFIG_DIR" | tee -a /etc/bash.bashrc
}

setup_jupyter() {
  echo "Setting up jupyter"
  USE_JUPYTER=${USE_JUPYTER:-true}

  if $USE_JUPYTER && [[ $ROLE == 'Master' ]]; then
    if [ -n "$JUPYTER_THEME" ]; then
      jt -t $JUPYTER_THEME
    fi

    echo "c.MappingKernelManager.default_kernel_name = 'pyspark'" >>~/.jupyter/jupyter_notebook_config.py
    echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >> ~/.jupyter/jupyter_notebook_config.py
  fi
}

setup_core
setup_jupyter
echo "Done"
