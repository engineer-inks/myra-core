#!/bin/bash

python setup.py -q develop

ENV=test py.test --cov=ink.core.forge.templates "${@:2}";
exitcode=$?
rm -f .coverage .coverage.*  # cleanup
exit $exitcode
