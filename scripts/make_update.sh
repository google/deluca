#!/usr/bin/env bash

# Assuming clean directory
bumpversion patch
rm -rf dist
python setup.py sdist
git push
twine upload dist/*
