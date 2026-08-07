#!/bin/zsh
cd "${0:A:h}"
python3 scripts/build_poetry_archive.py "$@"
