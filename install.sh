#!/bin/bash

conda install -c conda-forge git-lfs
git lfs fetch
git lfs pull
pip install -r requirements.txt
