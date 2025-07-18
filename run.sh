#!/bin/bash

pwd

python preprocess.py
python get_formal_context.py
python evaluate.py
python fca.py