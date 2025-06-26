#!/bin/bash

cd Reimplement/code/
python preprocess.py
python get_formal_context.py
python fca.py

cd ../../