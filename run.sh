#!/bin/bash

pwd

python preprocess.py
# Uncomment the following command if running closed-set reconstruction (fixed sets of candidate objects & attributes within a domain).
python get_formal_context.py
# Uncomment the following command if applying Gibbs sampling according to Definition 9.
# python get_formal_context.py \
#   --gibbs_only \
#   --gibbs_steps 500 \
#   --gibbs_burn_in 100 \
#   --gibbs_thinning 5 \
#   --gibbs_top_k 100 \
#   --gibbs_temperature 1.0 \
#   --gibbs_threshold 0.5
python evaluate.py
python fca.py