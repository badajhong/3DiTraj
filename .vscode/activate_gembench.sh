#!/bin/bash

# Load bash settings for colors and aliases
source ~/.bashrc

# Load Conda functions if not already available
if ! type "conda" > /dev/null 2>&1; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

# Activate only if not already active
if [[ $CONDA_DEFAULT_ENV != "gembench" ]]; then
    conda activate gembench
fi