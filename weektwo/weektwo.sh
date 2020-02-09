#!/bin/sh

export PYTHONPATH="/home/hep/dm2614/miniconda3/bin/python3"

source /home/hep/dm2614/miniconda3/etc/profile.d/conda.sh


conda activate ml
#conda install --file projects/mlatimperial/requirements.txt
python /home/hep/dm2614/projects/mlatimperial/weektwo.py 4
