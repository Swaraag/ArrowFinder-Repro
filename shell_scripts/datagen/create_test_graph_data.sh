#!/bin/bash

# for sink test
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/test.txt \
  --output output/mc_train_fold0/graph_data/sink/test.pt \
  --atom_type sink \
  --is_test True \

# for source test
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/test.txt \
  --output output/mc_train_fold0/graph_data/source/test.pt \
  --atom_type source \
  --is_test True \