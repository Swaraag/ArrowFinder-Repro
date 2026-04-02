#!/bin/bash

# for sink train
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/train.txt \
  --output output/mc_train_fold0/graph_data/sink_unfiltered/train.pt \
  --atom_type sink \

# for sink val
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/val.txt \
  --output output/mc_train_fold0/graph_data/sink_unfiltered/val.pt \
  --atom_type sink \

# for source train
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/train.txt \
  --output output/mc_train_fold0/graph_data/source_unfiltered/train.pt \
  --atom_type source \

# for source val
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/val.txt \
  --output output/mc_train_fold0/graph_data/source_unfiltered/val.pt \
  --atom_type source \