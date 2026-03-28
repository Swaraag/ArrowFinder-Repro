#!/bin/bash

# for sink train
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/train.txt \
  --output output/mc_train_fold0/graph_data/sink/train.pt \
  --atomtype sink \

# for sink val
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/val.txt \
  --output output/mc_train_fold0/graph_data/sink/val.pt \
  --atomtype sink \

# for source train
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/train.txt \
  --output output/mc_train_fold0/graph_data/source/train.pt \
  --atomtype source \

# for source val
python reaction_prediction/atom/architectures/make_atom_graph_data.py \
  --input data/mc_train_fold0/reformatted/val.txt \
  --output output/mc_train_fold0/graph_data/source/val.pt \
  --atomtype source \