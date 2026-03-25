#!/bin/bash

python reaction_prediction/atom/scripts/make_training_sink.py \
    --input data/mc_train_fold0/reformatted/train.txt \
    --output output/mc_train_fold0/atom_training/sink/train.hdf5 \
    --allid output/mc_train_fold0/allid/train_val_combined_allids.json

python reaction_prediction/atom/scripts/make_training_sink.py \
    --input data/mc_train_fold0/reformatted/val.txt \
    --output output/mc_train_fold0/atom_training/sink/val.hdf5 \
    --allid output/mc_train_fold0/allid/train_val_combined_allids.json