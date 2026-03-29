#!/bin/bash

# SINK
python reaction_prediction/atom/architectures/train_graph_transformer.py \
  --model_config_file_path model_configs/gt_config.json \
  --train_file_path output/mc_train_fold0/graph_data/sink/train.pt \
  --val_file_path output/mc_train_fold0/graph_data/sink/val.pt \
  --history_output_dir output/mc_train_fold0/models/graph_transformer/sink_plots \
  --model_output_file_path output/mc_train_fold0/models/graph_transformer/sink.pt

