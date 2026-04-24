python reaction_prediction/ranker/scripts/train_ranker_model.py \
  --train_h5 output/mc_train_fold0/ranker_training/train.hdf5 \
  --val_h5   output/mc_train_fold0/ranker_training/val.hdf5 \
  --out_dir  output/mc_train_fold0/models/ranker/ \
  --ranker_config model_configs/ranker_config.json