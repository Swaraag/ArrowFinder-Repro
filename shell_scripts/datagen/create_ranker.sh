python reaction_prediction/ranker/scripts/make_training_ranker.py \
  --input data/mc_train_fold0/reformatted/val.txt \
  --output output/mc_train_fold0/ranker_training/val.hdf5 \
  --allid output/mc_train_fold0/allid/train_val_combined_allids.json

python reaction_prediction/ranker/scripts/make_training_ranker.py \
  --input data/mc_train_fold0/reformatted/train.txt \
  --output output/mc_train_fold0/ranker_training/train.hdf5 \
  --allid output/mc_train_fold0/allid/train_val_combined_allids.json