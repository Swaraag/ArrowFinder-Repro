python reaction_prediction/eval/generate_predictions.py \
  --input  data/mc_train_fold0/with_quotes/test.txt \
  --output output/mc_train_fold0/preds \
  --allid  output/mc_train_fold0/allid/train_val_combined_allids.json \
  --source_model output/mc_train_fold0/models/atom/source.h5 \
  --sink_model   output/mc_train_fold0/models/atom/sink.h5 \
  --ranker_model output/mc_train_fold0/models/ranker/polar_ranker_single_best.h5 \
  --max_orbs 128 \
  --top_k 10 \
  --threshold 0.18
