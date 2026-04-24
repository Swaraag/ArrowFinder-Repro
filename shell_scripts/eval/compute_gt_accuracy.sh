python reaction_prediction/eval/compute_topk_accuracy.py \
  --targets data/mc_train_fold0/with_quotes/test.txt \
  --preds   output/mc_train_fold0/preds/gt_two_stage/preds_ascending.csv \
  --kmax 10