#!/usr/bin/env bash

# Model will save the predictions for the validation set, set the --use_test_set argument if you want to get the prediction on the test set
python run_lr.py --data_dir ./data/nela/2018/2label_domain_42 --input_key title --target_key label --save_prediction output/lr_preds/2label_domain_42.pred

