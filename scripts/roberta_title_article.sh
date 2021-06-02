#!/usr/bin/env bash

export TRAIN_DATA_PATH=./data/nela/2018/2label_domain_42/train.jsonl
export DEV_DATA_PATH=./data/nela/2018/2label_domain_42/dev.jsonl
export TEST_DATA_PATH=./data/nela/2018/2label_domain_42/test.jsonl

allennlp train scripts/allen_config/roberta_512article_tittxt.jsonnet -s output/rob_nela_2018_2label_tittxt_domain_42 --include-package my_allen_lib

# eval command
# allennlp predict utput/rob_nela_2018_2label_tittxt_domain_42 $DEV_DATA_PATH --output-file utput/rob_nela_2018_2label_tittxt_domain_42/source_dev_preds.txt --batch-size 8 --silent --cuda-device 0 --use-dataset-reader --include-package my_allen_lib





