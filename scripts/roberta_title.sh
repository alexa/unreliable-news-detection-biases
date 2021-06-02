#!/usr/bin/env bash

export DATA_DIR=./data/fakenews/2label_domain_42/
export MODEL_TYPE=roberta
export MODEL_PATH=roberta-base
export OUT_DIR=./output/roberta_nela_2018_2label_title_domain42_seed42

python run_bert.py --data_dir $DATA_DIR \
                   --model_type $MODEL_TYPE \
                   --model_name_or_path $MODEL_PATH \
                   --input_key title \
                   --target_key label \
                   --task_name news \
                   --do_train \
                   --do_eval \
                   --seed 42 \
                   --max_seq_length 128 \
                   --learning_rate 5e-5 \
                   --evaluate_during_training \
                   --logging_steps 1000 \
                   --save_steps 1000 \
                   --num_train_epochs 3.0 \
                   --per_gpu_train_batch_size 32 \
                   --per_gpu_eval_batch_size 32 \
                   --output_dir $OUT_DIR




#eval commands below
#python run_bert.py --data_dir $DATA_DIR \
#                   --model_type $MODEL_TYPE \
#                   --model_name_or_path $MODEL_PATH \
#                   --input_key title \
#                   --target_key label \
#                   --task_name news3 \
#                   --do_eval \
#                   --seed 42 \
#                   --max_seq_length 128 \
#                   --learning_rate 5e-5 \
#                   --evaluate_during_training \
#                   --logging_steps 1000 \
#                   --save_steps 1000 \
#                   --num_train_epochs 3.0 \
#                   --per_gpu_train_batch_size 32 \
#                   --per_gpu_eval_batch_size 32 \
#                   --output_dir $OUT_DIR
