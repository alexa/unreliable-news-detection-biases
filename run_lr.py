# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from bow_lr import *
import numpy as np
import argparse
from utils import load_train_dev_test_json, extract_key, extract_keys, remap_dict, load_jsonl
from pandas import DataFrame
import pickle
import os


def save_prediction(prediction, path):
    with open(path, 'w') as fw:
        for p in prediction:
            fw.write(str(p) + '\n')


def eval_on_dataset(model, dict_name_map, path, args):
    if "split_detail" in path:
        return
    print(path)
    eval_data = load_jsonl(path)
    if args.tittxt:
        eval_data = [remap_dict(ex, dict_name_map) for ex in eval_data]
        predicted = model.predict(DataFrame.from_dict(extract_keys(eval_data, ['title', 'text'])))
    else:
        predicted = model.predict(extract_key(eval_data, args.input_key))
    acc = np.mean(predicted == extract_key(eval_data, args.target_key))
    print(acc)


def run_lr(args):
    skl_data = load_train_dev_test_json(args.data_dir)

    eval_set = skl_data['dev'] if not args.use_test_set else skl_data['test']

    if args.tittxt:
        model = lr_plus_clf

        input_keys = args.input_key.strip().split(',')
        title_key, text_key = input_keys[0], input_keys[1]
        dict_name_map = {title_key:'title', text_key:'text'}
        mapped_train = [remap_dict(ex, dict_name_map) for ex in skl_data['train']]
        eval_set = [remap_dict(ex, dict_name_map) for ex in eval_set]

        model.fit(DataFrame.from_dict(extract_keys(mapped_train, ['title', 'text'])), extract_key(mapped_train, args.target_key))
    else:
        # model = NB_clf
        dict_name_map = None
        model = lr_clf

        model.fit(extract_key(skl_data['train'], args.input_key), extract_key(skl_data['train'], args.target_key))

    if args.tittxt:
        predicted = model.predict(DataFrame.from_dict(extract_keys(eval_set, ['title', 'text'])))
    else:
        predicted = model.predict(extract_key(eval_set, args.input_key))
    acc = np.mean(predicted == extract_key(eval_set, args.target_key))
    print(acc)

    if args.save_prediction is not None:
        save_prediction(predicted, args.save_prediction)

    if args.save_model is not None:
        with open(args.save_model, 'wb') as f:
            pickle.dump(model, f)

    if args.test_dir is not None:
        for subdir, dirs, files in os.walk(args.test_dir):
            for filename in files:
                filepath = subdir + os.sep + filename
                if (filepath.endswith('.jsonl') or filepath.endswith('.json')) and not filename.startswith('.'):
                    eval_on_dataset(model, dict_name_map, filepath, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Containing train/dev/test.jsonl"
    )

    parser.add_argument(
        "--input_key",
        default=None,
        type=str,
        required=True,
        help="The input key"
    )

    parser.add_argument(
        '--target_key',
        default=None,
        type=str,
        required=True,
        help="The target key"
    )

    parser.add_argument(
        '--use_test_set',
        action="store_true",
        help="if set use test set instead of dev set"
    )

    parser.add_argument(
        '--save_prediction',
        default=None,
        type=str,
        help="If not None, save the prediction to this location"
    )

    parser.add_argument(
        '--save_model',
        default=None,
        type=str,
        help="If not None, save the model to this location"
    )

    parser.add_argument(
        '--tittxt',
        action="store_true",
        help="if set, use both title and text, and the input_key argument should in this format: [TITLE_KEY],[TEXT_KEY]"
    )

    parser.add_argument(
        '--test_dir',
        default=None,
        type=str,
        help = "If not None, test on all jsonl files in that dir"
    )

    args = parser.parse_args()

    run_lr(args)