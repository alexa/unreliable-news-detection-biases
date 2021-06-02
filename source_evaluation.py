# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import json
from utils import load_jsonl, find_mode_num
import numpy as np
from collections import Counter
from pprint import pprint


def run_evaluation(args):
    # create local domain label
    label_keys = {}
    label_counts = Counter()
    gold_labels = []
    idx2source = []
    key_examples = load_jsonl(args.key_file)
    for ex in key_examples:
        label_keys[ex['source']] = int(ex['label'])
        idx2source.append(ex['source'])
        label_counts[ex['source']] += 1
        gold_labels.append(ex['label'])

    source_norm_acc = 0

    source_preds = {k:[] for k in label_keys.keys()}
    ind_source_accs = {k:[] for k in label_keys.keys()}
    if args.pred_type == 'clean':
        with open(args.pred_file, 'r') as fr:
            lines = fr.readlines()
            if len(lines) > 1:
                for i, line in enumerate(lines):
                    pred = int(line.strip())
                    source_preds[idx2source[i]].append(pred)
                    source_norm_acc += int(pred==gold_labels[i])/label_counts[idx2source[i]]
                    acc = int(pred == gold_labels[i])
                    ind_source_accs[idx2source[i]].append(acc)
            else:
                preds = eval(lines[0].strip())
                for i, pred in enumerate(preds):
                    source_preds[idx2source[i]].append(pred)
                    source_norm_acc += int(pred==gold_labels[i])/label_counts[idx2source[i]]
                    acc = int(pred == gold_labels[i])
                    ind_source_accs[idx2source[i]].append(acc)
    elif args.pred_type == 'full':
        pred_dicts = load_jsonl(args.pred_file)
        for i, pd in enumerate(pred_dicts):
            pred = int(pd['label'])
            source_preds[idx2source[i]].append(pred)
            source_norm_acc += int(pred == gold_labels[i]) / label_counts[idx2source[i]]
            acc = int(pred==gold_labels[i])
            ind_source_accs[idx2source[i]].append(acc)


    if args.print_detail:
        for s in label_keys.keys():
            source_size = label_counts[s]
            source_accuracy =  np.mean(ind_source_accs[s])
            print(f"Source {s}: SIZE {source_size}, Acc {source_accuracy}")


    source_accs_with_size = []
    source_accs = []
    for s in label_keys.keys():
        pred_mode = find_mode_num(source_preds[s])
        source_accs.append(int(pred_mode == label_keys[s]))
        source_accs_with_size.append((int(pred_mode == label_keys[s]), label_counts[s]))


    print(f"SOURCE LEVEL ACC: {np.mean(source_accs)}")


    if args.result_save_path is not None:
        save_dict = {}
        save_dict['source_level_acc'] = np.mean(source_accs)
        save_dict['source_norm_acc'] = source_norm_acc/len(label_keys.keys())
        if args.print_detail:
            save_dict['source_results'] = {s:{'size':label_counts[s], 'acc': np.mean(ind_source_accs[s])} for s in label_keys.keys()}
        with open(args.result_save_path, 'w') as fw:
            json.dump(save_dict, fw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating source level metrics from example level predictions, assume the prediction file is in the same order")

    parser.add_argument(
        '--pred_file',
        default=None,
        type=str,
        required=True,
        help="The prediction file"
    )

    parser.add_argument(
        '--key_file',
        default=None,
        type=str,
        required=True,
        help='The key file in jsonl format'
    )

    parser.add_argument(
        '--pred_type',
        default='clean',
        type=str,
        required=False,
        help="The content of the prediction file, select from [clean and full]"
    )

    parser.add_argument(
        '--print_detail',
        action="store_true",
        help="if set, print detailed results on all diff sources"
    )

    parser.add_argument(
        '--result_save_path',
        default=None,
        type=str,
        required=False,
        help= 'If not None, then save json predictions in the given path'
    )

    args = parser.parse_args()
    assert(args.pred_type in ["clean", "full"])

    run_evaluation(args)