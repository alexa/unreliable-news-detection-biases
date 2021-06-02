# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import json
from utils import load_jsonl, find_mode_num
import numpy as np
from collections import Counter
from pprint import pprint

TITLE_CORRECT_FILE = 'correct.title'
TITLE_WRONG_FILE = 'wrong.title'


def run_evaluation(args):
    # create local domain label
    label_keys = {}
    label_counts = Counter()
    gold_labels = []
    idx2source = []
    idx2title = []
    correct_titles = []
    wrong_titles = []
    key_examples = load_jsonl(args.key_file)
    for ex in key_examples:
        label_keys[ex['source']] = int(ex['label'])
        idx2source.append(ex['source'])
        idx2title.append(ex['title'])
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
                    if pred == gold_labels[i]:
                        correct_titles.append(idx2title[i])
                    else:
                        wrong_titles.append(idx2title[i])

                    source_norm_acc += int(pred==gold_labels[i])/label_counts[idx2source[i]]
                    acc = int(pred == gold_labels[i])
                    ind_source_accs[idx2source[i]].append(acc)
            else:
                preds = eval(lines[0].strip())
                for i, pred in enumerate(preds):
                    source_preds[idx2source[i]].append(pred)
                    if pred == gold_labels[i]:
                        correct_titles.append(idx2title[i])
                    else:
                        wrong_titles.append(idx2title[i])
                    source_norm_acc += int(pred==gold_labels[i])/label_counts[idx2source[i]]
                    acc = int(pred == gold_labels[i])
                    ind_source_accs[idx2source[i]].append(acc)
    elif args.pred_type == 'full':
        pred_dicts = load_jsonl(args.pred_file)
        for i, pd in enumerate(pred_dicts):
            # logits = pd['logits']
            # pred = int(np.argmax(logits))
            pred = int(pd['label'])
            source_preds[idx2source[i]].append(pred)
            if pred == gold_labels[i]:
                correct_titles.append(idx2title[i])
            else:
                wrong_titles.append(idx2title[i])
            source_norm_acc += int(pred == gold_labels[i]) / label_counts[idx2source[i]]
            acc = int(pred==gold_labels[i])
            ind_source_accs[idx2source[i]].append(acc)


    with open(TITLE_CORRECT_FILE, 'a') as fw:
        for t in correct_titles:
            fw.write(t+'\n')


    with open(TITLE_WRONG_FILE, 'a') as fw:
        for t in wrong_titles:
            fw.write(t+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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


    args = parser.parse_args()
    assert(args.pred_type in ["clean", "full"])

    run_evaluation(args)