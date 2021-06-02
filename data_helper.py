# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
from constants import *
import csv
from pprint import pprint
import json
from utils import get_train_dev_test_split, get_date, load_text, loweralphanumeric, load_train_dev_test_json, \
    load_jsonl, get_strdate_fromtimestamp, load_jsonl, remap_dict, find_mode_num, listDeduplicate
import tldextract
import nltk
import spacy
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
import sys


def read_NELA_2018_label():
    # Unfinished
    label_dict = {}
    label_path = os.path.join(NELA_2018_PATH, 'labels.csv')
    with open(label_path, 'r') as fr:
        rd = csv.reader(fr, delimiter=',')
        for i, line in enumerate(rd):
            if i == 0:
                continue
            label_name = line[0]
            label_dict[label_name] = None

    return label_dict


def read_NELA_2019_label():
    # Currently only extracting the daggregated label
    label_dict = {}
    label_path = os.path.join(NELA_2019_PATH, 'labels_new.csv')

    with open(label_path, 'r') as fr:
        rd = csv.reader(fr, delimiter=',')
        for i, line in enumerate(rd):
            if i == 0:
                continue
            label_name = line[0]
            aggregated_label = line[1]
            label_dict[label_name] = aggregated_label

    return label_dict


def set_NELA_2019_dist_random_label(seed=42, num_class=2):
    """This funciton assign random labels to nela 2019 sites"""
    label_dict = {}
    label_path = os.path.join(NELA_2019_PATH, 'labels_new.csv')

    if num_class == 2:
        valid_labels = ['0', '2']
    elif num_class == 3:
        valid_labels = ['0', '1', '2']

    assert (num_class in [2, 3])
    np.random.seed(seed)

    random_label_list = []

    with open(label_path, 'r') as fr:
        rd = csv.reader(fr, delimiter=',')
        for i, line in enumerate(rd):
            if i == 0:
                continue
            label_name = line[0]
            real_label = line[1]
            if real_label not in valid_labels:
                continue
            else:
                random_label_list.append(real_label)
    random_label_list = np.random.permutation(random_label_list).tolist()

    with open(label_path, 'r') as fr:
        rd = csv.reader(fr, delimiter=',')

        for i, line in enumerate(rd):
            if i == 0:
                continue
            label_name = line[0]
            real_label = line[1]
            if real_label not in valid_labels:
                label_dict[label_name] = ''
            else:
                label_dict[label_name] = random_label_list.pop()

    return label_dict


def set_NELA_2019_site_label(num_class=2):
    label_dict = {}
    label_path = os.path.join(NELA_2019_PATH, 'labels_new.csv')

    if num_class == 2:
        valid_labels = ['0', '2']
    elif num_class == 3:
        valid_labels = ['0', '1', '2']

    num_valid_labels = 0

    with open(label_path, 'r') as fr:
        rd = csv.reader(fr, delimiter=',')
        for i, line in enumerate(rd):
            if i == 0:
                continue
            label_name = line[0]
            real_label = line[1]
            if real_label not in valid_labels:
                label_dict[label_name] = ''
            else:
                label_dict[label_name] = str(num_valid_labels)
                num_valid_labels += 1

    return label_dict


def read_NELA_2019():
    examples = []

    for subdir, dirs, files in os.walk(NELA_2019_ARTICLE_PATH):
        for filename in files:
            filepath = subdir + os.sep + filename
            if not filename.startswith('.'):
                with open(filepath, 'r') as fr:
                    raw_examples = json.load(fr)
                for example in raw_examples:
                    example = example  # LGTM
                    examples.append(example)

    return examples


def read_NELA_2018():
    # First get a source level label_dict from read_NELA_2019_label
    examples = []

    # read content
    for subdir, dirs, files in os.walk(NELA_2018_ARTICLE_PATH):
        for filename in files:
            filepath = subdir + os.sep + filename

            if not filename.startswith('.'):
                example = {}
                name_splits = filename.strip().split('--')
                source = name_splits[0]
                date = name_splits[1]
                title = '--'.join(name_splits[2:])

                example['content'] = load_text(filepath)
                example['title'] = title
                example['date'] = date
                example['source'] = source
                examples.append(example)

    return examples


def split_NELA_2019_by_month(full_data, label_dict, num_labels, split_name, domains=None, seed=42, balance=True):
    filtered_examples = []
    for example in full_data:
        if domains is not None:
            if example['source'] not in domains:
                continue

        dom = example['source']
        processed_dom = loweralphanumeric(dom)
        example['source'] = processed_dom
        if processed_dom not in label_dict.keys():
            continue

        dom_label = label_dict[processed_dom]
        if num_labels == 2:
            if dom_label in ['0', '2']:
                if dom_label == '2':
                    example['label'] = 0
                elif dom_label == '0':
                    example['label'] = 1
                filtered_examples.append(example)
            else:
                continue
        elif num_labels == 3:
            if dom_label in ['0', '1', '2']:
                example['label'] = 2 - int(dom_label)
                filtered_examples.append(example)
            else:
                continue

    month_data = {m: [] for m in range(1, 13)}
    for example in filtered_examples:
        date = example['date']
        month = datetime.strptime(date, '%Y-%m-%d').month
        month_data[month].append(example)

    np.random.seed(seed)

    if balance:
        for m in range(1, 13):
            month_data[m], _ = balance_dataset(month_data[m], 'label')

    data_stats = {'num_labels': num_labels, 'seed': seed, 'size': {m: len(month_data[m]) for m in range(1, 13)}}
    if domains is not None:
        data_stats['domains'] = domains

    split_path = os.path.join(NELA_2019_PATH, split_name)
    if not os.path.exists(split_path):
        os.mkdir(split_path)
    else:
        raise FileExistsError

    # Write detailed split info
    detail_split_path = os.path.join(split_path, 'split_detail.json')
    with open(detail_split_path, 'w') as fw:
        json.dump(data_stats, fw)

    for m in range(1, 13):
        file_path = os.path.join(split_path, 'month%d.json' % m)
        with open(file_path, 'w') as fw:
            for ex in month_data[m]:
                json.dump(ex, fw)
                fw.write('\n')


def split_NELA_2018(full_data, label_dict, num_labels=2, split='random', balance=False, seed=42, date1=None, date2=None,
                    prefix=None):
    # First run read_NELA_2018 to get full_data

    assert (split in {'random', 'domain', 'time'})
    assert (num_labels in [2, 3, -1])

    filtered_examples = []
    positive_source = set()
    mixed_source = set()
    negative_source = set()
    for ex in full_data:
        dom = ex['source']
        processed_dom = loweralphanumeric(dom)
        ex['source'] = processed_dom
        if processed_dom not in label_dict.keys():
            continue

        dom_label = label_dict[processed_dom]
        if num_labels == 2:
            if dom_label in ['0', '2']:
                if dom_label == '2':
                    ex['label'] = 0
                    negative_source.add(processed_dom)
                elif dom_label == '0':
                    ex['label'] = 1
                    positive_source.add(processed_dom)
                filtered_examples.append(ex)
            else:
                continue
        elif num_labels == 3:
            if dom_label in ['0', '1', '2']:
                ex['label'] = 2 - int(dom_label)
                filtered_examples.append(ex)
                if dom_label == '2':
                    positive_source.add(processed_dom)
                elif dom_label == '1':
                    mixed_source.add(processed_dom)
                elif dom_label == '0':
                    negative_source.add(processed_dom)
            else:
                continue
        elif num_labels == -1:
            """use site label"""
            if dom_label != '':
                ex['label'] = int(dom_label)
                filtered_examples.append(ex)

    # Show overall label distribution
    if num_labels == 2:
        print("NEG: {0}".format(len([x for x in filtered_examples if x['label'] == 0])))
        print("POS: {0}".format(len([x for x in filtered_examples if x['label'] == 1])))
    elif num_labels == 3:
        print("NEG: {0}".format(len([x for x in filtered_examples if x['label'] == 0])))
        print("MIX: {0}".format(len([x for x in filtered_examples if x['label'] == 1])))
        print("POS: {0}".format(len([x for x in filtered_examples if x['label'] == 2])))

    np.random.seed(seed)

    if split == 'random':
        # random split to 0.8/0.1/0.1
        split_dir = "{0}label_{1}_{2}".format(num_labels, split, seed)
        train_data, dev_data, test_data = get_train_dev_test_split(filtered_examples, 0.8, 0.1)
    elif split == 'domain':
        # split domains to 0.5/0.25/0.25 since #domain is relatively small
        split_dir = "{0}label_{1}_{2}".format(num_labels, split, seed)
        train_pos_dom, dev_pos_dom, test_pos_dom = get_train_dev_test_split(list(positive_source), 0.5, 0.25)
        train_neg_dom, dev_neg_dom, test_neg_dom = get_train_dev_test_split(list(negative_source), 0.5, 0.25)
        if num_labels == 3:
            train_mix_dom, dev_mix_dom, test_mix_dom = get_train_dev_test_split(list(mixed_source), 0.5, 0.25)
        else:
            train_mix_dom = dev_mix_dom = test_mix_dom = []
        train_dom = OrderedDict.fromkeys(train_pos_dom + train_neg_dom + train_mix_dom)
        dev_dom = OrderedDict.fromkeys(dev_pos_dom + dev_neg_dom + dev_mix_dom)
        test_dom = OrderedDict.fromkeys(test_pos_dom + test_neg_dom + test_mix_dom)
        train_data = [ex for ex in filtered_examples if ex['source'] in train_dom]
        dev_data = [ex for ex in filtered_examples if ex['source'] in dev_dom]
        test_data = [ex for ex in filtered_examples if ex['source'] in test_dom]
    elif split == 'time':
        if date1 is None and date2 is None:
            # date1 is the boundary btw train & dev, date2 is the boundary btw dev & test
            dates = [get_date(ex['date']) for ex in filtered_examples]
            sorted_dates = sorted(dates)
            # default 0.8/0.1/0.1 split
            date1 = sorted_dates[int(0.8 * len(sorted_dates))]
            date2 = sorted_dates[int(0.9 * len(sorted_dates))]

        split_dir = "{0}label_{1}_{2}_{3}".format(num_labels, split, date1, date2)

        train_data, dev_data, test_data = [], [], []
        for ex in filtered_examples:
            ex_date = get_date(ex['date'])
            if ex_date <= date1:
                train_data.append(ex)
            elif ex_date <= date2:
                dev_data.append(ex)
            else:
                test_data.append(ex)

    if balance:
        train_data, _ = balance_dataset(train_data, 'label')
        dev_data, _ = balance_dataset(dev_data, 'label')
        test_data, _ = balance_dataset(test_data, 'label')

    # Calculate Dataset Statistics
    data_stats = {'num_labels': num_labels, 'split': split, 'seed': seed}
    train_neg_count = len([x for x in train_data if x['label'] == 0])
    train_pos_count = len([x for x in train_data if x['label'] == 1])
    dev_neg_count = len([x for x in dev_data if x['label'] == 0])
    dev_pos_count = len([x for x in dev_data if x['label'] == 1])
    test_neg_count = len([x for x in test_data if x['label'] == 0])
    test_pos_count = len([x for x in test_data if x['label'] == 1])
    data_stats['label_dist'] = {'train': {'pos': train_pos_count, 'neg': train_neg_count},
                                'dev': {'pos': dev_pos_count, 'neg': dev_neg_count},
                                'test': {'pos': test_pos_count, 'neg': test_neg_count}}

    if split == 'domain':
        data_stats['dom_spl'] = {'train': {'pos': train_pos_dom, 'neg': train_neg_dom},
                                 'dev': {'pos': dev_pos_dom, 'neg': dev_neg_dom},
                                 'test': {'pos': test_pos_dom, 'neg': test_neg_dom}}
        if num_labels == 3:
            data_stats['dom_spl']['train']['mix'] = train_mix_dom
            data_stats['dom_spl']['dev']['mix'] = dev_mix_dom
            data_stats['dom_spl']['test']['mix'] = test_mix_dom

    if split == 'time':
        data_stats['date_spl'] = {'date1': str(date1), 'date2': str(date2)}

    if prefix is not None:
        split_dir = prefix + '_' + split_dir

    split_path = os.path.join(NELA_2018_PATH, split_dir)
    if not os.path.exists(split_path):
        os.mkdir(split_path)
    else:
        raise FileExistsError

    # Write detailed split info
    detail_split_path = os.path.join(split_path, 'split_detail.json')
    with open(detail_split_path, 'w') as fw:
        json.dump(data_stats, fw)

    # Save split file in three file
    train_file_path = os.path.join(split_path, 'train.jsonl')
    dev_file_path = os.path.join(split_path, 'dev.jsonl')
    test_file_path = os.path.join(split_path, 'test.jsonl')

    with open(train_file_path, 'w') as fw:
        for ex in train_data:
            json.dump(ex, fw)
            fw.write('\n')

    with open(dev_file_path, 'w') as fw:
        for ex in dev_data:
            json.dump(ex, fw)
            fw.write('\n')

    with open(test_file_path, 'w') as fw:
        for ex in test_data:
            json.dump(ex, fw)
            fw.write('\n')


def expand_2label_to_3label_NELA2018(full_data, label_dict, twolabel_path, split='random', seed=42, balance=False):
    # First run read_NELA_2018 to get full_data
    assert (split in {'random', 'domain', 'time'})
    mixed_examples = []
    mixed_source = OrderedDict()
    for ex in full_data:
        dom = ex['source']
        processed_dom = loweralphanumeric(dom)
        ex['source'] = processed_dom
        if processed_dom not in label_dict.keys():
            continue

        dom_label = label_dict[processed_dom]
        if dom_label in ['1']:
            ex['label'] = 2 - int(dom_label)
            if dom_label == '1':
                mixed_source[processed_dom] = None
                mixed_examples.append(ex)
        else:
            continue
    print("MIX: {0}".format(len([x for x in mixed_examples if x['label'] == 1])))

    # read 2label data
    full_data_dict = load_train_dev_test_json(twolabel_path)
    train_2label_data, dev_2label_data, test_2label_data = full_data_dict['train'], full_data_dict['dev'], \
                                                           full_data_dict['test']
    # convert label
    for i, ex in enumerate(train_2label_data):
        if ex['label'] == 1:
            train_2label_data[i]['label'] = 2
    for i, ex in enumerate(dev_2label_data):
        if ex['label'] == 1:
            dev_2label_data[i]['label'] = 2
    for i, ex in enumerate(test_2label_data):
        if ex['label'] == 1:
            test_2label_data[i]['label'] = 2
    split_detail_2label = json.load(open(os.path.join(twolabel_path, 'split_detail.json')))

    np.random.seed(seed)

    split_dir = twolabel_path.replace("2label", "2to3label")
    if split == 'random':
        # random split to 0.8/0.1/0.1
        train_data, dev_data, test_data = get_train_dev_test_split(mixed_examples, 0.8, 0.1)
    elif split == 'domain':
        # split domains to 0.5/0.25/0.25 since #domain is relatively small
        train_mix_dom, dev_mix_dom, test_mix_dom = get_train_dev_test_split(list(mixed_source), 0.5, 0.25)
        train_dom = listDeduplicate(train_mix_dom)
        dev_dom = listDeduplicate(dev_mix_dom)
        test_dom = listDeduplicate(test_mix_dom)
        train_data = [ex for ex in mixed_examples if ex['source'] in train_dom]
        dev_data = [ex for ex in mixed_examples if ex['source'] in dev_dom]
        test_data = [ex for ex in mixed_examples if ex['source'] in test_dom]
    elif split == 'time':
        date1 = get_date(split_detail_2label['date_spl']['date1'])
        date2 = get_date(split_detail_2label['date_spl']['date2'])

        train_data, dev_data, test_data = [], [], []
        for ex in mixed_examples:
            ex_date = get_date(ex['date'])
            if ex_date <= date1:
                train_data.append(ex)
            elif ex_date <= date2:
                dev_data.append(ex)
            else:
                test_data.append(ex)

    train_data = train_data + train_2label_data
    dev_data = dev_data + dev_2label_data
    test_data = test_data + test_2label_data

    if balance:
        train_data, _ = balance_dataset(train_data, 'label')
        dev_data, _ = balance_dataset(dev_data, 'label')
        test_data, _ = balance_dataset(test_data, 'label')

    # Calculate Dataset Statistics
    data_stats = {'num_labels': 3, 'split': split, 'seed': seed}
    train_neg_count = len([x for x in train_data if x['label'] == 0])
    train_pos_count = len([x for x in train_data if x['label'] == 1])
    dev_neg_count = len([x for x in dev_data if x['label'] == 0])
    dev_pos_count = len([x for x in dev_data if x['label'] == 1])
    test_neg_count = len([x for x in test_data if x['label'] == 0])
    test_pos_count = len([x for x in test_data if x['label'] == 1])
    data_stats['label_dist'] = {'train': {'pos': train_pos_count, 'neg': train_neg_count},
                                'dev': {'pos': dev_pos_count, 'neg': dev_neg_count},
                                'test': {'pos': test_pos_count, 'neg': test_neg_count}}

    if split == 'domain':
        data_stats['dom_spl'] = split_detail_2label['dom_spl']
        data_stats['dom_spl']['train']['mix'] = train_mix_dom
        data_stats['dom_spl']['dev']['mix'] = dev_mix_dom
        data_stats['dom_spl']['test']['mix'] = test_mix_dom

    if split == 'time':
        data_stats['date_spl'] = {'date1': str(date1), 'date2': str(date2)}

    split_path = os.path.join(NELA_2018_PATH, split_dir)
    if not os.path.exists(split_path):
        os.mkdir(split_path)
    else:
        raise FileExistsError

    # Write detailed split info
    detail_split_path = os.path.join(split_path, 'split_detail.json')
    with open(detail_split_path, 'w') as fw:
        json.dump(data_stats, fw)

    # Save split file in three file
    train_file_path = os.path.join(split_path, 'train.jsonl')
    dev_file_path = os.path.join(split_path, 'dev.jsonl')
    test_file_path = os.path.join(split_path, 'test.jsonl')

    with open(train_file_path, 'w') as fw:
        for ex in train_data:
            json.dump(ex, fw)
            fw.write('\n')

    with open(dev_file_path, 'w') as fw:
        for ex in dev_data:
            json.dump(ex, fw)
            fw.write('\n')

    with open(test_file_path, 'w') as fw:
        for ex in test_data:
            json.dump(ex, fw)
            fw.write('\n')


def read_fakenewsnet():
    examples = []
    for subdir, dir, files in os.walk(FAKENEWSNET_RAW_PATH):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith('.json') and not filename.startswith('.'):
                try:
                    example = json.load(open(filepath))
                except:
                    print(filepath)
                    example = json.load(open(filepath))
                    exit()

                if len(example['text'].strip()) == 0:
                    continue

                if 'gossipcop' in filepath:
                    example['type'] = 'gossipcop'
                    if '/fake/' in filepath:
                        example['label'] = 0
                    elif '/real/' in filepath:
                        example['label'] = 1
                elif 'politifact' in filepath:
                    example['type'] = 'politifact'
                    if '/fake/' in filepath:
                        example['label'] = 0
                    elif '/real/' in filepath:
                        example['label'] = 1

                # clean key names
                if example['publish_date'] is not None:
                    example['date'] = get_strdate_fromtimestamp(example['publish_date'])
                else:
                    example['date'] = None

                # special process for archive websites
                if '.archive.' in example['source']:
                    # format web.archive.org/web/2018111111/reallink
                    archive_link = example['url'].strip()
                    after_archive_link = archive_link[archive_link.find('.archive.'):]
                    old_link = '/'.join(after_archive_link.split('/')[3:])
                    example['source'] = tldextract.extract(old_link).domain
                else:
                    example['source'] = tldextract.extract(example['source']).domain
                del example['publish_date']
                del example['top_img']
                del example['images']
                del example['movies']

                examples.append(example)
    return examples


def split_fakenewsnet(full_data, split='random', balance=False, seed=42, date1=None, date2=None, select_type='both'):
    # first run read_fakenewsnet to get full_data

    assert (split in {'random', 'domain', 'time'})
    assert (select_type in {'both', 'gossipcop', 'politifact'})
    if split in ['random', 'domain']:
        filtered_examples = full_data
    elif split in ['time']:
        filtered_examples = [ex for ex in full_data if ex['date'] is not None]

    if select_type != 'both':
        if select_type == 'gossipcop':
            filtered_examples = [ex for ex in filtered_examples if ex['type'] == 'gossipcop']
        elif select_type == 'politifact':
            filtered_examples = [ex for ex in filtered_examples if ex['type'] == 'politifact']

    all_sources = listDeduplicate([ex['source'] for ex in filtered_examples])

    print("NEG: {0}".format(len([x for x in filtered_examples if x['label'] == 0])))
    print("POS: {0}".format(len([x for x in filtered_examples if x['label'] == 1])))

    np.random.seed(seed)

    if split == 'random':
        # random split to 0.8/0.1/0.1
        split_dir = "{0}_{1}".format(split, seed)
        train_data, dev_data, test_data = get_train_dev_test_split(filtered_examples, 0.8, 0.1)
    elif split == 'domain':
        # split domains to 0.5/0.25/0.25 since #domain is relatively small
        split_dir = "{0}_{1}".format(split, seed)
        train_dom, dev_dom, test_dom = get_train_dev_test_split(list(all_sources), 0.5, 0.25)
        train_data = [ex for ex in filtered_examples if ex['source'] in train_dom]
        dev_data = [ex for ex in filtered_examples if ex['source'] in dev_dom]
        test_data = [ex for ex in filtered_examples if ex['source'] in test_dom]
    elif split == 'time':
        if date1 is None and date2 is None:
            # date1 is the boundary btw train & dev, date2 is the boundary btw dev & test
            dates = [get_date(ex['date']) for ex in filtered_examples]
            sorted_dates = sorted(dates)
            # default 0.8/0.1/0.1 split
            date1 = sorted_dates[int(0.8 * len(sorted_dates))]
            date2 = sorted_dates[int(0.9 * len(sorted_dates))]

        split_dir = "{0}_{1}_{2}".format(split, date1, date2)

        train_data, dev_data, test_data = [], [], []
        for ex in filtered_examples:
            ex_date = get_date(ex['date'])
            if ex_date <= date1:
                train_data.append(ex)
            elif ex_date <= date2:
                dev_data.append(ex)
            else:
                test_data.append(ex)

    if select_type != 'both':
        split_dir = '{0}_{1}'.format(split_dir, select_type)

    if balance:
        train_data, _ = balance_dataset(train_data, 'label')
        dev_data, _ = balance_dataset(dev_data, 'label')
        test_data, _ = balance_dataset(test_data, 'label')

    # Calculate Dataset Statistics
    data_stats = {'split': split, 'seed': seed}
    train_neg_count = len([x for x in train_data if x['label'] == 0])
    train_pos_count = len([x for x in train_data if x['label'] == 1])
    dev_neg_count = len([x for x in dev_data if x['label'] == 0])
    dev_pos_count = len([x for x in dev_data if x['label'] == 1])
    test_neg_count = len([x for x in test_data if x['label'] == 0])
    test_pos_count = len([x for x in test_data if x['label'] == 1])
    data_stats['label_dist'] = {'train': {'pos': train_pos_count, 'neg': train_neg_count},
                                'dev': {'pos': dev_pos_count, 'neg': dev_neg_count},
                                'test': {'pos': test_pos_count, 'neg': test_neg_count}}

    if split == 'domain':
        data_stats['dom_spl'] = {'train': train_dom,
                                 'dev': dev_dom,
                                 'test': test_dom}

    if split == 'time':
        data_stats['date_spl'] = {'date1': str(date1), 'date2': str(date2)}

    split_path = os.path.join(FAKENEWSNET_PATH, split_dir)
    if not os.path.exists(split_path):
        os.mkdir(split_path)
    else:
        raise FileExistsError

    # Write detailed split info
    detail_split_path = os.path.join(split_path, 'split_detail.json')
    with open(detail_split_path, 'w') as fw:
        json.dump(data_stats, fw)

    # Save split file in three file
    train_file_path = os.path.join(split_path, 'train.jsonl')
    dev_file_path = os.path.join(split_path, 'dev.jsonl')
    test_file_path = os.path.join(split_path, 'test.jsonl')

    with open(train_file_path, 'w') as fw:
        for ex in train_data:
            json.dump(ex, fw)
            fw.write('\n')

    with open(dev_file_path, 'w') as fw:
        for ex in dev_data:
            json.dump(ex, fw)
            fw.write('\n')

    with open(test_file_path, 'w') as fw:
        for ex in test_data:
            json.dump(ex, fw)
            fw.write('\n')


def balance_dataset(dataset, label_key):
    label_examples = dict()
    for ex in dataset:
        label = ex[label_key]
        if label not in label_examples:
            label_examples[label] = [ex]
        else:
            label_examples[label].append(ex)

    min_label_count = min([len(exs) for exs in label_examples.values()])
    balanced_res = []
    for label, exs in label_examples.items():
        exs = np.random.permutation(np.array(exs))
        label_examples[label] = exs[:min_label_count].tolist()
        balanced_res += label_examples[label]

    return balanced_res, min_label_count


def get_source_level_majority(dataset_path):
    key_examples = load_jsonl(dataset_path)
    label_keys = {}
    for ex in key_examples:
        label_keys[ex['source']] = int(ex['label'])

    num_sources = len(label_keys.keys())

    major_label_num = np.bincount(list(label_keys.values())).max()
    major_acc = major_label_num / num_sources
    print(dataset_path)
    print(major_acc)
    return major_acc


def get_readable_data(data_path_or_data, filter_func, num_examples, out_path, hide_label=True,
                      content_keys=['title', 'content'], label_keys=['label']):
    # This function samples a list of examples from the full_data and prints to the out_path

    if type(data_path_or_data) is str:
        full_data = load_jsonl(data_path_or_data)
    else:
        full_data = data_path_or_data

    filtered_data = [ex for ex in full_data if filter_func(ex) is True]

    np.random.seed(42)

    rand_permu = np.random.permutation(np.array(filtered_data))

    sampled_examples = rand_permu[:num_examples]

    label_keys = listDeduplicate(label_keys)
    content_keys = listDeduplicate(content_keys)

    contents = []
    labels = []
    for i, ex in enumerate(sampled_examples):
        contents.append({k: ex[k] for k in content_keys})
        if hide_label:
            labels.append({k: ex[k] for k in label_keys})

    with open(out_path + '.raw.jsonl', 'w') as fw:
        for ex in contents:
            json.dump(ex, fw)
            fw.write('\n')

    with open(out_path, 'w') as fw:
        for i, ex in enumerate(contents):
            fw.write(str(i) + '\n')
            pprint(ex, fw)
            fw.write('\n\n')

    if hide_label:
        with open(out_path + '.label', 'w') as fw:
            for i, l in enumerate(labels):
                fw.write(str(i) + '\n')
                pprint(l, fw)
                fw.write('\n\n')


def get_earliest_date(path):
    data = load_train_dev_test_json(path)
    full_data = data['train'] + data['dev'] + data['test']
    dates = [get_date(ex['date']) for ex in full_data]
    earliest_date = sorted(dates)[0]

    return earliest_date


def get_subset_by_date(path, date):
    old_data = load_train_dev_test_json(path)
    late_train = [ex for ex in old_data['train'] if get_date(ex['date']) > date]
    late_dev = [ex for ex in old_data['dev'] if get_date(ex['date']) > date]
    late_test = [ex for ex in old_data['test'] if get_date(ex['date']) > date]

    new_data_path = os.path.normpath(path) + '_late'
    if not os.path.exists(new_data_path):
        os.mkdir(new_data_path)
    else:
        raise FileExistsError

    # Save split file in three file
    train_file_path = os.path.join(new_data_path, 'train.jsonl')
    dev_file_path = os.path.join(new_data_path, 'dev.jsonl')
    test_file_path = os.path.join(new_data_path, 'test.jsonl')

    with open(train_file_path, 'w') as fw:
        for ex in late_train:
            json.dump(ex, fw)
            fw.write('\n')

    with open(dev_file_path, 'w') as fw:
        for ex in late_dev:
            json.dump(ex, fw)
            fw.write('\n')

    with open(test_file_path, 'w') as fw:
        for ex in late_test:
            json.dump(ex, fw)
            fw.write('\n')


def get_dataset_from_dict(dataset_path, dict_path, new_path, label_num=2):
    assert (label_num == 2)

    dataset = load_train_dev_test_json(dataset_path)
    pool = load_jsonl(dict_path)

    filtered_pool = {}
    for ex in pool:
        if ex['label'] == 1:
            continue
        elif ex['label'] == 2:
            ex['label'] == 1

        key = ex['source'].strip() + ex['date'].strip() + ex['title'].strip()
        if key not in filtered_pool:
            filtered_pool[key] = []
        filtered_pool[key].append(ex)

    new_train = []
    for ex in dataset['train']:
        key = ex['source'].strip() + ex['date'].strip() + ex['title'].strip()
        new_train.extend(filtered_pool[key])

    new_dev = []
    for ex in dataset['dev']:
        key = ex['source'].strip() + ex['date'].strip() + ex['title'].strip()
        new_dev.extend(filtered_pool[key])

    new_test = []
    for ex in dataset['test']:
        key = ex['source'].strip() + ex['date'].strip() + ex['title'].strip()
        new_test.extend(filtered_pool[key])

    if not os.path.exists(new_path):
        os.mkdir(new_path)
    else:
        raise FileExistsError

    # Save split file in three file
    train_file_path = os.path.join(new_path, 'train.jsonl')
    dev_file_path = os.path.join(new_path, 'dev.jsonl')
    test_file_path = os.path.join(new_path, 'test.jsonl')

    with open(train_file_path, 'w') as fw:
        for ex in new_train:
            json.dump(ex, fw)
            fw.write('\n')

    with open(dev_file_path, 'w') as fw:
        for ex in new_dev:
            json.dump(ex, fw)
            fw.write('\n')

    with open(test_file_path, 'w') as fw:
        for ex in new_test:
            json.dump(ex, fw)
            fw.write('\n')


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    mode = sys.argv[2]

    if len(sys.argv) == 4:
        seed = sys.argv[3]
    else:
        seed = 42

    if dataset_name == 'nela':
        if mode == 'random':
            label_dict = read_NELA_2019_label()
            data = read_NELA_2018()
            split_NELA_2018(data, label_dict, 2, 'random', True, seed=seed)
        elif mode == 'site':
            label_dict = read_NELA_2019_label()
            data = read_NELA_2018()
            split_NELA_2018(data, label_dict, 2, 'domain', True, seed=seed)
        elif mode == 'time':
            label_dict = read_NELA_2019_label()
            data = read_NELA_2018()
            split_NELA_2018(data, label_dict, 2, 'time', True, seed=seed)
        elif mode == 'random_label':
            label_dict = set_NELA_2019_dist_random_label(seed=seed, num_class=2)
            data = read_NELA_2018()
            split_NELA_2018(data, label_dict, 2, 'random', True, seed=seed)
        else:
            raise NotImplementedError
    elif dataset_name == 'fnn':
        if mode == 'random':
            full_data = read_fakenewsnet()
            split_fakenewsnet(full_data, split='random', balance=True, select_type='both', seed=seed)
        elif mode == 'site':
            full_data = read_fakenewsnet()
            split_fakenewsnet(full_data, split='domain', balance=True, select_type='both', seed=seed)
        elif mode == 'time':
            full_data = read_fakenewsnet()
            split_fakenewsnet(full_data, split='time', balance=True, select_type='both', seed=seed)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
