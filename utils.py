# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import json
import time
from datetime import date
from datetime import datetime
import re
import sys
from collections import OrderedDict


def get_train_dev_test_split(data, train_portion, dev_portion):
    assert(train_portion + dev_portion <= 1.0)
    test_portion = 1.0 - train_portion - dev_portion

    rand_spl = np.random.permutation(len(data))
    train_idxs = rand_spl[:int(train_portion * len(data))]
    dev_idxs = rand_spl[int(train_portion * len(data)): int((train_portion + dev_portion)*len(data))]
    test_idxs = rand_spl[int((train_portion + dev_portion)*len(data)):]

    data = np.array(data)

    train_data = data[train_idxs].tolist()
    dev_data = data[dev_idxs].tolist()
    test_data = data[test_idxs].tolist()

    return train_data, dev_data, test_data


def load_jsonl(path):
    res = []
    with open(path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res.append(json.loads(line.strip()))
    return res


def load_text(path):
    with open(path, 'r') as fr:
        content = fr.read()
    return content


def load_train_dev_test_json(path):
    full_data_dict = {'train': None, 'dev': None, 'test': None}

    train_path = os.path.join(path, 'train.jsonl')
    dev_path = os.path.join(path, 'dev.jsonl')
    test_path = os.path.join(path, 'test.jsonl')

    full_data_dict['train'] = load_jsonl(train_path)
    full_data_dict['dev'] = load_jsonl(dev_path)
    full_data_dict['test'] = load_jsonl(test_path)

    return full_data_dict


def extract_key(d, k):
    return [x[k] for x in d]


def extract_keys(d, ks):
    return {k: [x[k] for x in d] for k in ks}

def get_strdate_fromtimestamp(timestamp):
    res_date = datetime.fromtimestamp(timestamp)
    str_date = str(res_date)
    return str_date


def get_date(time_str):
    try:
        res_date = date.fromisoformat(time_str)
    except:
        date_str, daytime_str = time_str.strip().split(' ')
        res_date = date.fromisoformat(date_str)
    return res_date


def loweralphanumeric(s):
    s = re.sub(r'[^a-zA-Z0-9]','', s)
    s = s.lower()
    return s


def remap_dict(d, m):
    return {(m[name] if name in m else name): val  for name, val in d.items()}


def find_mode_num(l):
    return np.bincount(l).argmax()


def add_to_filename(path, tail):
    name, ext = os.path.splitext(path)
    return name+tail+ext


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def listDeduplicate(l):
    return list(OrderedDict.fromkeys(l))