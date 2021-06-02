# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os


DATASET_ROOT_PATH = './data'


FAKENEWSNET_PATH = os.path.join(DATASET_ROOT_PATH, 'fakenewsnet_dataset')

NELA_PATH = os.path.join(DATASET_ROOT_PATH, 'nela')

NELA_2018_PATH = os.path.join(NELA_PATH, '2018')
NELA_2018_ARTICLE_PATH = os.path.join(NELA_2018_PATH, 'articles')


NELA_2019_PATH = os.path.join(NELA_PATH, '2019')
NELA_2019_ARTICLE_PATH = os.path.join(NELA_2019_PATH, 'nela-eng-2019')

FAKENEWSNET_RAW_PATH = os.path.join(FAKENEWSNET_PATH, 'raw')

