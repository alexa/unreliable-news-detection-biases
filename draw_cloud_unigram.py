# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from os import path
from wordcloud import WordCloud
from collections import Counter
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

eng_stopwords = stopwords.words('english')

PRINT_TYPE = "CORRECT"

assert (PRINT_TYPE in ['CORRECT', 'WRONG'])


def get_high_pmi_words(c_text, w_text, num_words=100):
    '''We did not actually calculate pmi here, but the order should be the same. PMI=log p(w,l)/p(w,)p(,l) Here we calcualte freq(w,l)/freq(w)freq(l)'''
    word_counter, label_counter, word_label_counter = Counter(), Counter(), Counter()
    c_words = c_text.strip().split()
    label = 1
    for w in c_words:
        word_counter[w] += 1
        label_counter[label] += 1
        word_label_counter[(w, label)] += 1
    w_words = w_text.strip().split()
    label = 0
    for w in w_words:
        word_counter[w] += 1
        label_counter[label] += 1
        word_label_counter[(w, label)] += 1

    pmi_dicts = {}
    smoothing_num = 100
    for k in word_label_counter.keys():
        w, label = k
        pmi_dicts[k] = word_label_counter[k] / ((word_counter[w] + smoothing_num) * label_counter[label])

    c_str = ''
    w_str = ''
    for l in label_counter.keys():
        kvs = []
        for wl, pmi in pmi_dicts.items():
            w, label = wl
            if label == l:
                kvs.append((w, pmi))
        sorted_kvs = sorted(kvs, reverse=True, key=lambda x: x[1])
        top_words = [x[0] for x in sorted_kvs[:num_words]]
        weight = 1000 / np.max([x[1] for x in sorted_kvs[:num_words]])
        print(top_words)
        if l == 1:
            for k, v in sorted_kvs[:num_words]:
                # print(k,v)
                c_str += ' '.join([k for _ in range(int(weight * v))])
        elif l == 0:
            for k, v in sorted_kvs[:num_words]:
                w_str += ' '.join([k for _ in range(int(weight * v))])
    return c_str, w_str


def get_frequency_dict_for_text(sentence):
    tmpDict = {}

    # making dict for counting frequencies
    for text in sentence.split(" "):
        if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        if text.strip().lower() in eng_stopwords:
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    return tmpDict


d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

correct_text = open(path.join(d, 'correct.title')).read()
wrong_text = open(path.join(d, 'wrong.title')).read()

c_str, w_str = get_high_pmi_words(correct_text, wrong_text)

import matplotlib.pyplot as plt

if PRINT_TYPE == 'CORRECT':
    freq = get_frequency_dict_for_text(c_str)
else:
    freq = get_frequency_dict_for_text(w_str)

wc = WordCloud(max_font_size=60, background_color='white', width=800, height=400).generate_from_frequencies(freq)
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
if PRINT_TYPE == 'CORRECT':
    plt.savefig('cloud_correct.png')
else:
    plt.savefig('cloud_wrong.png')