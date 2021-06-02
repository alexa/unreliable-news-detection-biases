# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import argparse
import numpy as np


def analysis_lr(args):
    '''Print the high importance features used by the linear regression model'''

    num_features = args.topk_features
    ignore_idf = args.ignore_idf

    # load model
    with open(args.model_path, 'rb') as fr:
        model = pickle.load(fr)

    vect = model.get_params()['vect']
    tfidf = model.get_params()['tfidf']
    clf = model.get_params()['clf']

    vocab = vect.vocabulary_
    idfs = tfidf.idf_
    clf_weights = clf.coef_

    rev_vocab = {idx: tok for tok, idx in vocab.items()}

    label_dim, vocab_size = clf_weights.shape
    assert(label_dim == 1) #Otherwise, you shouldn't flatten it
    flat_weights = clf_weights.flatten()
    print(ignore_idf)
    if not ignore_idf:
        flat_weights = flat_weights * idfs
    inc_idx = np.argsort(flat_weights)
    pos_idx = inc_idx[-num_features:][::-1]
    neg_idx = inc_idx[:num_features]

    print("POSTIVE FEATURES")
    for i in range(num_features):
        print(f"FEATURE #{i}:\tTOKEN:{rev_vocab[pos_idx[i]]},\tWEIGHT:{flat_weights[pos_idx[i]]}")

    print("NEGATIVE FEATURES")
    for i in range(num_features):
        print(f"FEATURE #{i}:\tTOKEN:{rev_vocab[neg_idx[i]]},\tWEIGHT:{flat_weights[neg_idx[i]]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        default=None,
        type=str,
        required=True,
        help='The model pickle file'
    )

    parser.add_argument(
        '--topk_features',
        default=5,
        type=int,
        help='how many top features to print'
    )

    parser.add_argument(
        '--ignore_idf',
        action="store_true",
        help='if set True, ignore the influence of the idf step'
    )

    args = parser.parse_args()
    analysis_lr(args)