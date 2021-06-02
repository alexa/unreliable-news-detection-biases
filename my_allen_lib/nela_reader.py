'''
Adapted from:
 https://github.com/allenai/allennlp-models/blob/main/allennlp_models/pair_classification/dataset_readers/snli.py
'''

from typing import Dict, List, Any, Optional
import logging

from overrides import overrides
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer
from allennlp.common.checks import ConfigurationError

import json

logger = logging.getLogger(__name__)


@DatasetReader.register("nela")
class NelaDatasetReader(DatasetReader):
    """
    Reads tokens and their sentiment labels from the Stanford Sentiment Treebank.

    The Stanford Sentiment Treebank comes with labels
    from 0 to 4. `"5-class"` uses these labels as is. `"3-class"` converts the
    problem into one of identifying whether a sentence is negative, positive, or
    neutral sentiment. In this case, 0 and 1 are grouped as label 0 (negative sentiment),
    2 is converted to label 1 (neutral sentiment) and 3 and 4 are grouped as label 2
    (positive sentiment). `"2-class"` turns it into a binary classification problem
    between positive and negative sentiment. 0 and 1 are grouped as the label 0
    (negative sentiment), 2 (neutral) is discarded, and 3 and 4 are grouped as the label 1
    (positive sentiment).

    Expected format for each input line: a linearized tree, where nodes are labeled
    by their sentiment.

    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`

    Registered as a `DatasetReader` with name "sst_tokens".

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        input_key: str = "content",
        label_key: str = "label",
        use_title: str = "none",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._input_key = input_key
        self._label_key = label_key
        assert(use_title in ["none", "concat", "pair"])
        self._use_title = use_title
        self._max_length = self._tokenizer._max_length

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                example = json.loads(line)
                instance = self.text_to_instance(example)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(
        self, example: Dict[str, Any]
    ) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        # Parameters

        tokens : `List[str]`, required.
            The tokens in a given sentence.
        sentiment : `str`, optional, (default = None).
            The sentiment for this sentence.

        # Returns

        An `Instance` containing the following fields:
            tokens : `TextField`
                The tokens in the sentence or phrase.
            label : `LabelField`
                The sentiment label of the sentence or phrase.
        """
        if self._use_title == "concat":
            tokens = self._tokenizer.tokenize(example['title'] + " " + example[self._input_key])
        elif self._use_title == "none":
            tokens = self._tokenizer.tokenize(example[self._input_key])
        elif self._use_title == "pair":
            assert(self._tokenizer._add_special_tokens==False)
            title_tokens = self._tokenizer.tokenize(example['title'])
            content_tokens = self._tokenizer.tokenize(example[self._input_key])
            tokens = self._tokenizer.add_special_tokens(title_tokens, content_tokens)
            # A little bit of hacking to maintain the max length
            if self._max_length is not None:
                if len(tokens) > self._max_length:
                    tokens = tokens[:self._max_length-1] + tokens[-1:]



        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        label = example[self._label_key]
        if type(label) is int:
            label = str(label)
        fields["label"] = LabelField(label)
        metadata = {}
        metadata['content'] = example[self._input_key]
        if 'title' in example.keys():
            metadata['title'] = example['title']
        if 'source' in example.keys():
            metadata['source'] = example['source']

        return Instance(fields)
