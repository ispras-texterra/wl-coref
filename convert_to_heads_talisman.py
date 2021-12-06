from collections import defaultdict
import logging
import os
from typing import Tuple

import jsonlines


DATA_DIR = "data"
FILENAME = "english_{}{}.jsonlines"
LOGGING_LEVEL = logging.WARNING  # DEBUG to output all duplicate spans
SPLITS = ("development", "test", "train")


def get_head(mention: Tuple[int, int], doc: dict) -> int:
    """Returns the span's head, which is defined as the only word within the
    span whose head is outside of the span or None. In case there are no or
    several such words, the rightmost word is returned

    Args:
        mention (Tuple[int, int]): start and end (exclusive) of a span
        doc (dict): the document data

    Returns:
        int: word id of the spans' head
    """
    head_candidates = set()
    start, end = mention
    for i in range(start, end):
        ith_head = doc["head"][i]
        if ith_head is None or not (start <= ith_head < end):
            head_candidates.add(i)
    if len(head_candidates) == 1:
        return head_candidates.pop()
    return end - 1


def doc_from_text(text, sents_id, part_id=0, document_id="bc/cctv/00/cctv_0000"):
    # 'pos', 'deprel', 'head', 'head2span', 'word_clusters', 'span_clusters', 'word2subword', 'subwords', 'word_id']
    doc = {
            "document_id":      document_id,
            "cased_words":      [],
            "sent_id":          [],
            "part_id":          part_id,
            "speaker":          [],
            "clusters":         [],
            "subwords":         [],
            "span_clusters":    [],
            "head":             []
        }

    for word, sent_id in zip(text, sents_id):
        doc["cased_words"].append(word)
        doc["sent_id"].append(sent_id)
        doc["speaker"].append(0)
        doc["subwords"].append(word)
        doc["head"].append(None)


    return doc


