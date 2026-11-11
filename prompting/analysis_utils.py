"""Parsing and annotation lookup for prompt-result analyses."""

import ast
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
from shared.utils import load_pickle
from shared.text import DIVIDER, ABBREVIATIONS, protect_abbreviations, restore_abbreviations

CATEGORIES = list('ABCDEFGHI')
CATEGORY_CODE_BY_NAME = dict(
    zip(
        [
            'Inspirations and Influences',
            'Writing Process and Creative Insight',
            'Elements of Fictional Craft',
            'Editorial and Publishing Processes',
            'Quality and Evaluation',
            'Financial and Legal Aspects',
            'Gratitude and Acknowledgement',
            'Marketing and Promotion',
            'Impact and Reception',
        ],
        CATEGORIES,
    ),
)


def read_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_categories(x) -> set:
    if pd.isna(x):
        return set()
    s = str(x).strip()
    if not s or s == '[]':
        return set()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            out = {str(t).strip().upper() for t in val if str(t).strip()}
        else:
            out = {str(val).strip().upper()}
    except Exception:
        out = {
            t.strip().upper()
             for t in s.strip('[]').replace('"', '').replace("'", '').split(',') if t.strip()
        }
    out.discard('J')
    return out


def compute_confusion(gold_has: pd.Series, pred_has: pd.Series) -> dict:
    tp = int((gold_has & pred_has).sum())
    fp = int((~gold_has & pred_has).sum())
    fn = int((gold_has & ~pred_has).sum())
    tn = int((~gold_has & ~pred_has).sum())
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def safe_recall(tp: int, fn: int) -> float:
    denom = tp + fn
    if denom == 0:
        return 0.0
    return tp / denom


def set_to_str(s: set) -> str:
    if not s:
        return ''
    ordered = [c for c in CATEGORIES if c in s]
    return ','.join(ordered)


def iter_sentence_spans_from_anno_text(anno_text: str):
    match = re.search(f'{re.escape(DIVIDER)}\\n\\n(.*?)\\n\\n{re.escape(DIVIDER)}', anno_text, re.DOTALL)
    if match:
        rel_text = match.group(1)
        rel_start = match.start(1)
    else:
        rel_text = anno_text
        rel_start = 0
    rel_text = protect_abbreviations(rel_text, ABBREVIATIONS)
    sentence_pattern = re.compile('.{6,}?[.!?;](?:["\\\'*]*)?(?:\\[\\[\\d+\\]\\])?(?=\\s|$)', re.DOTALL)
    segments = sentence_pattern.findall(rel_text)
    remainder = rel_text[sum((len(s) for s in segments)):].strip()
    if remainder:
        segments.append(remainder)
    sentences = [restore_abbreviations(s) for s in segments]
    sentences = [re.sub('\\[\\[\\d+\\]\\]$', '', s).strip() for s in sentences if s.strip()]
    offset = rel_start
    for i, sent in enumerate(sentences):
        sent_start = anno_text.find(sent, offset)
        if sent_start == -1:
            continue
        sent_end = sent_start + len(sent)
        offset = sent_end
        yield (i, sent_start, sent_end, sent)


def build_subcategory_map():
    letters = load_pickle()
    subcat_map = defaultdict(set)
    reviewed_letters = [
        l
         for l in letters if getattr(l, 'reviewed', False) and getattr(l, 'intertextual_traces', None)
    ]
    for letter in reviewed_letters:
        lid = str(getattr(letter, 'id', ''))
        anno_text = getattr(letter, 'anno_text', None)
        if not lid or not isinstance(anno_text, str) or (not anno_text.strip()):
            continue
        spans = list(iter_sentence_spans_from_anno_text(anno_text))
        if not spans:
            continue
        for sent_id, sent_start, sent_end, _ in spans:
            for trace in getattr(letter, 'intertextual_traces', []):
                for ann in getattr(trace, 'annotations', []):
                    main_cat_name = getattr(ann, 'category', None)
                    if not main_cat_name or main_cat_name not in CATEGORY_CODE_BY_NAME:
                        continue
                    cat_code = CATEGORY_CODE_BY_NAME[main_cat_name]
                    a0 = getattr(ann, 'start_offset', None)
                    a1 = getattr(ann, 'end_offset', None)
                    if a0 is None or a1 is None:
                        continue
                    if a1 <= sent_start or a0 >= sent_end:
                        continue
                    sub = getattr(ann, 'subcategory', None)
                    if not sub:
                        continue
                    if isinstance(sub, list):
                        for sc in sub:
                            if isinstance(sc, str) and sc.strip():
                                subcat_map[lid, sent_id, cat_code].add(sc.strip())
                    elif isinstance(sub, str) and sub.strip():
                        subcat_map[lid, sent_id, cat_code].add(sub.strip())
    return {k: sorted(v) for k, v in subcat_map.items()}


def parse_known_categories(x) -> set:
    if pd.isna(x):
        return set()
    s = str(x).strip()
    if not s or s == '[]':
        return set()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            out = {str(t).strip().upper() for t in val if str(t).strip()}
        else:
            out = {str(val).strip().upper()}
    except Exception:
        out = {
            t.strip().upper()
             for t in s.strip('[]').replace('"', '').replace("'", '').split(',') if t.strip()
        }
    out.discard('J')
    out = {c for c in out if c in CATEGORIES}
    return out
