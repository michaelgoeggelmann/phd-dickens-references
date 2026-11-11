"""Sentence segmentation, annotation overlap, and model inputs."""

import re
from collections import defaultdict
from shared.labels import CATEGORY_INDEX
from shared.text import (
    DIVIDER,
    ABBREVIATIONS,
    protect_abbreviations,
    restore_abbreviations,
    build_context,
)

NUM_CATEGORIES = len(CATEGORY_INDEX)


def iter_sentences(letters, require_traces=False):
    """Yield reviewed sentences with overlapping annotation categories."""
    for letter in letters:
        if (
            not getattr(letter, 'reviewed', False)
            or require_traces and (not letter.intertextual_traces)
        ):
            continue
        anno_text = letter.anno_text
        match = re.search(f'{DIVIDER}\\n\\n(.*?)\\n\\n{DIVIDER}', anno_text, re.DOTALL)
        if not match:
            continue
        rel_text = match.group(1)
        rel_start = match.start(1)
        rel_text = protect_abbreviations(rel_text, ABBREVIATIONS)
        sentence_pattern = re.compile('.{6,}?[.!?;](?:["\\\'*]*)?(?:\\[\\[\\d+\\]\\])?(?=\\s|$)', re.DOTALL)
        segments = sentence_pattern.findall(rel_text)
        remainder = rel_text[sum((len(s) for s in segments)):].strip()
        if remainder:
            segments.append(remainder)
        sentences = [restore_abbreviations(s) for s in segments]
        sentences = [re.sub('\\[\\[\\d+\\]\\]$', '', s).strip() for s in sentences if s.strip()]
        offset = rel_start
        sentence_data = []
        for sent in sentences:
            sent_start = anno_text.find(sent, offset)
            if sent_start == -1:
                continue
            sent_end = sent_start + len(sent)
            offset = sent_end
            sentence_data.append((sent_start, sent_end, sent))
        sentence_to_cats = defaultdict(set)
        for trace in getattr(letter, 'intertextual_traces', []) or []:
            for ann in trace.annotations:
                if not ann.category:
                    continue
                for i, (sent_start, sent_end, _) in enumerate(sentence_data):
                    if not (ann.end_offset <= sent_start or ann.start_offset >= sent_end):
                        sentence_to_cats[i].add(ann.category)
        for i, (s, e, txt) in enumerate(sentence_data):
            cats = sentence_to_cats.get(i, set())
            vec = [0] * NUM_CATEGORIES
            for cat in cats:
                if cat in CATEGORY_INDEX:
                    vec[CATEGORY_INDEX[cat]] = 1
            yield {
                'letter_id': letter.id,
                'sentence_id': i,
                'text': txt,
                'sentence_data': sentence_data,
                'category_vector': vec,
                'is_ref': int(bool(cats)),
            }


def prepare_sentences(letters, window_size=0, use_target_tags=True, require_traces=False):
    """Build sentence inputs with the requested context window and target markers."""
    rows = []
    for entry in iter_sentences(letters, require_traces=require_traces):
        context = build_context(entry['sentence_data'], entry['sentence_id'], window_size, use_target_tags)
        rows.append(
            {
                'letter_id': entry['letter_id'],
                'sentence_id': entry['sentence_id'],
                'context': context,
                'category_vector': entry['category_vector'],
            },
        )
    return rows


def baseline_context(entry, window_size):
    """Join a target sentence and its neighbours for the heuristic baselines."""
    i = entry['sentence_id']
    sentence_data = entry['sentence_data']
    previous = ' '.join((sentence_data[j][2] for j in range(max(0, i - window_size), i)))
    following = ' '.join(
        (sentence_data[j][2] for j in range(i + 1, min(len(sentence_data), i + 1 + window_size))),
    )
    if window_size == 0:
        return entry['text']
    return f"{previous} <target> {entry['text']} </target> {following}".strip()
