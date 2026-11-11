"""Tabular views of reviewed letters and their annotations."""

from shared.labels import CATEGORY_INDEX
from shared.text import DIVIDER, ABBREVIATIONS, protect_abbreviations, restore_abbreviations
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pprint import pprint
from shared.utils import load_pickle
from shared.classes import Letter, Annotation


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def is_uncertain(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == 'y'
    return False


def split_annotation_sentences(anno_text):
    match = re.search(f'{DIVIDER}\\n\\n(.*?)\\n\\n{DIVIDER}', anno_text, re.DOTALL)
    if not match:
        return ([], None, None)
    rel_text = match.group(1)
    rel_start = match.start(1)
    protected_text = protect_abbreviations(rel_text, ABBREVIATIONS)
    sentence_pattern = re.compile('.{6,}?[.!?;](?:["\\\'*]*)?(?:\\[\\[\\d+\\]\\])?(?=\\s|$)', re.DOTALL)
    segments = sentence_pattern.findall(protected_text)
    remainder = protected_text[sum((len(s) for s in segments)):].strip()
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
    return (sentence_data, rel_text, rel_start)


def prepare_tables():
    """Build letter, sentence, trace, and annotation tables from the local corpus."""
    id2label = {i: label for label, i in CATEGORY_INDEX.items()}
    label2id = CATEGORY_INDEX
    NUM_CATEGORIES = len(label2id)
    print('Label mapping:')
    pprint(label2id)
    letters = load_pickle()
    reviewed_letters = [letter for letter in letters if getattr(letter, 'reviewed', False)]
    all_sentences = []
    letter_rows = []
    sentence_rows = []
    trace_rows = []
    annotation_rows = []
    sentence_annotation_rows = []
    skipped_no_relevant_block = []
    for letter in reviewed_letters:
        anno_text = getattr(letter, 'anno_text', '')
        sentence_data, rel_text, rel_start = split_annotation_sentences(anno_text)
        if not sentence_data:
            skipped_no_relevant_block.append(letter.id)
            continue
        traces = getattr(letter, 'intertextual_traces', []) or []
        sentence_to_cats = defaultdict(set)
        for trace in traces:
            for ann in getattr(trace, 'annotations', []) or []:
                if not getattr(ann, 'category', None):
                    continue
                for i, (sent_start, sent_end, _) in enumerate(sentence_data):
                    if not (ann.end_offset <= sent_start or ann.start_offset >= sent_end):
                        sentence_to_cats[i].add(ann.category)
        letter_rows.append(
            {
                'letter_id': letter.id,
                'reviewed': getattr(letter, 'reviewed', False),
                'is_negative_example': getattr(letter, 'is_negative_example', False),
                'has_intertextual_traces': bool(traces),
                'trace_count': len(traces),
                'sentence_count': len(sentence_data),
                'anno_text_length_chars': len(anno_text),
                'relevant_text_length_chars': len(rel_text) if rel_text is not None else 0,
            },
        )
        for trace_index, trace in enumerate(traces):
            annotations = getattr(trace, 'annotations', []) or []
            trace_rows.append(
                {
                    'letter_id': letter.id,
                    'trace_index': trace_index,
                    'annotation_count': len(annotations),
                },
            )
            for annotation_index, ann in enumerate(annotations):
                annotation_id = f'{letter.id}__t{trace_index}__a{annotation_index}'
                start_offset = getattr(ann, 'start_offset', None)
                end_offset = getattr(ann, 'end_offset', None)
                annotation_rows.append(
                    {
                        'annotation_id': annotation_id,
                        'letter_id': letter.id,
                        'trace_index': trace_index,
                        'annotation_index': annotation_index,
                        'start_offset': start_offset,
                        'end_offset': end_offset,
                        'phrase': getattr(ann, 'phrase', None),
                        'excerpt': anno_text[start_offset:end_offset] if start_offset is not None and end_offset is not None else None,
                        'category': getattr(ann, 'category', None),
                        'subcategory': getattr(ann, 'subcategory', None),
                        'target_novel': getattr(ann, 'target_novel', None),
                        'temporal_perspective': getattr(ann, 'temporal_perspective', None),
                        'uncertain': getattr(ann, 'uncertain', None),
                        'type_of_reference': getattr(ann, 'type_of_reference', None),
                    },
                )
        for i, (s, e, txt) in enumerate(sentence_data):
            cats = sentence_to_cats.get(i, set())
            category_vector = [0] * NUM_CATEGORIES
            valid_categories = set()
            invalid_categories = set()
            for cat in cats:
                if cat in CATEGORY_INDEX:
                    category_vector[CATEGORY_INDEX[cat]] = 1
                    valid_categories.add(cat)
                else:
                    invalid_categories.add(cat)
            subcategories = set()
            target_novels = set()
            temporal_perspectives = set()
            reference_types = set()
            uncertain = False
            overlapping_annotation_ids = []
            overlapping_annotation_count = 0
            for trace_index, trace in enumerate(traces):
                for annotation_index, ann in enumerate(getattr(trace, 'annotations', []) or []):
                    if not getattr(ann, 'category', None):
                        continue
                    if not (ann.end_offset <= s or ann.start_offset >= e):
                        annotation_id = f'{letter.id}__t{trace_index}__a{annotation_index}'
                        overlapping_annotation_ids.append(annotation_id)
                        overlapping_annotation_count += 1
                        sentence_annotation_rows.append(
                            {
                                'letter_id': letter.id,
                                'sentence_id': i,
                                'annotation_id': annotation_id,
                                'trace_index': trace_index,
                                'annotation_index': annotation_index,
                                'category': getattr(ann, 'category', None),
                                'subcategory': getattr(ann, 'subcategory', None),
                                'target_novel': getattr(ann, 'target_novel', None),
                                'temporal_perspective': getattr(ann, 'temporal_perspective', None),
                                'uncertain': getattr(ann, 'uncertain', None),
                                'type_of_reference': getattr(ann, 'type_of_reference', None),
                            },
                        )
                        for subcat in as_list(getattr(ann, 'subcategory', None)):
                            if subcat:
                                subcategories.add(subcat)
                        for target in as_list(getattr(ann, 'target_novel', None)):
                            if target:
                                target_novels.add(target)
                        for temporal in as_list(getattr(ann, 'temporal_perspective', None)):
                            if temporal:
                                temporal_perspectives.add(temporal)
                        for ref_type in as_list(getattr(ann, 'type_of_reference', None)):
                            if ref_type:
                                reference_types.add(ref_type)
                        if is_uncertain(getattr(ann, 'uncertain', None)):
                            uncertain = True
            has_reference = any(category_vector)
            row = {
                'letter_id': letter.id,
                'sentence_id': i,
                'sentence_start': s,
                'sentence_end': e,
                'sentence_text': txt,
                'category_vector': category_vector,
                'has_reference': has_reference,
                'categories': sorted(valid_categories),
                'invalid_categories': sorted(invalid_categories),
                'subcategories': sorted(subcategories),
                'target_novels': sorted(target_novels),
                'temporal_perspectives': sorted(temporal_perspectives),
                'uncertain': uncertain,
                'reference_types': sorted(reference_types),
                'overlapping_annotation_ids': overlapping_annotation_ids,
                'overlapping_annotation_count': overlapping_annotation_count,
            }
            sentence_rows.append(row)
            all_sentences.append(
                {
                    'letter_id': letter.id,
                    'sentence_id': i,
                    'context': txt,
                    'category_vector': category_vector,
                    'subcategories': sorted(subcategories),
                    'target_novels': sorted(target_novels),
                    'temporal_perspectives': sorted(temporal_perspectives),
                    'uncertain': uncertain,
                    'reference_types': sorted(reference_types),
                },
            )
    letters_df = pd.DataFrame(letter_rows)
    sentences_df = pd.DataFrame(sentence_rows)
    traces_df = pd.DataFrame(trace_rows)
    annotations_df = pd.DataFrame(annotation_rows)
    sentence_annotation_df = pd.DataFrame(sentence_annotation_rows)
    if not sentences_df.empty:
        sentence_agg = sentences_df.groupby('letter_id').agg(
            extracted_sentence_count=('sentence_id', 'count'),
            reference_sentence_count=('has_reference', 'sum'),
            annotation_overlap_sentence_count=('overlapping_annotation_count', lambda x: (x > 0).sum()),
        ).reset_index()
        letters_df = letters_df.merge(sentence_agg, on='letter_id', how='left')
        for col in [
            'extracted_sentence_count',
            'reference_sentence_count',
            'annotation_overlap_sentence_count',
        ]:
            letters_df[col] = letters_df[col].fillna(0).astype(int)
        letters_df['reference_sentence_share'] = letters_df.apply(
            lambda row: row['reference_sentence_count'] / row['extracted_sentence_count'] if row['extracted_sentence_count'] else 0,
            axis=1,
        )
    if not annotations_df.empty:
        annotation_agg = annotations_df.groupby('letter_id').agg(annotation_count=('annotation_id', 'count')).reset_index()
        letters_df = letters_df.merge(annotation_agg, on='letter_id', how='left')
        letters_df['annotation_count'] = letters_df['annotation_count'].fillna(0).astype(int)
    else:
        letters_df['annotation_count'] = 0
    print('\nData loaded and structured.')
    print(f'Letters loaded from pickle: {len(letters):,}')
    print(f'Reviewed letters: {len(reviewed_letters):,}')
    print(f'Reviewed letters with extracted sentences: {len(letters_df):,}')
    print(
        f'Skipped reviewed letters without extracted relevant text block: {len(skipped_no_relevant_block):,}',
    )
    print(f'\nExtracted sentences from all reviewed letters: {len(sentences_df):,}')
    print(
        f"Referential sentences according to category_vector: {sentences_df['has_reference'].sum():,}",
    )
    print(
        f"NoRef sentences according to category_vector: {(~sentences_df['has_reference']).sum():,}",
    )
    print(f'\nTraces extracted: {len(traces_df):,}')
    print(f'Annotations extracted: {len(annotations_df):,}')
    print(f'Annotation-sentence links: {len(sentence_annotation_df):,}')
    return {
        'skipped_no_relevant_block': skipped_no_relevant_block,
        'letters': letters,
        'reviewed_letters': reviewed_letters,
        'letters_df': letters_df,
        'sentences_df': sentences_df,
        'traces_df': traces_df,
        'annotations_df': annotations_df,
        'sentence_annotation_df': sentence_annotation_df,
        'all_sentences': all_sentences,
    }


def has_valid_reference_annotation(letter):
    traces = getattr(letter, 'intertextual_traces', []) or []
    for trace in traces:
        annotations = getattr(trace, 'annotations', []) or []
        for ann in annotations:
            category = getattr(ann, 'category', None)
            if category in CATEGORY_INDEX:
                return True
    return False
