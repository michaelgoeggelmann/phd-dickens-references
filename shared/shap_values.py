"""Token truncation and offset-based aggregation of SHAP values."""


def truncate_long_texts(texts, ids, tokenizer, max_length=508):
    """Shorten inputs to the token budget used for SHAP."""
    truncated_texts = []
    truncated_ids = []
    for text, text_id in zip(texts, ids):
        tokenized = tokenizer(text, add_special_tokens=False)
        input_ids = tokenized['input_ids']
        if len(input_ids) > max_length:
            truncated_ids.append(text_id)
            input_ids = input_ids[:max_length]
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
        truncated_texts.append(text)
    print(f' {len(truncated_ids)} texts were truncated.')
    if truncated_ids:
        print(f' Truncated text IDs: {truncated_ids}')
    else:
        print(' No truncation needed.')
    return truncated_texts
import numpy as np
import pandas as pd
import shap
from transformers import AutoTokenizer


def merge_shap_values_with_offsets(original_text, shap_values, tokenizer):
    """Merge tokenizer pieces using their character offsets."""
    encoding = tokenizer(original_text, return_offsets_mapping=True, add_special_tokens=True)
    tokens = encoding.tokens()
    offsets = encoding['offset_mapping']
    merged_tokens = []
    merged_shap_values = []
    current_word = ''
    current_shap = np.zeros_like(shap_values[0])
    current_offset = offsets[0]
    footnote_buffer = ''
    footnote_shap = np.zeros_like(shap_values[0])
    for i, (token, shap_val, offset) in enumerate(zip(tokens, shap_values, offsets)):
        if i == 0:
            merged_tokens.append('[CLS]')
            merged_shap_values.append(shap_val)
            continue
        if i == len(tokens) - 1:
            if current_word:
                merged_tokens.append(current_word)
                merged_shap_values.append(current_shap)
            merged_tokens.append('[SEP]')
            merged_shap_values.append(shap_val)
            current_word = ''
            continue
        if token in {'.', ',', '!', '?', '*', ':', ';', '(', ')', '"', "'", '|'}:
            if current_word:
                merged_tokens.append(current_word)
                merged_shap_values.append(current_shap)
            merged_tokens.append(token)
            merged_shap_values.append(shap_val)
            current_word = ''
            current_shap = np.zeros_like(shap_values[0])
            continue
        if token == '[' and (not footnote_buffer):
            footnote_buffer = '['
            footnote_shap = shap_val
            continue
        if footnote_buffer:
            footnote_buffer += token
            footnote_shap += shap_val
            if footnote_buffer.endswith(']]'):
                if current_word:
                    merged_tokens.append(current_word)
                    merged_shap_values.append(current_shap)
                merged_tokens.append(footnote_buffer)
                merged_shap_values.append(footnote_shap)
                footnote_buffer = ''
                footnote_shap = np.zeros_like(shap_values[0])
                current_word = ''
                current_shap = np.zeros_like(shap_values[0])
            continue
        if i > 0 and offset[0] > current_offset[1]:
            if current_word:
                merged_tokens.append(current_word)
                merged_shap_values.append(current_shap)
            current_word = token
            current_shap = shap_val
        else:
            if token.startswith('##'):
                current_word += token[2:]
            else:
                current_word += token
            current_shap += shap_val
        current_offset = offset
    if current_word:
        merged_tokens.append(current_word)
        merged_shap_values.append(current_shap)
    return (merged_tokens, np.array(merged_shap_values))
