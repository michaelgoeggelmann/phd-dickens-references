"""Lemma-based keyword classification of reference categories."""

from shared import paths
from shared.labels import CATEGORY_INDEX
from shared.sentences import iter_sentences, baseline_context
import os
import re
import spacy
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report
from tqdm import tqdm
from shared.utils import load_pickle


def run():
    output_dir = str(paths.SENTENCE_BASELINES / 'baseline_class')
    os.makedirs(output_dir, exist_ok=True)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    NUM_CATEGORIES = len(CATEGORY_INDEX)
    KEYWORDS = {
        0: {
            'proof',
            'revise',
            'correct',
            'draft',
            'submit',
            'print',
            'publish',
            'instalment',
            'translate',
            'reprint',
        },
        1: {
            'character',
            'plot',
            'story',
            'micawber',
            'fagin',
            'gamp',
            'pecksniff',
            'dialogue',
            'scene',
            'setting',
        },
        2: {
            'contract',
            'agreement',
            'payment',
            'terms',
            'advance',
            'copyright',
            'piracy',
            'offer',
            'clause',
            'royalty',
        },
        3: {
            'gift',
            'present',
            'copy',
            'send',
            'dedicate',
            'inscription',
            'gratitude',
            'thanks',
            'celebration',
            'dinner',
        },
        4: {
            'reception',
            'readership',
            'audience',
            'popularity',
            'acclaim',
            'praise',
            'review',
            'response',
            'success',
            'adaptation',
        },
        5: {
            'influence',
            'inspire',
            'source',
            'basis',
            'shape',
            'derive',
            'idea',
            'prompted by',
            'drew on',
            'connected to',
        },
        6: {
            'advertise',
            'promote',
            'announce',
            'publicise',
            'audience',
            'lecture',
            'reading',
            'recommend',
            'circulation',
            'engage',
        },
        7: {
            'quality',
            'merit',
            'excellence',
            'success',
            'best',
            'like',
            'prefer',
            'handsome',
            'fine',
            'fault',
        },
        8: {
            'begin',
            'write',
            'draft',
            'revise',
            'finish',
            'chapter',
            'progress',
            'task',
            'pace',
            'complete',
        },
    }

    def lemmatize(text):
        return {token.lemma_.lower() for token in nlp(text)}
    sentence_dataset = list(iter_sentences(load_pickle()))
    for w in range(0, 7):
        print(f'\n Context window {w} is being processed...')
        all_gold = []
        all_pred = []
        results = []
        for entry in tqdm(sentence_dataset, desc=f'Window {w}'):
            sentence_context = baseline_context(entry, w)
            gold = entry['category_vector']
            lemmas = lemmatize(sentence_context)
            pred = [0] * NUM_CATEGORIES
            for cat_id in range(NUM_CATEGORIES):
                if any((word in lemmas for word in KEYWORDS[cat_id])):
                    pred[cat_id] = 1
            all_gold.append(gold)
            all_pred.append(pred)
            results.append(
                {
                    'letter_id': entry['letter_id'],
                    'sentence_id': entry['sentence_id'],
                    'context_window': w,
                    'sentence': sentence_context,
                    'gold_vector': gold,
                    'pred_vector': pred,
                },
            )
        report = classification_report(
            all_gold,
            all_pred,
            target_names=list(CATEGORY_INDEX.keys()),
            zero_division=0,
            digits=4,
        )
        print(report)
        with open(os.path.join(output_dir, f'classification_report_window{w}.txt'), 'w') as f:
            f.write(report)
        df = pd.DataFrame(results)
        df.to_csv(
            os.path.join(output_dir, f'keyword_baseline_predictions_window{w}.csv'),
            index=False,
        )
    print(f'\n Baseline results saved to {output_dir}')
if __name__ == '__main__':
    run()
