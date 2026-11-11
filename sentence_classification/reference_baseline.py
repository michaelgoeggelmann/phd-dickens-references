"""Title and named-entity heuristics for sentence-level reference detection."""

from shared import paths
from shared.sentences import iter_sentences, baseline_context
import os
import re
import torch
import spacy
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast
from shared.utils import load_pickle


def run():
    output_dir = str(paths.SENTENCE_BASELINES / 'baseline_ref')
    os.makedirs(output_dir, exist_ok=True)
    reference_titles = [
        'Horatio Sparkins',
        'Sketches by Boz',
        'Pickwick Papers',
        'Oliver Twist',
        'Nicholas Nickleby',
        'Old Curiosity Shop',
        'Barnaby Rudge',
        "Master Humphrey's Clock",
        'American Notes',
        'A Christmas Carol',
        'Martin Chuzzlewit',
        'The Chimes',
        'The Cricket on the Hearth',
        'The Battle of Life',
        "The Haunted Man and the Ghost's Bargain",
        'The Lamplighter',
        'Pictures of Italy',
        'Dombey and Son',
        'David Copperfield',
        'Bleak House',
        'Hard Times',
        "A Child's History of England",
        'Little Dorrit',
        'A Tale of Two Cities',
        'Great Expectations',
        'Our Mutual Friend',
        'Edwin Drood',
    ]
    specific_words = [
        'Sparkins',
        'Pickwick',
        'Nickleby',
        'Barnaby',
        'Chuzzlewit',
        'Dombey',
        'Copperfield',
        'Dorrit',
        'Drood',
        'Carol',
    ]
    nlp = spacy.load('en_core_web_sm')

    def has_title_mention(sentence):
        return (
            any((title in sentence for title in reference_titles))
            or any((w in sentence for w in specific_words))
        )

    def has_ner_trigger(sentence):
        doc = nlp(sentence)
        return any((ent.label_ in {'PERSON', 'WORK_OF_ART', 'ORG', 'EVENT', 'GPE'} for ent in doc.ents))

    def predict_ref(sentence, method):
        if method == 'title':
            return has_title_mention(sentence)
        elif method == 'ner':
            return has_ner_trigger(sentence)
        else:
            return False

    def run_baseline(data, variant_name):
        methods = ['title', 'ner']
        results = {m: [] for m in methods}
        gold = []
        csv_rows = []
        for entry in data:
            sent = entry['sentence']
            gold_label = entry['is_ref']
            gold.append(gold_label)
            row = {
                'letter_id': entry['letter_id'],
                'sentence_id': entry['sentence_id'],
                'sentence': sent,
                'gold': gold_label,
            }
            for method in methods:
                pred = int(predict_ref(sent, method))
                results[method].append(pred)
                row[f'pred_{method}'] = pred
            csv_rows.append(row)
        for method in methods:
            print(f'\n Evaluation: {method.upper()} (window={variant_name})')
            report = classification_report(gold, results[method], target_names=['NoRef', 'Ref'], digits=4)
            print(report)
            with open(
                os.path.join(output_dir, f'classification_report_{method}_window{variant_name}.txt'),
                'w',
            ) as f:
                f.write(report)
        df = pd.DataFrame(csv_rows)
        df.to_csv(
            os.path.join(output_dir, f'ref_baseline_predictions_window{variant_name}.csv'),
            index=False,
        )
    sentence_dataset = list(iter_sentences(load_pickle()))
    for w in range(0, 7):
        dataset_variant = []
        for entry in sentence_dataset:
            dataset_variant.append(
                {
                    'letter_id': entry['letter_id'],
                    'sentence_id': entry['sentence_id'],
                    'sentence': baseline_context(entry, w),
                    'is_ref': entry['is_ref'],
                },
            )
        print(f'\n Evaluating the baseline with context window {w}...')
        run_baseline(dataset_variant, str(w))
    print(f'\n Baselines for contexts 0 to 6 completed. Results: {output_dir}')
if __name__ == '__main__':
    run()
