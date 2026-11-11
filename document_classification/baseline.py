"""Hierarchical work-title matching for document-level reference detection."""

from shared import paths
import pickle
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils import Bunch


def run():
    paths.prepare_directories()
    pickle_path = str(paths.DOCUMENT_DATA / 'processed_data_metadata.pkl')
    output_dir = str(paths.DOCUMENT_BASELINE)
    colors = ['#17becf', '#9467bd', '#7f7f7f', '#1f497d']
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
    ]
    stopwords = {
        'the',
        'and',
        'of',
        'in',
        'on',
        'at',
        'a',
        'an',
        'to',
        'by',
        'with',
        'for',
        'our',
        'mutual',
        'friend',
    }
    token_window_size = 10

    def tokenize(text):
        """Tokenise text for the title-matching heuristic."""
        return re.findall('\\w+', text.lower())

    def exact_title_match(text, title):
        """Search for a full title, co-occurring title words, or a distinctive name."""
        return re.search(f'[^\\w]?{re.escape(title)}[^\\w]?', text, re.IGNORECASE) is not None

    def co_occurrence_search(text_tokens, title):
        """Check whether title tokens occur within the configured window."""
        title_tokens = [word.lower() for word in title.split() if word.lower() not in stopwords]
        if len(title_tokens) == 1:
            return any(
                (re.search(f'[^\\w]?{re.escape(title_tokens[0])}[^\\w]?', token) for token in text_tokens),
            )
        word_positions = {word: [] for word in title_tokens}
        for idx, token in enumerate(text_tokens):
            if token in word_positions:
                word_positions[token].append(idx)
        for i, word1 in enumerate(title_tokens):
            for j, word2 in enumerate(title_tokens):
                if i != j:
                    pos1 = word_positions[word1]
                    pos2 = word_positions[word2]
                    if (
                        pos1
                        and pos2
                        and any((abs(p1 - p2) <= token_window_size for p1 in pos1 for p2 in pos2))
                    ):
                        return True
        return False
    with open(pickle_path, 'rb') as f:
        processed_data = pickle.load(f)
    true_labels = []
    predicted_labels = []
    detected_examples = []
    exact_match_preds = []
    specific_word_preds = []
    co_occurrence_preds = []
    exact_match_count = 0
    specific_word_count = 0
    co_occurrence_count = 0
    baseline_predictions = {}
    method_applied = {'exact': [], 'cooccurrence': [], 'specific': []}
    for letter_id, text, label in zip(processed_data['id'], processed_data['text'], processed_data['label']):
        text_tokens = tokenize(text)
        found_titles_exact = [title for title in reference_titles if exact_title_match(text, title)]
        found_titles_cooccurrence = []
        found_specific_words = []
        if found_titles_exact:
            method_applied['exact'].append(letter_id)
        else:
            found_titles_cooccurrence = [title for title in reference_titles if co_occurrence_search(text_tokens, title)]
            if found_titles_cooccurrence:
                method_applied['cooccurrence'].append(letter_id)
            else:
                found_specific_words = [
                    word
                     for word in specific_words if re.search(f'\\b{re.escape(word)}\\b', text, re.IGNORECASE)
                ]
                if found_specific_words:
                    method_applied['specific'].append(letter_id)
        detected = bool(found_titles_exact or found_titles_cooccurrence or found_specific_words)
        true_labels.append(label)
        predicted_labels.append(int(detected))
        baseline_predictions[letter_id] = int(detected)
        exact_match_preds.append(int(bool(found_titles_exact)))
        co_occurrence_preds.append(int(bool(found_titles_cooccurrence)))
        specific_word_preds.append(int(bool(found_specific_words)))
    all_ids = processed_data['id']
    label_dict = dict(zip(all_ids, true_labels))
    pred_dict = {
        'exact': dict(zip(all_ids, exact_match_preds)),
        'cooccurrence': dict(zip(all_ids, co_occurrence_preds)),
        'specific': dict(zip(all_ids, specific_word_preds)),
        'overall': dict(zip(all_ids, predicted_labels)),
    }

    def evaluate_subset(method_key, ids):
        y_true = [label_dict[i] for i in ids]
        y_pred = [pred_dict[method_key][i] for i in ids]
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        return Bunch(
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm.tolist(),
            y_true=y_true,
            y_pred=y_pred,
        )
    subset_results = {
        'Exact Matching': evaluate_subset('exact', all_ids),
        'Co-occurrence Matching': evaluate_subset('cooccurrence', [i for i in all_ids if i not in method_applied['exact']]),
        'Specific Word Matching': evaluate_subset(
            'specific',
            [
                i
                 for i in all_ids if i not in method_applied['exact'] and i not in method_applied['cooccurrence']
            ],
        ),
        'Overall': evaluate_subset('overall', all_ids),
    }
    for method, result in subset_results.items():
        cm = result.confusion_matrix
        conf_matrix_path = os.path.join(output_dir, f"{method.replace(' ', '_').lower()}_confusion_matrix.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Ref', 'Ref'],
            yticklabels=['No Ref', 'Ref'],
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
    baseline_pickle_path = os.path.join(str(paths.DOCUMENT_DATA), 'baseline_predictions.pkl')
    with open(baseline_pickle_path, 'wb') as f:
        pickle.dump(baseline_predictions, f)
    classification_report_path = os.path.join(output_dir, 'baseline_classification_report.txt')
    with open(classification_report_path, 'w') as f:
        for method, metrics in subset_results.items():
            f.write(f'Results for {method}:\n')
            f.write(f'  Precision: {metrics.precision:.4f}\n')
            f.write(f'  Recall: {metrics.recall:.4f}\n')
            f.write(f'  F1-Score: {metrics.f1:.4f}\n')
            f.write(f'  Confusion Matrix: {metrics.confusion_matrix}\n\n')
    waterfall_plot_path = os.path.join(output_dir, 'hierarchical_detection.png')
    exact_match_count = len(method_applied['exact'])
    co_occurrence_count = len(method_applied['cooccurrence'])
    specific_word_count = len(method_applied['specific'])
    values = [exact_match_count, co_occurrence_count, specific_word_count]
    cumulative_values = np.cumsum(values).tolist()
    cumulative_values.insert(0, 0)
    methods = ['Title String Match', 'Co-occurrence', 'Specific Words', 'Total']
    total_value = sum(values)
    colors = ['#6baed6', '#4292c6', '#2171b5']
    plt.figure(figsize=(8, 5))
    for i in range(len(values)):
        plt.bar(methods[i], values[i], bottom=cumulative_values[i], color=colors[i])
    plt.bar(methods[3], total_value, color='#084594', edgecolor='black', linestyle='--')
    for i, val in enumerate(values):
        plt.text(
            i,
            cumulative_values[i] + val / 2,
            str(val),
            ha='center',
            va='center',
            fontsize=12,
            color='white',
        )
    plt.text(
        3,
        total_value / 2,
        str(total_value),
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        color='white',
    )
    plt.xlabel('Detection Method', fontsize=12)
    plt.ylabel('Cumulative Number of References Detected', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    waterfall_plot_path = os.path.join(output_dir, 'hierarchical_detection.png')
    plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    for method, metrics in subset_results.items():
        print(f'\n **Results for {method}:**')
        print(f'  Precision: {metrics.precision:.4f}')
        print(f'  Recall: {metrics.recall:.4f}')
        print(f'  F1-Score: {metrics.f1:.4f}')
        print(f'  Confusion Matrix: {metrics.confusion_matrix}')
    print(f'\n Classification report saved: {classification_report_path}')
    print(f' Confusion matrix image saved: {conf_matrix_path}')
    print(f' Hierarchical detection plot saved: {waterfall_plot_path}')

    def save_false_positives(output_path, data, predictions, labels):
        false_positives = [i for i in data['id'] if labels[i] == 0 and predictions[i] == 1]
        if false_positives:
            with open(output_path, 'w') as f:
                f.write('False Positives (Predicted as Reference, but actually No Reference):\n\n')
                for letter_id, text in zip(data['id'], data['text']):
                    if letter_id not in false_positives:
                        continue
                    text_tokens = tokenize(text)
                    hits_exact = [title for title in reference_titles if exact_title_match(text, title)]
                    hits_cooc = [
                        title
                         for title in reference_titles if co_occurrence_search(text_tokens, title)
                    ]
                    hits_specific = [
                        word
                         for word in specific_words if re.search(f'\\b{re.escape(word)}\\b', text, re.IGNORECASE)
                    ]
                    f.write(f'--- Letter ID: {letter_id} ---\n')
                    if hits_exact:
                        f.write(f"Exact Match: {', '.join(hits_exact)}\n")
                    if hits_cooc:
                        f.write(f"Co-occurrence Match: {', '.join(hits_cooc)}\n")
                    if hits_specific:
                        f.write(f"Specific Word Match: {', '.join(hits_specific)}\n")
                    f.write('\n--- Letter Text ---\n')
                    f.write(text.strip() + '\n\n')
                    f.write('=' * 80 + '\n\n')
            print(f' Detailed false positives with triggers saved to: {output_path}')
        else:
            print(' No false positives found.')
    fp_output_path = os.path.join(output_dir, 'false_positives.txt')
    save_false_positives(fp_output_path, processed_data, baseline_predictions, label_dict)
if __name__ == '__main__':
    run()
