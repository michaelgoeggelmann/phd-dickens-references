"""BERT evaluation on the same sentence IDs as the prompt experiment."""

from shared import paths
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def run():
    BERT_FILES = {
        0: Path(str(paths.sentence_run(0) / 'general/all_folds_predictions.csv')),
        4: Path(str(paths.sentence_run(4) / 'general/all_folds_predictions.csv')),
    }
    LLAMA_SAMPLE = Path(str(paths.PROMPT_RESULTS / 'sentence/nocontext/llm_results_sentence_multilabel_L1.csv'))
    KEYS = ['letter_id', 'sentence_id']
    CODES = list('ABCDEFGHI')
    BERT_INDEX_TO_CODE = list('DCFGIAHEB')
    CATEGORY_NAMES = {
        'A': 'Inspirations and Influences',
        'B': 'Writing Process and Creative Insight',
        'C': 'Elements of Fictional Craft',
        'D': 'Editorial and Publishing Processes',
        'E': 'Quality and Evaluation',
        'F': 'Financial and Legal Aspects',
        'G': 'Gratitude and Acknowledgement',
        'H': 'Marketing and Promotion',
        'I': 'Impact and Reception',
    }

    def read_csv(path, required_columns):
        if not path.is_file():
            raise FileNotFoundError(f'File not found: {path}')
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f'Missing columns in {path.name}: {sorted(missing)}\nAvailable columns: {list(df.columns)}',
            )
        return df

    def normalise_keys(df, source):
        df = df.copy()
        df['letter_id'] = df['letter_id'].astype('string').str.strip()
        numeric_id = pd.to_numeric(df['sentence_id'], errors='raise')
        if numeric_id.isna().any() or (numeric_id % 1 != 0).any() or df['letter_id'].isna().any():
            raise ValueError(f'Invalid sentence IDs in {source}')
        df['sentence_id'] = numeric_id.astype('int64')
        duplicated = df.duplicated(KEYS, keep=False)
        if duplicated.any():
            raise ValueError(f'{source}: {duplicated.sum()} duplicate (letter_id, sentence_id) rows')
        return df

    def parse_bert_vector(raw):
        if isinstance(raw, (list, tuple, np.ndarray)):
            vector = np.asarray(raw, dtype=float)
        else:
            s = str(raw).strip()
            try:
                vector = np.asarray(ast.literal_eval(s), dtype=float)
            except (SyntaxError, ValueError):
                vector = np.fromstring(s.strip('[]').replace(',', ' '), sep=' ')
        if vector.shape != (9,) or not np.isfinite(vector).all():
            raise ValueError(f'Invalid BERT vector: {raw!r}')
        if np.any((vector < 0) | (vector > 1)):
            raise ValueError(f'BERT vector outside [0,1]: {raw!r}')
        return {BERT_INDEX_TO_CODE[i] for i in range(9) if vector[i] >= 0.5}

    def parse_llama_gold(raw):
        if pd.isna(raw):
            raise ValueError('Missing Llama gold label')
        s = str(raw).strip()
        try:
            val = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            val = s.strip('[]').replace('"', '').replace("'", '').split(',')
        if isinstance(val, str):
            val = [val]
        if not isinstance(val, (list, tuple, set)):
            raise ValueError(f'Invalid Llama gold label: {raw!r}')
        labels = {str(x).strip().upper() for x in val if str(x).strip()}
        unknown = labels - set(CODES) - {'J'}
        if unknown:
            raise ValueError(f'Unknown Llama labels: {unknown}')
        return labels & set(CODES)

    def to_matrix(series):
        return np.asarray([[int(code in labels) for code in CODES] for labels in series], dtype=int)

    def evaluate(df, name):
        y_true = to_matrix(df['gold'])
        y_pred = to_matrix(df['pred'])
        binary_true = y_true.any(axis=1).astype(int)
        binary_pred = y_pred.any(axis=1).astype(int)
        print('\n' + '=' * 75)
        print(name)
        print('=' * 75)
        print(f'Total sentences: {len(df):,}')
        print(f'Gold Ref:        {binary_true.sum():,}')
        print(f'Gold NoRef:      {(binary_true == 0).sum():,}')
        print('\n--- BINARY: Ref / NoRef ---')
        print(
            classification_report(
                binary_true,
                binary_pred,
                labels=[1, 0],
                target_names=['Ref', 'NoRef'],
                digits=4,
                zero_division=0,
            ),
        )
        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()
        print(f'TP={tp} | FN={fn} | TN={tn} | FP={fp}')
        print('\n--- MULTILABEL: Categories A-I ---')
        for average in ['macro', 'weighted', 'micro']:
            score = f1_score(y_true, y_pred, average=average, zero_division=0)
            print(f'{average.capitalize():8s} F1: {score:.4f}')
        p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        rows = []
        for i, code in enumerate(CODES):
            rows.append(
                {
                    'Category': code,
                    'Name': CATEGORY_NAMES[code],
                    'P': p[i],
                    'R': r[i],
                    'F1': f1[i],
                    'n': int(support[i]),
                },
            )
        category_table = pd.DataFrame(rows)
        print(
            '\n' + category_table.to_string(
                index=False,
                formatters={'P': '{:.4f}'.format, 'R': '{:.4f}'.format, 'F1': '{:.4f}'.format},
            ),
        )
        return {
            'Setup': name,
            'N': len(df),
            'Ref F1': f1_score(binary_true, binary_pred, pos_label=1, zero_division=0),
            'NoRef F1': f1_score(binary_true, binary_pred, pos_label=0, zero_division=0),
            'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
    llama = read_csv(LLAMA_SAMPLE, KEYS + ['gold_categories'])
    llama = normalise_keys(llama, 'Llama')
    if len(llama) != 7900:
        raise ValueError(f'Expected 7,900 Llama sentences, found {len(llama)}')
    llama['llama_gold'] = llama['gold_categories'].apply(parse_llama_gold)
    llama = llama[KEYS + ['llama_gold']].copy()
    print(f'Llama evaluation subset loaded: {len(llama):,} unique sentences')
    results = []
    for n, path in BERT_FILES.items():
        bert = read_csv(path, KEYS + ['true_labels', 'pred_labels'])
        bert = normalise_keys(bert, f'BERT n={n}')
        bert['gold'] = bert['true_labels'].apply(parse_bert_vector)
        bert['pred'] = bert['pred_labels'].apply(parse_bert_vector)
        bert = bert[KEYS + ['gold', 'pred']].copy()
        results.append(evaluate(bert, f'BERT n={n} | FULL DATASET'))
        matched = llama.merge(bert, on=KEYS, how='left', validate='one_to_one', indicator=True)
        missing = matched['_merge'].ne('both').sum()
        print(f'\nMatched BERT n={n}: {len(matched) - missing:,} / 7,900')
        if missing:
            raise ValueError(f'BERT n={n}: {missing} Llama sentences could not be matched.')
        mismatches = matched.apply(lambda row: row['llama_gold'] != row['gold'], axis=1)
        mismatch_count = int(mismatches.sum())
        print(f'Gold-label mismatches: {mismatch_count}')
        if mismatch_count > 0:
            print('\nExamples of mismatched gold labels:')
            print(
                matched.loc[mismatches, KEYS + ['llama_gold', 'gold']].head(10).to_string(
                    index=False,
                ),
            )
            raise ValueError(
                f'BERT n={n}: {mismatch_count} gold-label mismatches between BERT and Llama.',
            )
        results.append(evaluate(matched, f'BERT n={n} | SAME 7,900 LLAMA SENTENCES'))
        binary_gold = np.array([[int(code in labels) for code in CODES] for labels in matched['gold']])
        binary_pred = np.array([[int(code in labels) for code in CODES] for labels in matched['pred']])
        report = classification_report(
            binary_gold,
            binary_pred,
            target_names=[CATEGORY_NAMES[c] for c in CODES],
            zero_division=0,
            output_dict=True,
        )
        score_rows = [
            {
                'Category': CATEGORY_NAMES[c],
                'Context': n,
                'F1': report[CATEGORY_NAMES[c]]['f1-score'],
            }
             for c in CODES
        ]
        (paths.PROMPT_RESULTS / 'summaries').mkdir(parents=True, exist_ok=True)
        pd.DataFrame(score_rows).to_csv(
            paths.PROMPT_RESULTS / 'summaries' / f'bert_context_{n}.csv',
            index=False,
        )
    print('\n' + '=' * 75)
    print('FINAL BERT COMPARISON')
    print('=' * 75)
    summary = pd.DataFrame(results)
    print(
        summary.to_string(
            index=False,
            formatters={
                'Ref F1': '{:.4f}'.format,
                'NoRef F1': '{:.4f}'.format,
                'Macro F1': '{:.4f}'.format,
                'Weighted F1': '{:.4f}'.format,
            },
        ),
    )
    print('\nEvaluation completed successfully.')
if __name__ == '__main__':
    run()
