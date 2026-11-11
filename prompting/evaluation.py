"""Scoring of document and sentence prompt outputs."""

from shared import paths
import os
import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

BASE_DIR = str(paths.PROMPT_RESULTS)
CATEGORY_MAP = {
    'A': 'Inspirations and Influences',
    'B': 'Writing Process and Creative Insight',
    'C': 'Elements of Fictional Craft',
    'D': 'Editorial and Publishing Processes',
    'E': 'Quality and Evaluation',
    'F': 'Financial and Legal Aspects',
    'G': 'Gratitude and Acknowledgement',
    'H': 'Marketing and Promotion',
    'I': 'Impact and Reception',
    'J': 'No reference to Dickens’s works present',
}
LABELS_MAIN = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
from prompting.analysis_utils import read_file


def prompt_level(filename):
    import re
    m = re.search('L(\\d+)', filename)
    return f'L{m.group(1)}' if m else 'unknown'


def reorder_metrics(df):
    order = ['P', 'R', 'F1', 'n']
    existing = [o for o in order if o in df.columns.get_level_values(1)]
    return df.reindex(existing, axis=1, level=1)


def force_int_n(df):
    n_cols = [col for col in df.columns if col[1] == 'n']
    for col in n_cols:
        df[col] = df[col].fillna(0).astype(int)
    return df


def normalise_label_list(labels):
    if labels is None:
        return ['J']
    if isinstance(labels, tuple):
        labels = list(labels)
    if not isinstance(labels, list):
        return ['J']
    if len(labels) == 0:
        return ['J']
    seen = set()
    out = []
    for x in labels:
        if x is None:
            continue
        x = str(x).strip()
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out if out else ['J']


def to_binary_ref(labels):
    labs = set(labels)
    return 'Ref' if any((l in labs for l in LABELS_MAIN)) else 'NoRef'


def evaluate_document(doc_dir):
    rows = []
    for f in sorted(os.listdir(doc_dir)):
        if not f.endswith('.csv'):
            continue
        df = read_file(os.path.join(doc_dir, f))
        if 'label' not in df or 'gold_label' not in df:
            continue
        y_true = df['gold_label'].astype(str).str.strip()
        y_pred = df['label'].astype(str).str.strip()
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        L = prompt_level(f)
        for label in sorted(set(y_true) | set(y_pred)):
            vals = report[label]
            rows.append(
                {
                    'Prompt': L,
                    'Class': label,
                    'P': round(vals['precision'], 2),
                    'R': round(vals['recall'], 2),
                    'F1': round(vals['f1-score'], 2),
                    'n': int(vals['support']),
                },
            )
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print(' No document results found.')
        return pd.DataFrame()
    pivot = df_out.pivot(index='Prompt', columns='Class', values=['P', 'R', 'F1', 'n'])
    pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    pivot = reorder_metrics(pivot)
    pivot = force_int_n(pivot).round(2)
    print('\n=== Document-level summary ===')
    print(pivot)
    return pivot


def evaluate_multilabel_binary(sent_dir):
    context_results = {}
    for context_type in ['context_4', 'nocontext']:
        ctx_path = os.path.join(sent_dir, context_type)
        if not os.path.exists(ctx_path):
            continue
        rows = []
        for f in sorted(os.listdir(ctx_path)):
            if not f.endswith('.csv'):
                continue
            df = read_file(os.path.join(ctx_path, f))
            if 'categories' not in df or 'gold_categories' not in df:
                continue
            df['pred'] = df['categories'].apply(lambda x: normalise_label_list(ast.literal_eval(x)))
            df['true'] = df['gold_categories'].apply(lambda x: normalise_label_list(ast.literal_eval(x)))
            y_true = [to_binary_ref(labs) for labs in df['true']]
            y_pred = [to_binary_ref(labs) for labs in df['pred']]
            report = classification_report(
                y_true,
                y_pred,
                target_names=['NoRef', 'Ref'],
                zero_division=0,
                output_dict=True,
            )
            L = prompt_level(f)
            for label in ['NoRef', 'Ref']:
                vals = report[label]
                rows.append(
                    {
                        'Prompt': L,
                        'Class': label,
                        'P': round(vals['precision'], 2),
                        'R': round(vals['recall'], 2),
                        'F1': round(vals['f1-score'], 2),
                        'n': int(vals['support']),
                    },
                )
        df_bin = pd.DataFrame(rows)
        if not df_bin.empty:
            pivot = df_bin.pivot(index='Prompt', columns='Class', values=['P', 'R', 'F1', 'n'])
            pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
            pivot = reorder_metrics(pivot)
            pivot = force_int_n(pivot).round(2)
            context_results[context_type] = pivot
    print('\n=== Multilabel → Binary (Ref/NoRef) summaries ===')
    for ctx, tab in context_results.items():
        print(f'\n {ctx}')
        print(tab)
    return context_results


def evaluate_sentence_categories(sent_dir):
    results = []
    for context_type in ['context_4', 'nocontext']:
        context_path = os.path.join(sent_dir, context_type)
        if not os.path.exists(context_path):
            continue
        for f in sorted(os.listdir(context_path)):
            if not f.endswith('.csv'):
                continue
            df = read_file(os.path.join(context_path, f))
            if 'categories' not in df or 'gold_categories' not in df:
                continue
            df['pred_raw'] = df['categories'].apply(lambda x: normalise_label_list(ast.literal_eval(x)))
            df['true_raw'] = df['gold_categories'].apply(lambda x: normalise_label_list(ast.literal_eval(x)))
            df['pred'] = df['pred_raw'].apply(lambda labs: [l for l in labs if l in LABELS_MAIN])
            df['true'] = df['true_raw'].apply(lambda labs: [l for l in labs if l in LABELS_MAIN])
            mlb = MultiLabelBinarizer(classes=LABELS_MAIN)
            y_true = mlb.fit_transform(df['true'])
            y_pred = mlb.transform(df['pred'])
            report = classification_report(
                y_true,
                y_pred,
                target_names=mlb.classes_,
                zero_division=0,
                output_dict=True,
            )
            L = prompt_level(f)
            for cat in mlb.classes_:
                vals = report[cat]
                results.append(
                    {
                        'Prompt': L,
                        'Context': context_type,
                        'Category': CATEGORY_MAP.get(cat, cat),
                        'P': round(vals['precision'], 2),
                        'R': round(vals['recall'], 2),
                        'F1': round(vals['f1-score'], 2),
                        'n': int(vals['support']),
                    },
                )
    if not results:
        print(' No multilabel category results found.')
        return ({}, pd.DataFrame())
    df_res = pd.DataFrame(results)
    (paths.PROMPT_RESULTS / 'summaries').mkdir(parents=True, exist_ok=True)
    df_res.to_csv(paths.PROMPT_RESULTS / 'summaries' / 'category_scores.csv', index=False)
    tables = {}
    for cat, group in df_res.groupby('Category'):
        pivot = group.pivot(index='Prompt', columns='Context', values=['P', 'R', 'F1', 'n'])
        pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        pivot = reorder_metrics(pivot)
        pivot = force_int_n(pivot).round(2)
        tables[cat] = pivot
    summary_rows = []
    for ctx in ['context_4', 'nocontext']:
        ctx_data = df_res[df_res['Context'] == ctx]
        if ctx_data.empty:
            continue
        for prompt, sub in ctx_data.groupby('Prompt'):
            macro = sub[['P', 'R', 'F1']].mean()
            n_total = sub['n'].sum()
            if n_total == 0:
                weighted_p = weighted_r = weighted_f1 = 0.0
            else:
                weighted = (sub[['P', 'R', 'F1']].T * sub['n']).T.sum() / n_total
                weighted_p = float(weighted['P'])
                weighted_r = float(weighted['R'])
                weighted_f1 = float(weighted['F1'])
            summary_rows.append(
                {
                    'Context': ctx,
                    'Prompt': prompt,
                    'Macro_P': round(float(macro['P']), 2),
                    'Macro_R': round(float(macro['R']), 2),
                    'Macro_F1': round(float(macro['F1']), 2),
                    'Weighted_P': round(weighted_p, 2),
                    'Weighted_R': round(weighted_r, 2),
                    'Weighted_F1': round(weighted_f1, 2),
                    'Total_n': int(n_total),
                },
            )
    summary_df = pd.DataFrame(summary_rows)
    print('\n=== Sentence-level summaries by category (A–I) ===')
    for cat, tab in tables.items():
        print(f'\n {cat}')
        print(tab)
    macro_df = summary_df[['Context', 'Prompt', 'Macro_P', 'Macro_R', 'Macro_F1', 'Total_n']].copy()
    weighted_df = summary_df[['Context', 'Prompt', 'Weighted_P', 'Weighted_R', 'Weighted_F1', 'Total_n']].copy()
    print('\n=== Macro Averages per Prompt (across A–I) ===')
    print(macro_df)
    print('\n=== Weighted Averages per Prompt (across A–I) ===')
    print(weighted_df)
    return (tables, summary_df)
