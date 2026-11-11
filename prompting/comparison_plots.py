"""Prompt-level and model-comparison figures from evaluated predictions."""

from shared import paths
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REFERENCE_CATEGORIES_9 = [
    'Inspirations & Influences',
    'Writing Process & Creative Insight',
    'Elements of Fictional Craft',
    'Editorial & Publishing Processes',
    'Quality & Evaluation',
    'Financial & Legal Aspects',
    'Gratitude & Acknowledgement',
    'Marketing & Promotion',
    'Impact & Reception',
]
PROMPT_LEVELS = ['L1', 'L2', 'L3', 'L4', 'L5']
CONTEXT_SIZES = [0, 4]
OUT_DIR = paths.FIGURES_DIR / 'prompt_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)
COLOR_CTX0 = '#B3B3B3'
COLOR_CTX4 = '#4D4D4D'
BERT_LINE_COLOR = 'black'


def load_performance_data() -> pd.DataFrame:
    cat_map = {
        'Editorial and Publishing Processes': 'Editorial & Publishing Processes',
        'Elements of Fictional Craft': 'Elements of Fictional Craft',
        'Financial and Legal Aspects': 'Financial & Legal Aspects',
        'Gratitude and Acknowledgement': 'Gratitude & Acknowledgement',
        'Impact and Reception': 'Impact & Reception',
        'Inspirations and Influences': 'Inspirations & Influences',
        'Marketing and Promotion': 'Marketing & Promotion',
        'Quality and Evaluation': 'Quality & Evaluation',
        'Writing Process and Creative Insight': 'Writing Process & Creative Insight',
    }
    scores = pd.read_csv(paths.PROMPT_RESULTS / 'summaries' / 'category_scores.csv')
    llama_data = {}
    for category, group in scores.groupby('Category'):
        llama_data[category] = {(0 if row.Context == 'nocontext' else 4, row.Prompt): row.F1 for row in group.itertuples()}
    bert_f1_n0 = pd.read_csv(paths.PROMPT_RESULTS / 'summaries' / 'bert_context_0.csv').set_index('Category')['F1'].to_dict()
    bert_f1_n4 = pd.read_csv(paths.PROMPT_RESULTS / 'summaries' / 'bert_context_4.csv').set_index('Category')['F1'].to_dict()
    rows = []
    for src_cat, values in llama_data.items():
        cat = cat_map[src_cat]
        for context_n in CONTEXT_SIZES:
            for p in PROMPT_LEVELS:
                rows.append(
                    dict(
                        system=f'Llama {p}',
                        model='llama',
                        prompt=p,
                        context_n=context_n,
                        category=cat,
                        f1=float(values[context_n, p]),
                    ),
                )
    for src_cat, internal_cat in cat_map.items():
        rows.append(
            dict(
                system='BERT',
                model='bert',
                prompt=None,
                context_n=0,
                category=internal_cat,
                f1=float(bert_f1_n0[src_cat]),
            ),
        )
        rows.append(
            dict(
                system='BERT',
                model='bert',
                prompt=None,
                context_n=4,
                category=internal_cat,
                f1=float(bert_f1_n4[src_cat]),
            ),
        )
    df = pd.DataFrame(rows)
    df['category'] = pd.Categorical(df['category'], categories=REFERENCE_CATEGORIES_9, ordered=True)
    df['prompt'] = pd.Categorical(df['prompt'], categories=PROMPT_LEVELS, ordered=True)
    df['context_n'] = pd.Categorical(df['context_n'], categories=CONTEXT_SIZES, ordered=True)
    return df


def savefig(fig: plt.Figure, filename: str) -> None:
    out = OUT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches='tight')


def _dedupe_legend(handles, labels):
    seen = set()
    new_h, new_l = ([], [])
    for h, l in zip(handles, labels):
        if l not in seen:
            new_h.append(h)
            new_l.append(l)
            seen.add(l)
    return (new_h, new_l)


def plot_prompt_dynamics_bars(df: pd.DataFrame, ncols: int=3) -> None:
    assert ncols in (2, 3), 'Use ncols=2 or ncols=3.'
    n_panels = len(REFERENCE_CATEGORIES_9)
    nrows = int(np.ceil(n_panels / ncols))
    if ncols == 2:
        fig_w, fig_h = (12, 14)
        out_name = 'prompt_category_dynamics_two_columns.png'
        legend_ncol = 2
        tight_rect = (0, 0.08, 1, 1)
        legend_anchor_y = 0.02
    else:
        fig_w, fig_h = (16, 10)
        out_name = 'prompt_category_dynamics.png'
        legend_ncol = 2
        tight_rect = (0, 0.1, 1, 1)
        legend_anchor_y = 0.02
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)
    x = np.arange(len(PROMPT_LEVELS))
    bar_w = 0.38
    bert0 = df[(df['system'] == 'BERT') & (df['context_n'] == 0)].set_index('category')['f1'].reindex(
        REFERENCE_CATEGORIES_9,
    )
    bert4 = df[(df['system'] == 'BERT') & (df['context_n'] == 4)].set_index('category')['f1'].reindex(
        REFERENCE_CATEGORIES_9,
    )
    for ax, cat in zip(axes, REFERENCE_CATEGORIES_9):
        llama_n0, llama_n4 = ([], [])
        for p in PROMPT_LEVELS:
            v0 = df[(df['system'] == f'Llama {p}') & (df['context_n'] == 0) & (df['category'] == cat)]['f1'].mean()
            v4 = df[(df['system'] == f'Llama {p}') & (df['context_n'] == 4) & (df['category'] == cat)]['f1'].mean()
            llama_n0.append(v0)
            llama_n4.append(v4)
        ax.bar(
            x - bar_w / 2,
            llama_n0,
            width=bar_w,
            color=COLOR_CTX0,
            edgecolor='black',
            label='Llama (context window n=0)',
        )
        ax.bar(
            x + bar_w / 2,
            llama_n4,
            width=bar_w,
            color=COLOR_CTX4,
            edgecolor='black',
            label='Llama (context window n=4)',
        )
        if np.isfinite(bert0.loc[cat]):
            ax.axhline(
                float(bert0.loc[cat]),
                linewidth=1,
                linestyle=':',
                color=BERT_LINE_COLOR,
                label='BERT (context window n=0)',
            )
        if np.isfinite(bert4.loc[cat]):
            ax.axhline(
                float(bert4.loc[cat]),
                linewidth=1,
                linestyle='--',
                color=BERT_LINE_COLOR,
                label='BERT (context window n=4)',
            )
        ax.set_title(cat)
        ax.set_xticks(x)
        ax.set_xticklabels(PROMPT_LEVELS)
        ax.set_xlabel('Prompt level')
        ax.set_ylim(0, 1)
        ax.set_ylabel('F1')
    if ncols == 2:
        legend_ax = axes[len(REFERENCE_CATEGORIES_9)]
        legend_ax.axis('off')
    else:
        for i in range(len(REFERENCE_CATEGORIES_9), len(axes)):
            axes[i].axis('off')
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = _dedupe_legend(handles, labels)
    if ncols == 2:
        legend_ax.legend(handles, labels, loc='center', ncol=1, frameon=True)
    else:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=2,
            bbox_to_anchor=(0.5, 0.02),
            borderaxespad=0.0,
        )
    savefig(fig, out_name)
    plt.show()
    plt.close(fig)


def main() -> None:
    df = load_performance_data()
    plot_prompt_dynamics_bars(df, ncols=3)
    plot_prompt_dynamics_bars(df, ncols=2)
    print(f'Saved plots to: {OUT_DIR.resolve()}')
if __name__ == '__main__':
    main()
