"""Plots of computed token attributions."""

from pathlib import Path
from html import escape
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_split_shap_for_letter(csv_path, text_id, output_path, n=5, show_text=False):
    """Plot the strongest positive and negative token contributions for one letter."""
    df = pd.read_csv(csv_path)
    df = df[df['text_id'] == text_id].copy()
    if df.empty:
        print(f'No attributions found for {text_id}.')
        return
    df = df[df['shap_Ref'].apply(lambda x: round(x, 2) != 0.0)]
    top_pos = df[df['shap_Ref'] > 0].sort_values('shap_Ref', ascending=False).head(n)
    top_neg = df[df['shap_Ref'] < 0].sort_values('shap_Ref', ascending=True).head(n)
    tokens = list(top_pos['token'].to_list()) + list(top_neg['token'].to_list()[::-1])
    pos_vals = list(top_pos['shap_Ref'].to_list()) + [0] * len(top_neg)
    neg_vals = [0] * len(top_pos) + list(top_neg['shap_Ref'].to_list()[::-1])
    if not show_text:
        tokens = [f'[TOKEN {i + 1}]' for i in range(len(tokens))]
    y_pos = range(len(tokens))
    plt.figure(figsize=(7, 0.6 * len(tokens)))
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(tokens)))
    ax.barh(y_pos, neg_vals, color='blue', alpha=0.7, label='negative')
    ax.barh(y_pos, pos_vals, color='red', alpha=0.7, label='positive')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP-Value')
    for i, (pos, neg) in enumerate(zip(pos_vals, neg_vals)):
        if pos != 0:
            ax.text(pos + 0.01, i, f'{pos:+.2f}', ha='left', va='center', fontsize=9)
        if neg != 0:
            ax.text(neg - 0.01, i, f'{neg:+.2f}', ha='right', va='center', fontsize=9)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'Saved: {output_path}')


def render_token_attributions(csv_path, text_id, output_path, show_text=False):
    """Render saved token scores as an HTML strip, with text hidden by default."""
    frame = pd.read_csv(csv_path)
    frame = frame.loc[frame['text_id'] == text_id]
    if frame.empty:
        raise ValueError(f'No token attributions found for {text_id}.')
    parts = [
        '<html><head><meta charset="utf-8"><style>',
        '.token-box {display:inline-block;margin:2px;padding:4px 6px;border-radius:4px;font-family:sans-serif;font-size:14px;color:white;text-align:center;} .score {display:block;font-size:10px;margin-top:2px;min-height:12px;color:white;}',
        '</style></head><body><div>',
    ]
    for index, row in enumerate(frame.itertuples(index=False)):
        value = float(row.shap_Ref)
        norm_value = max(min(value / 0.04, 1), -1)
        red = int(255 * norm_value) if norm_value > 0 else 0
        blue = int(255 * abs(norm_value)) if norm_value < 0 else 0
        token = escape(str(row.token)) if show_text else f'[TOKEN {index + 1}]'
        parts.append(
            f'<span class="token-box" style="background-color:rgb({red},0,{blue})">{token}<span class="score">{value:+.4f}</span></span> ',
        )
    parts.append('</div></body></html>')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(parts), encoding='utf-8')
    return output_path
