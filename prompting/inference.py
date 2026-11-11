"""Schema-constrained prompt inference and label-token probabilities."""

from shared import paths
from shared.labels import CATEGORY_INDEX
import os, pickle, json, random, re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import defaultdict, Counter
import numpy as np
import csv
import time
from pathlib import Path
from pprint import pprint
from typing import Optional, Dict, Any, List, Union
import math
from llama_cpp import Llama


def run(experiment='sentence_multilabel', context_window=4, prompt_level='L4', limit=0):
    """Run one prompt configuration, resuming an existing result file when present."""
    EXPERIMENT = experiment
    CONTEXT_WINDOW = context_window
    PROMPT_LEVEL = prompt_level
    LIMIT = limit
    OUTPUT_MODE = 'JSON'
    paths.prepare_directories()
    BASE_DIR = paths.ROOT
    DATA_DIR = str(paths.PROMPT_DATA)
    PROMPTS_DIR = str(paths.PROMPTS_DIR)
    RESULTS_DIR = str(
        paths.PROMPT_RESULTS / ('document' if EXPERIMENT == 'document' else 'sentence/nocontext' if CONTEXT_WINDOW == 0 else 'sentence/context_4'),
    )
    LOGS_DIR = str(paths.PROMPT_RESULTS / 'logs')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    MODEL_PATH = str(paths.gguf_model())
    llm = Llama(
        model_path=MODEL_PATH,
        seed=42,
        n_ctx=8192,
        n_threads=8,
        n_gpu_layers=-1,
        logits_all=True,
        verbose=False,
    )

    def stratified_sample_document(data, limit):
        pos = [d for d in data if d['label'] == 1]
        neg = [d for d in data if d['label'] == 0]
        n_each = min(len(pos), len(neg), limit // 2)
        sampled = random.sample(pos, n_each) + random.sample(neg, n_each)
        random.shuffle(sampled)
        counts = Counter([d['label'] for d in sampled])
        print('\n Document-level distribution:')
        print(f'  Total: {len(sampled)}')
        print(f'  Ref (1): {counts[1]}')
        print(f'  NoRef (0): {counts[0]}')
        print(f'  Ratio: {counts[1] / len(sampled):.2f} Ref / {counts[0] / len(sampled):.2f} NoRef')
        return sampled
    NUM_CATEGORIES = len(CATEGORY_INDEX)

    def stratified_sample_sentences(data, limit):
        label_counts = defaultdict(list)
        for s in data:
            cats = [i for i, v in enumerate(s['category_vector']) if v == 1]
            if len(cats) == 0:
                label_counts['NoRef'].append(s)
            else:
                for c in cats:
                    label_counts[c].append(s)
        total = sum((len(v) for v in label_counts.values()))
        sampled = []
        for k, samples in label_counts.items():
            share = max(3, int(limit * (len(samples) / total)))
            share = min(share, len(samples))
            sampled.extend(random.sample(samples, share))
        if len(sampled) > limit:
            sampled = random.sample(sampled, limit)
        random.shuffle(sampled)
        print('\n Sentence-level distribution:')
        print(f'  Total: {len(sampled)}')
        cat_counter = Counter()
        for s in sampled:
            cats = [i for i, v in enumerate(s['category_vector']) if v == 1]
            if not cats:
                cat_counter['NoRef'] += 1
            else:
                for c in cats:
                    cat_counter[c] += 1
        keys_sorted = ['NoRef'] + [i for i in range(NUM_CATEGORIES)]
        for k in keys_sorted:
            if k in cat_counter:
                label = 'NoRef' if k == 'NoRef' else f'{k} – {list(CATEGORY_INDEX.keys())[k]}'
                print(f'  {label:<45}: {cat_counter[k]}')
        return sampled
    SYSTEM_PROMPT = 'You are a specialist in nineteenth-century literature who analyses Charles Dickens’s letters for self-referential statements concerning his own works and authorship.'

    def _messages_for(prompt: str):
        return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]

    def get_doc_schema(prompt_level: str):
        base = {
            'type': 'object',
            'properties': {'label': {'type': 'string', 'enum': ['YES', 'NO']}},
            'required': ['label'],
            'additionalProperties': False,
        }
        if prompt_level == 'L5':
            base['properties']['rationale'] = {'type': 'string', 'maxLength': 400}
            base['properties']['evidence'] = {'type': 'string', 'maxLength': 200}
        return base

    def get_sent_schema(prompt_level: str):
        base = {
            'type': 'object',
            'properties': {
                'categories': {
                    'type': 'array',
                    'items': {'type': 'string', 'enum': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']},
                    'uniqueItems': True,
                },
            },
            'required': ['categories'],
            'additionalProperties': False,
        }
        if prompt_level == 'L5':
            base['properties']['rationale'] = {'type': 'string', 'maxLength': 400}
            base['properties']['evidence'] = {'type': 'string', 'maxLength': 200}
        return base

    def generate_one(
        prompt: str,
        max_tokens: int=None,
        max_new_tokens: int=256,
        mode: str='text',
        temperature: float=0.0,
        return_logprobs: bool=False,
        top_k_alternatives: int=6,
        experiment: str='document',
    ) -> Union[str, Dict[str, Any]]:
        import math, json, re
        if max_tokens is None:
            max_tokens = max_new_tokens
        messages = _messages_for(prompt)
        common_kwargs = dict(
            messages=messages,
            temperature=temperature,
            top_p=1.0,
            top_k=1,
            repeat_penalty=1.0,
            max_tokens=max_tokens,
        )
        if mode in {'doc_json', 'sent_json'}:
            schema_fn = get_doc_schema if mode == 'doc_json' else get_sent_schema
            schema = schema_fn(PROMPT_LEVEL)
            response_format = {'type': 'json_object', 'schema': schema}
        else:
            response_format = None
        logprob_kwargs = {}
        if return_logprobs:
            logprob_kwargs = dict(
                logprobs=True,
                top_logprobs=int(top_k_alternatives) if top_k_alternatives and top_k_alternatives > 0 else 0,
            )
        response = llm.create_chat_completion(
            **common_kwargs,
            **{'response_format': response_format} if response_format else {},
            **logprob_kwargs,
        )
        choice = response['choices'][0]
        text = choice['message']['content'].strip()
        if mode.endswith('_json'):
            if text.endswith('{') or text.endswith(','):
                text = text.rstrip(',{') + '}'
            try:
                json.loads(text)
            except json.JSONDecodeError:
                m = re.search('```(?:json)?\\s*(\\{[\\s\\S]*?\\})\\s*```', text)
                if m:
                    inner = m.group(1).strip()
                    try:
                        json.loads(inner)
                        text = inner
                    except Exception:
                        pass
        if not return_logprobs:
            return text
        TARGET_TOKENS = {'YES', 'NO'} if experiment == 'document' else {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}
        tokens_out = []
        content_lp = choice.get('logprobs', {}).get('content', []) if 'logprobs' in choice else []
        for item in content_lp:
            tok = item.get('token', '').strip()
            if tok.lower() == 'ration':
                break
            if not tok or tok.upper() not in TARGET_TOKENS:
                continue
            lp = item.get('logprob', None)
            try:
                p = math.exp(float(lp)) if lp is not None else None
            except Exception:
                p = None
            alts_pretty = []
            for alt in item.get('top_logprobs', []) or []:
                alp = alt.get('logprob', None)
                try:
                    prob_alt = math.exp(float(alp)) if alp is not None else None
                except Exception:
                    prob_alt = None
                alts_pretty.append(
                    {'token': alt.get('token', '').strip(), 'logprob': alp, 'prob': prob_alt},
                )
            tokens_out.append(
                {'token': tok.upper(), 'logprob': lp, 'prob': p, 'top_alternatives': alts_pretty},
            )
        return {'text': text, 'tokens': tokens_out}
    CATEGORY_TOKENS = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

    def get_resume_batch(csv_path, batch_size):
        if not Path(csv_path).exists():
            return 0
        try:
            df = pd.read_csv(csv_path)
            if df.empty or 'batch_id' not in df.columns:
                return 0
            last_batch = int(df['batch_id'].max())
            print(f' Found existing results. Resuming from batch {last_batch + 1}...')
            return last_batch + 1
        except Exception as e:
            print(f' Could not read existing results ({e}), starting from batch 0.')
            return 0

    def category_vector_to_labels(vector):
        index_to_letter = {0: 'D', 1: 'C', 2: 'F', 3: 'G', 4: 'I', 5: 'A', 6: 'H', 7: 'E', 8: 'B'}
        try:
            return [index_to_letter[i] for i, val in enumerate(vector) if val == 1]
        except Exception:
            return []

    def parse_json_response(response_text, experiment):
        import json, re
        raw = '' if response_text is None else str(response_text)
        txt = re.sub('```(?:json)?', '', raw).strip('` \r\n\t')
        txt = re.sub('[\\x00-\\x1f\\x7f-\\x9f]', '', txt)
        warnings = []
        try:
            data = json.loads(txt)
            return _extract_fields_jsondict(data, raw, experiment, valid=True, warnings=warnings)
        except Exception as e:
            warnings.append(f' json.loads failed: {type(e).__name__}: {e}')
        hx = _extract_fields_heuristic(txt, experiment)
        hx['raw_response'] = raw
        hx['valid'] = (
            hx.get('label') in ('YES', 'NO')
            or (
                experiment != 'document'
                and isinstance(hx.get('categories'), list)
                and len(hx['categories']) > 0
            )
        )
        hx_w = hx.get('warnings', [])
        if isinstance(hx_w, str):
            hx_w = [hx_w] if hx_w else []
        hx['warnings'] = warnings + hx_w + [' Heuristic fallback extraction used']
        return hx

    def _extract_fields_jsondict(data, raw_response, experiment, valid=True, warnings=None):
        if warnings is None:
            warnings = []
        parsed = {
            'valid': valid,
            'raw_response': raw_response,
            'warnings': list(warnings),
            'label': None,
            'rationale': None,
            'evidence': None,
            'categories': None,
        }
        if experiment == 'document':
            parsed['label'] = data.get('label')
            parsed['rationale'] = data.get('rationale')
            parsed['evidence'] = data.get('evidence')
        else:
            cats = data.get('categories', [])
            parsed['categories'] = cats if isinstance(cats, list) else []
            parsed['rationale'] = data.get('rationale')
            parsed['evidence'] = data.get('evidence')
        return parsed

    def _extract_fields_heuristic(text, experiment):
        import re
        out = {'warnings': [], 'label': None, 'rationale': None, 'evidence': None, 'categories': None}
        m = re.search('(?i)\\blabel\\b\\s*[:=]\\s*["\\\']?\\s*(YES|NO)\\s*["\\\']?', text)
        if m:
            out['label'] = m.group(1).upper()
        out['rationale'] = _extract_string_value_after_key(text, 'rationale')
        out['evidence'] = _extract_string_value_after_key(text, 'evidence')
        if experiment != 'document':
            cats = re.findall('\\b([A-J])\\b', text, flags=re.I)
            cats = sorted({c.upper() for c in cats})
            out['categories'] = cats or None
        return out

    def _extract_string_value_after_key(text, key):
        import re
        km = re.search(f'(?i)\\b{re.escape(key)}\\b\\s*:\\s*', text)
        if not km:
            return None
        i = km.end()
        qm = re.search('"', text[i:])
        if not qm:
            return None
        start = i + qm.start() + 1
        qend = text.find('"', start)
        if qend == -1:
            sentinels = []
            a = text.find('assistant', start)
            if a != -1:
                sentinels.append(a)
            brace_line = text.find('\n{', start)
            if brace_line != -1:
                sentinels.append(brace_line)
            sentinels.append(len(text))
            qend = min(sentinels)
        value = text[start:qend]
        value = value.rstrip(']} )').strip()
        return value or None

    def build_fieldnames(experiment: str, prompt_level: str):
        common_meta_doc = [
            'batch_id',
            'letter_id',
            'prompt',
            'raw_response',
            'processing_time_sec',
            'valid',
            'warnings',
        ]
        common_meta_sent = [
            'batch_id',
            'letter_id',
            'sentence_id',
            'prompt',
            'raw_response',
            'processing_time_sec',
            'valid',
            'warnings',
        ]
        if experiment == 'document':
            task_cols = ['label', 'gold_label']
            if prompt_level == 'L5':
                task_cols += ['rationale', 'evidence']
            return common_meta_doc + task_cols
        else:
            task_cols = ['categories', 'gold_categories']
            if prompt_level == 'L5':
                task_cols += ['rationale', 'evidence']
            return common_meta_sent + task_cols

    def ensure_results_file(path: str, fieldnames: list):
        import csv, time
        if not Path(path).exists() or os.stat(path).st_size == 0:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            print(f' Created new results file with header: {path}')
            return
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
        except Exception as e:
            print(f' Could not read existing header ({e}). Rotating file.')
            ts = time.strftime('%Y%m%d-%H%M%S')
            os.rename(path, f'{path}.bak-{ts}')
            ensure_results_file(path, fieldnames)
            return
        if existing_header != fieldnames:
            ts = time.strftime('%Y%m%d-%H%M%S')
            backup = f'{path}.bak-{ts}'
            os.rename(path, backup)
            print(f' Header mismatch → rotated old file to {backup}')
            ensure_results_file(path, fieldnames)

    def _normalize_categories_list(x):
        import re
        if x is None:
            parts = []
        elif isinstance(x, str):
            parts = [p for p in re.split('[,\\s]+', x.strip()) if p]
        elif isinstance(x, (list, tuple, set)):
            parts = list(x)
        else:
            parts = []
        out = sorted({str(p).strip().upper() for p in parts if str(p).strip() in CATEGORY_TOKENS})
        return out

    def _extract_slotwise_probs(token_logprobs, predicted_labels, experiment):
        import math
        import numpy as np
        results = []
        if not token_logprobs:
            return results
        for slot_idx, label in enumerate(predicted_labels or [], start=1):
            if not label:
                continue
            target_tok = label.strip().upper()
            match = next(
                (t for t in token_logprobs if t.get('token', '').strip().upper() == target_tok),
                None,
            )
            if not match:
                continue
            lp_raw = match.get('logprob', None)
            try:
                lp_val = float(lp_raw) if lp_raw is not None and (not isinstance(lp_raw, str)) else None
            except Exception:
                lp_val = None
            try:
                p_val = math.exp(lp_val) if lp_val is not None else None
            except Exception:
                p_val = None
            lp_val = round(lp_val, 6) if isinstance(lp_val, (float, int)) else None
            p_val = round(p_val, 6) if isinstance(p_val, (float, int)) else None
            alts = []
            for alt in match.get('top_alternatives', []) or []:
                alt_lp_raw = alt.get('logprob', None)
                try:
                    alt_lp = float(alt_lp_raw) if alt_lp_raw is not None and (not isinstance(alt_lp_raw, str)) else None
                except Exception:
                    alt_lp = None
                try:
                    alt_p = math.exp(alt_lp) if alt_lp is not None else None
                except Exception:
                    alt_p = None
                alt_lp = round(alt_lp, 6) if isinstance(alt_lp, (float, int)) else None
                alt_p = round(alt_p, 6) if isinstance(alt_p, (float, int)) else None
                alts.append(
                    {'token': alt.get('token', '').strip(), 'logprob': alt_lp, 'prob': alt_p},
                )
            results.append(
                {
                    'slot': slot_idx,
                    'label': label,
                    'token': target_tok,
                    'logprob': lp_val,
                    'prob': p_val,
                    'alternatives': alts,
                },
            )
        return results

    def _safe_json(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: _safe_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_safe_json(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, 'item') and callable(obj.item):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        else:
            return obj

    def run_batched_inference(
        prompts,
        experiment,
        batch_size=20,
        max_error_rate=0.3,
        show_overall_timing=True,
    ):
        import json, math, time, csv
        from tqdm import tqdm
        fieldnames = build_fieldnames(experiment, PROMPT_LEVEL)
        for col in ['predicted_tokens', 'slot_probs_json']:
            if col not in fieldnames:
                fieldnames.append(col)
        ensure_results_file(RESULTS_FILE, fieldnames)
        start_batch = get_resume_batch(RESULTS_FILE, batch_size)
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        total_time_all = 0.0
        total_items_all = 0
        for batch_idx in range(start_batch, total_batches):
            start_time = time.time()
            start, end = (batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(prompts)))
            batch = prompts[start:end]
            print(f' Processing batch {batch_idx + 1}/{total_batches} ({start}–{end})...')
            results, error_count = ([], 0)
            for p in tqdm(batch, desc=f'Batch {batch_idx + 1}'):
                t0 = time.time()
                try:
                    response = generate_one(
                        prompt=p['prompt'],
                        mode=GEN_MODE,
                        max_new_tokens=256,
                        return_logprobs=True,
                        top_k_alternatives=4,
                        experiment=experiment,
                    )
                    text = response.get('text', '')
                    token_logprobs = response.get('tokens', [])
                    parsed = parse_json_response(text, experiment)
                    if experiment == 'document':
                        predicted_labels = [parsed.get('label')] if parsed.get('label') else []
                    else:
                        predicted_labels = _normalize_categories_list(parsed.get('categories'))
                    slotwise_info = _extract_slotwise_probs(token_logprobs, predicted_labels, experiment)
                    parsed['predicted_tokens'] = json.dumps(_safe_json([t['token'] for t in token_logprobs]), ensure_ascii=False)
                    parsed['slot_probs_json'] = json.dumps(_safe_json(slotwise_info), ensure_ascii=False)
                    meta = {
                        'batch_id': batch_idx,
                        'letter_id': p.get('letter_id') or p.get('id'),
                        'prompt': p['prompt'],
                        'processing_time_sec': round(time.time() - t0, 2),
                    }
                    if experiment != 'document':
                        meta['sentence_id'] = p.get('sentence_id')
                    parsed.update(meta)
                    if experiment == 'document':
                        parsed['gold_label'] = 'YES' if p.get('label') == 1 else 'NO'
                    else:
                        gold_vec = category_vector_to_labels(p.get('category_vector', []))
                        gold_cats = _normalize_categories_list(gold_vec)
                        parsed['gold_categories'] = json.dumps(_safe_json(gold_cats or ['J']), ensure_ascii=False)
                        parsed['categories'] = json.dumps(_safe_json(predicted_labels), ensure_ascii=False)
                    if not parsed.get('valid', True):
                        error_count += 1
                    row = {k: parsed.get(k) for k in fieldnames}
                    if isinstance(row.get('warnings'), list):
                        row['warnings'] = '; '.join(row['warnings'])
                    results.append(row)
                except Exception as e:
                    error_count += 1
                    print(f' Error generating response: {e}')
                    row = {k: None for k in fieldnames}
                    row.update(
                        {
                            'batch_id': batch_idx,
                            'letter_id': p.get('letter_id') or p.get('id'),
                            'prompt': p['prompt'],
                            'processing_time_sec': round(time.time() - t0, 2),
                            'valid': False,
                            'warnings': f'Exception: {e}',
                        },
                    )
                    if experiment != 'document':
                        row['sentence_id'] = p.get('sentence_id')
                    results.append(row)
            with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(results)
                f.flush()
                os.fsync(f.fileno())
            batch_time = round(time.time() - start_time, 2)
            avg_time = round(batch_time / (len(batch) or 1), 2)
            error_rate = error_count / (len(batch) or 1)
            print(
                f' Saved batch {batch_idx + 1} ({error_count} errors, {error_rate:.0%} error rate, avg {avg_time}s/item)',
            )
            total_time_all += batch_time
            total_items_all += len(batch)
            if error_rate > max_error_rate:
                raise RuntimeError(f' ERROR LIMIT REACHED in batch {batch_idx + 1}: {error_rate:.0%}')
        print(' All batches processed successfully!')
        if show_overall_timing and total_items_all > 0:
            overall_avg = round(total_time_all / total_items_all, 2)
            print(
                f'Overall average time per prompt: {overall_avg}s ({total_items_all} total items, {total_time_all:.1f}s total)',
            )
    if EXPERIMENT == 'document':
        DOC_PATH = os.path.join(DATA_DIR, 'documents.pkl')
        with open(DOC_PATH, 'rb') as f:
            docs = pickle.load(f)
        for d in docs:
            recipient = d.get('recipient')
            if recipient and recipient != 'Unknown':
                d['recipient'] = recipient.title()
        if LIMIT:
            docs = stratified_sample_document(docs, LIMIT)
        else:
            print(f' Loaded full dataset: {len(docs)} documents')
    elif EXPERIMENT in ['sentence_binary', 'sentence_multilabel']:
        SENT_PATH = os.path.join(DATA_DIR, 'sentences.pkl')
        with open(SENT_PATH, 'rb') as f:
            sentences = pickle.load(f)
        instances = sentences[CONTEXT_WINDOW]
        DOC_PATH = os.path.join(DATA_DIR, 'documents.pkl')
        with open(DOC_PATH, 'rb') as f:
            docs = pickle.load(f)
        for d in docs:
            recipient = d.get('recipient')
            if recipient and recipient != 'Unknown':
                d['recipient'] = recipient.title()
        doc_meta = {d['id']: {'recipient': d['recipient'], 'date': d['date']} for d in docs}
        for s in instances:
            meta = doc_meta.get(s['letter_id'], {'recipient': 'Unknown', 'date': 'Unknown'})
            recipient = meta.get('recipient') or 'Unknown'
            date = meta.get('date') or 'Unknown'
            if recipient != 'Unknown':
                recipient = recipient.title()
            s['recipient'] = recipient
            s['date'] = date
        missing_meta = sum((1 for s in instances if s['recipient'] == 'Unknown' or s['date'] == 'Unknown'))
        if missing_meta > 0:
            print(f'  Warning: {missing_meta} sentences could not be matched with metadata.')
        if LIMIT:
            instances = stratified_sample_sentences(instances, LIMIT)
            print(
                f' Loaded stratified sample: {len(instances)} sentences (context={CONTEXT_WINDOW})',
            )
        else:
            print(f' Loaded full dataset: {len(instances)} sentences (context={CONTEXT_WINDOW})')
            ref_sentences = [s for s in instances if sum(s['category_vector']) > 0]
            noref_sentences = [s for s in instances if sum(s['category_vector']) == 0]
            n_ref = len(ref_sentences)
            n_noref = len(noref_sentences)
            print(f' Reference stats before balancing:')
            print(f'   Ref (≥1 category): {n_ref}')
            print(f'   NoRef (all 0):     {n_noref}')
            if n_noref > n_ref:
                random.seed(42)
                noref_sampled = random.sample(noref_sentences, n_ref)
                instances = ref_sentences + noref_sampled
                random.shuffle(instances)
                print(
                    f' Balanced dataset to {len(instances)} sentences ({n_ref} Ref + {len(noref_sampled)} NoRef)',
                )
            else:
                print('  Fewer or equal NoRef sentences than Ref sentences – keeping all.')
    DOC_PROMPTS_PATH = str(paths.PROMPTS_DIR / 'document.json')
    SENTENCE_PROMPTS_PATH = str(paths.PROMPTS_DIR / 'sentence.json')
    with open(DOC_PROMPTS_PATH, 'r') as f:
        doc_prompts = json.load(f)
    with open(SENTENCE_PROMPTS_PATH, 'r') as f:
        sentence_prompts = json.load(f)

    def fill_placeholders(template: str, replacements: dict) -> str:
        for key, value in replacements.items():
            placeholder = '{' + key + '}'
            if placeholder in template:
                safe_value = str(value) if value is not None else 'Unknown'
                template = template.replace(placeholder, safe_value)
        return template

    def count_tokens(text: str) -> int:
        try:
            return len(llm.tokenize(text.encode('utf-8')))
        except Exception:
            return max(1, len(text) // 4)
    prepared_prompts = []
    if EXPERIMENT == 'document':
        template = doc_prompts[PROMPT_LEVEL]
        for d in docs:
            prompt = fill_placeholders(
                template,
                {
                    'TEXT': d.get('text', ''),
                    'RECIPIENT': d.get('recipient') or 'Unknown',
                    'DATE': d.get('date') or 'Unknown',
                },
            )
            prepared_prompts.append(
                {
                    'id': d['id'],
                    'prompt': prompt,
                    'label': d['label'],
                    'n_tokens': count_tokens(prompt),
                },
            )
        print(f' Generated {len(prepared_prompts)} document-level prompts.')
    elif EXPERIMENT in ['sentence_multilabel', 'sentence_binary']:
        if CONTEXT_WINDOW == 0:
            template = sentence_prompts['no_context'][PROMPT_LEVEL]
            for s in instances:
                match = re.search('<target>(.*?)</target>', s['context'], re.DOTALL)
                target_sentence = match.group(1).strip() if match else ''
                prompt = fill_placeholders(
                    template,
                    {
                        'SENTENCE': target_sentence,
                        'RECIPIENT': s.get('recipient') or 'Unknown',
                        'DATE': s.get('date') or 'Unknown',
                    },
                )
                prepared_prompts.append(
                    {
                        'letter_id': s['letter_id'],
                        'sentence_id': s['sentence_id'],
                        'prompt': prompt,
                        'category_vector': s.get('category_vector'),
                        'n_tokens': count_tokens(prompt),
                    },
                )
            print(f' Generated {len(prepared_prompts)} sentence-level prompts (no context).')
        else:
            template = sentence_prompts['context'][PROMPT_LEVEL]
            for s in instances:
                full_context = s['context']
                before = re.split('<target>', full_context)[0].strip()
                after = re.split('</target>', full_context)[1].strip() if '</target>' in full_context else ''
                match = re.search('<target>(.*?)</target>', full_context, re.DOTALL)
                target = match.group(1).strip() if match else ''
                prompt = fill_placeholders(
                    template,
                    {
                        'CONTEXT_BEFORE': before,
                        'TARGET': target,
                        'CONTEXT_AFTER': after,
                        'RECIPIENT': s.get('recipient') or 'Unknown',
                        'DATE': s.get('date') or 'Unknown',
                    },
                )
                prepared_prompts.append(
                    {
                        'letter_id': s['letter_id'],
                        'sentence_id': s['sentence_id'],
                        'prompt': prompt,
                        'category_vector': s.get('category_vector'),
                        'n_tokens': count_tokens(prompt),
                    },
                )
            print(f' Generated {len(prepared_prompts)} sentence-level prompts (with context).')
    if prepared_prompts:
        prompt_lengths_chars = [len(p['prompt']) for p in prepared_prompts]
        prompt_lengths_tokens = [p.get('n_tokens', count_tokens(p['prompt'])) for p in prepared_prompts]
        print(
            f' Prompt length (chars) – min: {min(prompt_lengths_chars)}, max: {max(prompt_lengths_chars)}, avg: {np.mean(prompt_lengths_chars):.1f}',
        )
        print(
            f' Prompt length (tokens, approx) – min: {min(prompt_lengths_tokens)}, max: {max(prompt_lengths_tokens)}, avg: {np.mean(prompt_lengths_tokens):.1f}',
        )
    RESULTS_FILE = os.path.join(RESULTS_DIR, f'llm_results_{EXPERIMENT}_{PROMPT_LEVEL}.csv')
    BATCH_SIZE = 100
    if OUTPUT_MODE.upper() == 'JSON':
        GEN_MODE = 'doc_json' if EXPERIMENT == 'document' else 'sent_json'
    else:
        GEN_MODE = 'text'
    run_batched_inference(
        prepared_prompts,
        EXPERIMENT,
        batch_size=BATCH_SIZE,
        show_overall_timing=True,
    )
    print(f'\n Inference complete for {EXPERIMENT.upper()} ({PROMPT_LEVEL}, {OUTPUT_MODE})')
    print(f' Results saved to:\n{RESULTS_FILE}')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run prompt-based reference classification.')
    parser.add_argument(
        '--task',
        choices=['document', 'sentence_multilabel'],
        default='sentence_multilabel',
    )
    parser.add_argument('--context', type=int, choices=[0, 4], default=4)
    parser.add_argument('--level', choices=['L1', 'L2', 'L3', 'L4', 'L5'], default='L4')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    run(
        experiment=args.task,
        context_window=args.context,
        prompt_level=args.level,
        limit=args.limit,
    )
