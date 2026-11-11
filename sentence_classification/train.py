"""Sentence-level multilabel BERT training and held-out evaluation."""

from shared import paths
from shared.labels import CATEGORY_INDEX
from shared.sentences import prepare_sentences
import re
from collections import defaultdict
from shared.utils import load_pickle
from shared.classes import Letter, Annotation
import os
import pickle
import random
import numpy as np
from tqdm import tqdm
from pprint import pprint
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers import DataCollatorWithPadding
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset
from collections import Counter
import shutil
from transformers import default_data_collator, TrainerCallback
from scipy.special import expit
from datetime import datetime
import pandas as pd
from torch.utils.data import DataLoader


def run(window_size=0, use_target_tags=True, require_traces=False):
    """Train the multilabel model and save the predictions for each test fold."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS backend (Apple Silicon GPU)')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    id2label = {i: label for label, i in CATEGORY_INDEX.items()}
    label2id = CATEGORY_INDEX
    num_labels = len(label2id)
    print('Label mapping:')
    pprint(label2id)
    all_bert_sentences = prepare_sentences(
        load_pickle(),
        window_size=window_size,
        use_target_tags=use_target_tags,
        require_traces=require_traces,
    )

    class SentenceClassificationDataset(Dataset):

        def __init__(self, entries, tokenizer, max_length=512, return_text=False):
            self.entries = entries
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.return_text = return_text

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, idx):
            entry = self.entries[idx]
            context = entry['context']
            encoding = self.tokenizer(
                context,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
            )
            item = {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(entry['category_vector'], dtype=torch.float),
                'letter_id': entry['letter_id'],
                'sentence_id': entry['sentence_id'],
            }
            if self.return_text:
                item['raw_text'] = context
            return item
    n_splits = 10
    max_length = 512
    has_category = [any((v == 1 for v in entry['category_vector'])) for entry in all_bert_sentences]
    print(f'\n Total sentences: {len(all_bert_sentences)}')
    print(f' Sentences with at least one category: {sum(has_category)}')
    print(f' Sentences without any category: {len(all_bert_sentences) - sum(has_category)}')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_datasets = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(all_bert_sentences, has_category)):
        print('\n' + '=' * 100)
        print(f' Fold {fold_idx + 1}/{n_splits}')
        trainval_data = [all_bert_sentences[i] for i in trainval_idx]
        test_data = [all_bert_sentences[i] for i in test_idx]
        random.shuffle(trainval_data)
        val_size = int(0.1 * len(trainval_data))
        val_data = trainval_data[:val_size]
        train_data = trainval_data[val_size:]

        def category_stats(dataset, name):
            counter = Counter()
            for entry in dataset:
                for i, v in enumerate(entry['category_vector']):
                    if v == 1:
                        counter[i] += 1
            total = len(dataset)
            with_category = sum((1 for entry in dataset if any(entry['category_vector'])))
            print(
                f' {name}: {total} sentences | With category: {with_category} | Empty: {total - with_category}',
            )
            print('   Category distribution:')
            for i in range(len(entry['category_vector'])):
                print(f'     {i}: {counter[i]}')
        category_stats(train_data, 'Train')
        category_stats(val_data, 'Val')
        category_stats(test_data, 'Test')
        test_ids = [(d['letter_id'], d['sentence_id']) for d in test_data]
        print(f' Test IDs (first 5): {test_ids[:5]} ...')
        train_dataset = SentenceClassificationDataset(train_data, tokenizer, max_length)
        val_dataset = SentenceClassificationDataset(val_data, tokenizer, max_length)
        test_dataset = SentenceClassificationDataset(test_data, tokenizer, max_length, return_text=True)
        fold_datasets.append(
            {
                'fold': fold_idx,
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset,
                'test_ids': test_ids,
            },
        )
    print('\n Cross-validation splits created.')

    class MacroF1EarlyStoppingCallback(TrainerCallback):

        def __init__(self, early_stopping_patience=2, early_stopping_threshold=0.0):
            self.early_stopping_patience = early_stopping_patience
            self.early_stopping_threshold = early_stopping_threshold
            self._best_metric = None
            self._patience_counter = 0

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            metric = metrics.get('macro_f1', None)
            if metric is None:
                return control
            if self._best_metric is None or metric > self._best_metric + self.early_stopping_threshold:
                self._best_metric = metric
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                print(
                    f'  Macro-F1 did not improve. Patience: {self._patience_counter}/{self.early_stopping_patience}',
                )
                if self._patience_counter >= self.early_stopping_patience:
                    control.should_training_stop = True
                    print(' Early stopping triggered based on macro-F1.')
            return control

    def compute_metrics(eval_pred):
        """Calculate classification metrics from logits and gold labels."""
        logits, labels = eval_pred
        probs = expit(logits)
        preds = (probs > 0.5).astype(int)
        report = classification_report(
            labels,
            preds,
            output_dict=True,
            zero_division=0,
            target_names=[id2label[i] for i in range(len(id2label))],
        )
        gold_is_ref = np.any(labels == 1, axis=1)
        pred_is_ref = np.any(preds == 1, axis=1)
        ref_report = classification_report(
            gold_is_ref,
            pred_is_ref,
            labels=[0, 1],
            target_names=['NoRef', 'Ref'],
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'ref_precision': ref_report['Ref']['precision'],
            'ref_recall': ref_report['Ref']['recall'],
            'ref_f1': ref_report['Ref']['f1-score'],
            'noref_precision': ref_report['NoRef']['precision'],
            'noref_recall': ref_report['NoRef']['recall'],
            'noref_f1': ref_report['NoRef']['f1-score'],
        }
        for category in id2label.values():
            metrics[f'{category}_precision'] = report[category]['precision']
            metrics[f'{category}_recall'] = report[category]['recall']
            metrics[f'{category}_f1'] = report[category]['f1-score']
        return metrics

    def train_fold(fold_index, train_dataset, val_dataset, output_dir):
        """Train one fold and return the trainer with its selected checkpoint."""
        print(f'\n Starting training for Fold {fold_index}')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-cased',
            num_labels=num_labels,
            problem_type='multi_label_classification',
            id2label=id2label,
            label2id=label2id,
        ).to(
            device,
        )
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            learning_rate=2e-05,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='macro_f1',
            save_total_limit=2,
            seed=42,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            callbacks=[MacroF1EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        print(f'\n Final model automatically saved to: {output_dir}')
        return trainer
    pd.set_option('display.max_columns', None)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    cv_output_dir = str(
        paths.SENTENCE_MODELS / (f'context_{window_size}' + ('_untagged' if require_traces else '')) / f'cv_{timestamp}',
    )
    os.makedirs(cv_output_dir, exist_ok=True)
    print(f' Created base output directory: {cv_output_dir}')
    results = []
    for fold_info in fold_datasets:
        fold = fold_info['fold']
        train_dataset = fold_info['train']
        val_dataset = fold_info['val']
        test_dataset = fold_info['test']
        test_ids = fold_info['test_ids']
        fold_dir = os.path.join(cv_output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        print(f'\n===  Fold {fold} ===')
        trainer = train_fold(fold, train_dataset, val_dataset, fold_dir)
        best_model_path = trainer.state.best_model_checkpoint
        model = BertForSequenceClassification.from_pretrained(best_model_path).to(device)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].cpu().numpy()
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels)
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        probs = expit(logits)
        preds = (probs > 0.5).astype(int)
        gold_is_ref = np.any(labels == 1, axis=1)
        pred_is_ref = np.any(preds == 1, axis=1)
        ref_report = classification_report(
            gold_is_ref,
            pred_is_ref,
            labels=[0, 1],
            target_names=['NoRef', 'Ref'],
            output_dict=True,
            zero_division=0,
        )
        cat_report = classification_report(
            labels,
            preds,
            target_names=[id2label[i] for i in range(num_labels)],
            output_dict=True,
            zero_division=0,
        )
        result = {
            'fold': fold,
            'macro_precision': cat_report['macro avg']['precision'],
            'macro_recall': cat_report['macro avg']['recall'],
            'macro_f1': cat_report['macro avg']['f1-score'],
            'weighted_precision': cat_report['weighted avg']['precision'],
            'weighted_recall': cat_report['weighted avg']['recall'],
            'weighted_f1': cat_report['weighted avg']['f1-score'],
            'ref_precision': ref_report['Ref']['precision'],
            'ref_recall': ref_report['Ref']['recall'],
            'ref_f1': ref_report['Ref']['f1-score'],
            'noref_precision': ref_report['NoRef']['precision'],
            'noref_recall': ref_report['NoRef']['recall'],
            'noref_f1': ref_report['NoRef']['f1-score'],
        }
        for label, name in id2label.items():
            result[f'{name}_precision'] = cat_report[name]['precision']
            result[f'{name}_recall'] = cat_report[name]['recall']
            result[f'{name}_f1'] = cat_report[name]['f1-score']
        results.append(result)
        true_label_names = []
        pred_label_names = []
        for true_vec, pred_vec in zip(labels, preds):
            true_names = [id2label[i] for i, val in enumerate(true_vec) if val == 1]
            pred_names = [id2label[i] for i, val in enumerate(pred_vec) if val == 1]
            true_label_names.append(', '.join(true_names))
            pred_label_names.append(', '.join(pred_names))
        pred_df = pd.DataFrame(
            {
                'letter_id': [entry['letter_id'] for entry in test_dataset],
                'sentence_id': [entry['sentence_id'] for entry in test_dataset],
                'true_labels': labels.tolist(),
                'true_names': true_label_names,
                'pred_labels': preds.tolist(),
                'pred_names': pred_label_names,
                'probabilities': probs.tolist(),
            },
        )
        if hasattr(test_dataset, 'return_text') and test_dataset.return_text:
            pred_df['raw_text'] = [entry['raw_text'] for entry in test_dataset]
        pred_file = os.path.join(fold_dir, 'predictions.csv')
        pred_df.to_csv(pred_file, index=False)
        print(f' Predictions saved to {pred_file}')
        shap_data = {
            'model_path': best_model_path,
            'tokenized_test_data': {
                'raw_texts': [entry['raw_text'] for entry in test_dataset],
                'letter_ids': [entry['letter_id'] for entry in test_dataset],
                'sentence_ids': [entry['sentence_id'] for entry in test_dataset],
                'true_labels': [entry['labels'].tolist() for entry in test_dataset],
            },
        }
        shap_path = os.path.join(fold_dir, 'logs', f'shap_data_fold_{fold}.pkl')
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        with open(shap_path, 'wb') as f:
            pickle.dump(shap_data, f)
        print(f' SHAP inputs saved to: {shap_path}')
        rows = []
        rows.append(
            ['Ref vs NoRef', 'Ref', result['ref_precision'], result['ref_recall'], result['ref_f1']],
        )
        rows.append(
            [
                'Ref vs NoRef',
                'NoRef',
                result['noref_precision'],
                result['noref_recall'],
                result['noref_f1'],
            ],
        )
        rows.append(
            [
                'Averages',
                'Macro',
                result['macro_precision'],
                result['macro_recall'],
                result['macro_f1'],
            ],
        )
        rows.append(
            [
                'Averages',
                'Weighted',
                result['weighted_precision'],
                result['weighted_recall'],
                result['weighted_f1'],
            ],
        )
        for i in range(num_labels):
            name = id2label[i]
            rows.append(
                [
                    'Per Category',
                    name,
                    result[f'{name}_precision'],
                    result[f'{name}_recall'],
                    result[f'{name}_f1'],
                ],
            )
        fold_df = pd.DataFrame(rows, columns=['Group', 'Label', 'Precision', 'Recall', 'F1'])
        print(f'\n Fold {fold} results:')
        display(fold_df)
        fold_eval_path = os.path.join(fold_dir, 'evaluation.txt')
        with open(fold_eval_path, 'w') as f:
            f.write(f'EVALUATION FOR FOLD {fold}\n')
            f.write('==========================\n\n')
            f.write('Ref vs NoRef (binary):\n')
            f.write(
                f"  Ref     - Precision: {result['ref_precision']:.4f} | Recall: {result['ref_recall']:.4f} | F1: {result['ref_f1']:.4f}\n",
            )
            f.write(
                f"  NoRef   - Precision: {result['noref_precision']:.4f} | Recall: {result['noref_recall']:.4f} | F1: {result['noref_f1']:.4f}\n\n",
            )
            f.write('Macro Avg:\n')
            f.write(
                f"  Precision: {result['macro_precision']:.4f} | Recall: {result['macro_recall']:.4f} | F1: {result['macro_f1']:.4f}\n",
            )
            f.write('Weighted Avg:\n')
            f.write(
                f"  Precision: {result['weighted_precision']:.4f} | Recall: {result['weighted_recall']:.4f} | F1: {result['weighted_f1']:.4f}\n\n",
            )
            f.write('Per-category scores:\n')
            for label, name in id2label.items():
                f.write(
                    f"  {name:35} - Precision: {result[f'{name}_precision']:.4f} | Recall: {result[f'{name}_recall']:.4f} | F1: {result[f'{name}_f1']:.4f}\n",
                )
    df = pd.DataFrame(results).sort_values('fold')
    df.to_csv(os.path.join(cv_output_dir, 'fold_metrics.csv'), index=False)
    print(df.to_string(index=False))
    eval_dir = os.path.join(cv_output_dir, 'general')
    os.makedirs(eval_dir, exist_ok=True)
    all_preds = []
    for fold_info in fold_datasets:
        fold = fold_info['fold']
        fold_dir = os.path.join(cv_output_dir, f'fold_{fold}')
        pred_path = os.path.join(fold_dir, 'predictions.csv')
        fold_preds = pd.read_csv(pred_path)
        fold_preds['fold'] = fold
        all_preds.append(fold_preds)
    all_preds_df = pd.concat(all_preds, ignore_index=True)
    all_preds_path = os.path.join(eval_dir, 'all_folds_predictions.csv')
    all_preds_df.to_csv(all_preds_path, index=False)
    print(f" All folds' predictions saved to: {all_preds_path}")
    output_file = os.path.join(eval_dir, 'evaluation_results.txt')
    with open(output_file, 'w') as f:
        f.write('EVALUATION SUMMARY\n')
        f.write('===================\n\n')
        for row in df.to_dict(orient='records'):
            f.write(f"FOLD {row['fold']}\n")
            f.write('-' * 30 + '\n')
            f.write('Ref vs NoRef (binary):\n')
            f.write(
                f"  Ref     - Precision: {row['ref_precision']:.4f} | Recall: {row['ref_recall']:.4f} | F1: {row['ref_f1']:.4f}\n",
            )
            f.write(
                f"  NoRef   - Precision: {row['noref_precision']:.4f} | Recall: {row['noref_recall']:.4f} | F1: {row['noref_f1']:.4f}\n\n",
            )
            f.write('Macro Avg:\n')
            f.write(
                f"  Precision: {row['macro_precision']:.4f} | Recall: {row['macro_recall']:.4f} | F1: {row['macro_f1']:.4f}\n",
            )
            f.write('Weighted Avg:\n')
            f.write(
                f"  Precision: {row['weighted_precision']:.4f} | Recall: {row['weighted_recall']:.4f} | F1: {row['weighted_f1']:.4f}\n\n",
            )
            f.write('Per-category scores:\n')
            for label, name in id2label.items():
                f.write(
                    f"  {name:35} - Precision: {row[f'{name}_precision']:.4f} | Recall: {row[f'{name}_recall']:.4f} | F1: {row[f'{name}_f1']:.4f}\n",
                )
            f.write('\n')
        f.write('===================\n')
        f.write('AVERAGE SCORES\n')
        f.write('===================\n')
        for col in [
            'macro_precision',
            'macro_recall',
            'macro_f1',
            'weighted_precision',
            'weighted_recall',
            'weighted_f1',
        ]:
            mean = df[col].mean()
            std = df[col].std()
            f.write(f'{col}: {mean:.4f} ± {std:.4f}\n')
        f.write('\n===================\n')
        f.write('PER-CATEGORY AVERAGE SCORES\n')
        f.write('===================\n')
        for i in range(num_labels):
            name = id2label[i]
            precision_mean = df[f'{name}_precision'].mean()
            recall_mean = df[f'{name}_recall'].mean()
            f1_mean = df[f'{name}_f1'].mean()
            f.write(
                f'{name:35} - Precision: {precision_mean:.4f} | Recall: {recall_mean:.4f} | F1: {f1_mean:.4f}\n',
            )
    print(f'\n All metrics saved to: {output_file}')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train the sentence classifier.')
    parser.add_argument('--context', type=int, choices=range(7), default=None)
    parser.add_argument(
        '--untagged',
        action='store_true',
        help='Use the untagged, one-sentence-context configuration on letters with traces.',
    )
    options = parser.parse_args()
    context = options.context if options.context is not None else 1 if options.untagged else 0
    if options.untagged and context != 1:
        parser.error('The untagged configuration uses one sentence of context.')
    run(window_size=context, use_target_tags=not options.untagged, require_traces=options.untagged)
