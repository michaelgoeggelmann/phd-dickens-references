"""Document-level BERT training and title-prefilter evaluation."""

from shared import paths
from torch import nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
import os
import logging
import pickle
import torch
import numpy as np
import random
import csv
import glob
import re
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertTokenizer,
    TrainerCallback,
)
from sklearn.model_selection import KFold
from datasets import Dataset
from transformers import set_seed
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_recall_curve,
)
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd


def run(use_metadata=False, num_epochs=5):
    """Train and evaluate the text or metadata model across ten folds."""
    set_seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = paths.DOCUMENT_MODELS / ('metadata' if use_metadata else 'plain') / f'cv_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    paths.prepare_directories()
    DATA_PATH = str(paths.DOCUMENT_DATA / 'processed_data_metadata.pkl')
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    BASELINE_PATH = str(paths.DOCUMENT_DATA / 'baseline_predictions.pkl')
    with open(BASELINE_PATH, 'rb') as f:
        test_baseline_predictions = pickle.load(f)
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_mapping = {}
    fold_indices_list = list(kf.split(data['text']))
    for i in range(k):
        test_idx = fold_indices_list[i][1]
        val_idx = fold_indices_list[(i + 1) % k][1]
        train_idx = np.concatenate([fold_indices_list[j][1] for j in range(k) if j not in [i, (i + 1) % k]])
        fold_mapping[i] = {'test': test_idx, 'val': val_idx, 'train': train_idx}
    all_c_f1_scores = []
    all_c_precisions = []
    all_c_recalls = []
    all_b_f1_scores = []
    all_b_precisions = []
    all_b_recalls = []
    all_m_f1_scores = []
    all_m_precisions = []
    all_m_recalls = []
    all_m_full_precisions = []
    all_m_full_recalls = []
    all_m_full_f1_scores = []
    best_checkpoints = []
    best_checkpoint_results = []
    all_true_labels = []
    all_pred_probabilities = []
    cv_results = {}
    total_baseline_count = 0
    total_model_count = 0

    def compute_metrics(eval_pred):
        """Calculate classification metrics from logits and gold labels."""
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=-1)
        labels = np.array(labels)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        return {'eval_precision': precision, 'eval_recall': recall, 'eval_f1': f1}

    def save_confusion_matrix(y_true, y_pred, fold, filename):
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Reference', 'Reference'],
            yticklabels=['No Reference', 'Reference'],
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(filename)
        plt.close()
        print(f' Confusion matrix for Fold {fold} saved to {filename}')

    def save_classification_report(y_true, y_pred, fold, filename):
        report = classification_report(y_true, y_pred, target_names=['No Reference', 'Reference'])
        with open(filename, 'w') as f:
            f.write(f'Classification Report - Fold {fold}\n')
            f.write(report)
        print(f' Classification report for Fold {fold} saved to {filename}')

    def plot_cv_boxplots(
        all_precisions,
        all_recalls,
        all_f1_scores,
        save_path='cv_metrics_boxplots.png',
    ):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=[all_precisions, all_recalls, all_f1_scores], palette='Blues')
        plt.xticks([0, 1, 2], ['Precision', 'Recall', 'F1-Score'])
        plt.ylabel('Score')
        plt.grid(axis='y')
        plt.savefig(save_path)
        plt.show()
        print(f' Boxplots saved to {save_path}')

    def plot_cv_confusion_matrix(
        all_true_labels,
        all_pred_probabilities,
        save_path='cv_aggregated_confusion_matrix.png',
    ):
        if len(all_true_labels) == 0 or len(all_pred_probabilities) == 0:
            print(' No data available for the confusion matrix.')
            return
        aggregated_conf_matrix = confusion_matrix(all_true_labels, all_pred_probabilities > 0.5)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            aggregated_conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Reference', 'Reference'],
            yticklabels=['No Reference', 'Reference'],
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(save_path)
        plt.show()
        print(f' Confusion matrix saved to {save_path}')

    def plot_cv_metrics_over_folds(
        all_precisions,
        all_recalls,
        all_f1_scores,
        save_path='cv_metrics_over_folds.png',
    ):
        if len(all_f1_scores) == 0:
            print(' No fold-level F1 scores available.')
            return
        folds = np.arange(1, len(all_f1_scores) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(folds, all_precisions, marker='o', label='Precision', linestyle='--')
        plt.plot(folds, all_recalls, marker='s', label='Recall', linestyle='--')
        plt.plot(folds, all_f1_scores, marker='^', label='F1-Score', linestyle='--')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid()
        plt.xticks(folds)
        plt.savefig(save_path)
        plt.show()
        print(f' Fold metrics saved to {save_path}')

    def plot_cv_precision_recall_curve(
        all_true_labels,
        all_pred_probabilities,
        save_path='cv_precision_recall_curve.png',
    ):
        if len(all_true_labels) == 0 or len(all_pred_probabilities) == 0:
            print(' No data available for the precision-recall curve.')
            return
        precision, recall, _ = precision_recall_curve(all_true_labels, all_pred_probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.show()
        print(f' Precision-recall curve saved to {save_path}')
    for fold in range(k):
        test_idx = fold_mapping[fold]['test']
        val_idx = fold_mapping[fold]['val']
        train_idx = fold_mapping[fold]['train']
        train_labels = [data['label'][i] for i in train_idx]
        val_labels = [data['label'][i] for i in val_idx]
        test_labels = [data['label'][i] for i in test_idx]
        train_label_dist = Counter(train_labels)
        val_label_dist = Counter(val_labels)
        test_label_dist = Counter(test_labels)
        print(f'\nFold {fold + 1}:')
        print(
            f'    Training data: Folds {[i + 1 for i in range(k) if i not in [fold, (fold + 1) % k]]} | Distribution: {train_label_dist}',
        )
        print(f'    Validation data: Fold {(fold + 1) % k + 1} | Distribution: {val_label_dist}')
        print(f'    Test data: Fold {fold + 1} | Distribution: {test_label_dist}')
    print('\n Fold and label distributions listed.')
    for fold in range(k):
        print(f'\n Processing Fold {fold + 1}/{k}')
        test_idx = fold_mapping[fold]['test']
        val_idx = fold_mapping[fold]['val']
        train_idx = fold_mapping[fold]['train']
        test_texts = [data['text'][i] for i in test_idx]
        test_labels = [data['label'][i] for i in test_idx]
        test_ids = [data['id'][i] for i in test_idx]
        baseline_predictions = np.array([test_baseline_predictions[test_id] for test_id in test_ids])
        decision_source = np.array(['baseline' if pred == 1 else 'model' for pred in baseline_predictions])
        val_texts = [data['text'][i] for i in val_idx]
        val_labels = [data['label'][i] for i in val_idx]
        train_texts = [data['text'][i] for i in train_idx]
        train_labels = [data['label'][i] for i in train_idx]
        indices_for_model = [i for i in range(len(test_ids)) if baseline_predictions[i] == 0]
        test_texts_for_model = [test_texts[i] for i in indices_for_model]
        test_labels_for_model = [test_labels[i] for i in indices_for_model]
        test_ids_for_model = [test_ids[i] for i in indices_for_model]
        print(
            f"    Baseline predicted 'No Reference' for {len(indices_for_model)} instances, passing them to BERT.",
        )
        train_label_dist = Counter(train_labels)
        val_label_dist = Counter(val_labels)
        test_label_dist = Counter(test_labels)
        print(f' Fold {fold + 1}:')
        print(f'    Training Folds: {[i + 1 for i in range(k) if i not in [fold, (fold + 1) % k]]}')
        print(f'    Validation Fold: {(fold + 1) % k + 1}')
        print(f'    Test Fold: {fold + 1}')
        print(
            f'    Dataset sizes - Train: {len(train_texts)}, Eval: {len(val_texts)}, Test: {len(test_texts)}',
        )
        print(
            f'    Label distribution - Train: {train_label_dist}, Eval: {val_label_dist}, Test: {test_label_dist}',
        )
        MODEL_DIR = str(run_dir / f'cv_fold_{fold + 1}')
        LOG_DIR = os.path.join(MODEL_DIR, 'logs')
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        fold_info_path = os.path.join(LOG_DIR, 'fold_info.txt')
        with open(fold_info_path, 'w') as f:
            f.write(f'Fold {fold + 1}\n')
            f.write(
                f'Training Folds: {[i + 1 for i in range(k) if i not in [fold, (fold + 1) % k]]}\n',
            )
            f.write(f'Validation Fold: {(fold + 1) % k + 1}\n')
            f.write(f'Test Fold: {fold + 1}\n')
            f.write(
                f'Dataset sizes - Train: {len(train_texts)}, Eval: {len(val_texts)}, Test (all): {len(test_texts)}, Test (after baseline): {len(indices_for_model)}\n',
            )
            f.write(
                f'Label distribution - Train: {train_label_dist}, Eval: {val_label_dist}, Test (all): {test_label_dist}\n',
            )
        print(f' Fold information saved to {fold_info_path}')
        training_logfile = os.path.join(LOG_DIR, 'training.log')
        test_logfile = os.path.join(LOG_DIR, 'test_results.txt')
        validation_logfile = os.path.join(LOG_DIR, 'epochs_validation_results.txt')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(training_logfile)],
        )
        print(f' Training started for Fold {fold + 1}/{k}. Logs stored in: {LOG_DIR}')
        logging.info(f'Training started for Fold {fold + 1}/{k}.')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        eval_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
        test_encodings = tokenizer(test_texts_for_model, truncation=True, padding=True, max_length=512)
        csv_test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
        train_dataset = Dataset.from_dict(
            {
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask'],
                'labels': train_labels,
                **({'metadata': [data['metadata'][i] for i in train_idx]} if use_metadata else {}),
            },
        )
        eval_dataset = Dataset.from_dict(
            {
                'input_ids': eval_encodings['input_ids'],
                'attention_mask': eval_encodings['attention_mask'],
                'labels': val_labels,
                **({'metadata': [data['metadata'][i] for i in val_idx]} if use_metadata else {}),
            },
        )
        test_dataset = Dataset.from_dict(
            {
                'input_ids': test_encodings['input_ids'],
                'attention_mask': test_encodings['attention_mask'],
                'labels': test_labels_for_model,
                'letter_ids': test_ids_for_model,
                **({'metadata': [data['metadata'][test_idx[i]] for i in indices_for_model]} if use_metadata else {}),
            },
        )
        csv_test_dataset = Dataset.from_dict(
            {
                'input_ids': csv_test_encodings['input_ids'],
                'attention_mask': csv_test_encodings['attention_mask'],
                'labels': test_labels,
                'letter_ids': test_ids,
                **({'metadata': [data['metadata'][i] for i in test_idx]} if use_metadata else {}),
            },
        )
        if use_metadata:
            metadata_dim = len(data['metadata'][0])
            config = BertConfig.from_pretrained('bert-base-cased', num_labels=2)
            model = BertWithMetadata.from_pretrained(
                'bert-base-cased',
                config=config,
                metadata_dim=metadata_dim,
            )
        else:
            model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=5e-05,
            per_device_train_batch_size=8,
            num_train_epochs=num_epochs,
            logging_dir=LOG_DIR,
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            report_to=['tensorboard'],
            log_level='info',
        )

        class LoggingCallback(TrainerCallback):

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    with open(training_logfile, 'a') as f:
                        f.write(f'{logs}\n')
        trainer = (TrainerWithMetadata if use_metadata else Trainer)(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[LoggingCallback()],
        )
        trainer.train()
        best_model_path = trainer.state.best_model_checkpoint
        best_checkpoints.append(best_model_path)
        with open(validation_logfile, 'w') as f:
            for entry in trainer.state.log_history:
                if 'eval_loss' in entry:
                    f.write(
                        f"Epoch {entry['epoch']}: Loss={entry['eval_loss']:.4f}, Precision={entry.get('eval_precision', 'N/A')}, Recall={entry.get('eval_recall', 'N/A')}, F1={entry.get('eval_f1', 'N/A')}\n",
                    )
        if use_metadata:
            best_model = BertWithMetadata.from_pretrained(
                trainer.state.best_model_checkpoint,
                config=config,
                metadata_dim=metadata_dim,
            )
        else:
            best_model = BertForSequenceClassification.from_pretrained(trainer.state.best_model_checkpoint)
        if not use_metadata:
            best_model.eval()
        with torch.no_grad():
            outputs = best_model(
                input_ids=torch.tensor(test_dataset['input_ids']),
                attention_mask=torch.tensor(test_dataset['attention_mask']),
                **{'metadata': torch.tensor(test_dataset['metadata'])} if use_metadata else {},
            )
            model_predictions_rest = torch.argmax(outputs['logits'], axis=-1).numpy()
            probabilities = torch.nn.functional.softmax(outputs['logits'], dim=-1)[:, 1].numpy()
        combined_predictions = baseline_predictions.copy()
        model_index = 0
        for idx in range(len(combined_predictions)):
            if baseline_predictions[idx] == 0:
                combined_predictions[idx] = model_predictions_rest[model_index]
                model_index += 1
        final_probabilities = baseline_predictions.astype(float)
        model_index = 0
        for idx in range(len(final_probabilities)):
            if baseline_predictions[idx] == 0:
                final_probabilities[idx] = probabilities[model_index]
                model_index += 1
        mask_baseline = decision_source == 'baseline'
        mask_model = decision_source == 'model'
        total_baseline_count += np.sum(mask_baseline)
        total_model_count += np.sum(mask_model)
        true_labels_combined = np.array(test_labels)
        combined_predictions = np.array(combined_predictions)
        precision_comb, recall_comb, f1_comb, _ = precision_recall_fscore_support(
            true_labels_combined,
            combined_predictions,
            average='binary',
        )
        precision_base, recall_base, f1_base, _ = precision_recall_fscore_support(
            true_labels_combined,
            baseline_predictions,
            average='binary',
        )
        precision_model, recall_model, f1_model, _ = precision_recall_fscore_support(
            true_labels_combined[mask_model],
            combined_predictions[mask_model],
            average='binary',
        )
        all_c_precisions.append(precision_comb)
        all_c_recalls.append(recall_comb)
        all_c_f1_scores.append(f1_comb)
        all_b_precisions.append(precision_base)
        all_b_recalls.append(recall_base)
        all_b_f1_scores.append(f1_base)
        all_m_precisions.append(precision_model)
        all_m_recalls.append(recall_model)
        all_m_f1_scores.append(f1_model)
        best_checkpoint_results.append((best_model_path, precision_comb, recall_comb, f1_comb))
        all_true_labels.extend(test_labels)
        all_pred_probabilities.extend(final_probabilities.tolist())
        if not use_metadata:
            best_model.eval()
        with torch.no_grad():
            full_outputs = best_model(
                input_ids=torch.tensor(csv_test_dataset['input_ids']),
                attention_mask=torch.tensor(csv_test_dataset['attention_mask']),
                **{'metadata': torch.tensor(csv_test_dataset['metadata'])} if use_metadata else {},
            )
            model_predictions_full = torch.argmax(full_outputs['logits'], axis=-1).numpy()
            full_probabilities = torch.nn.functional.softmax(full_outputs['logits'], dim=-1)[:, 1].numpy()
        y_true_full = np.array(test_labels)
        y_pred_full = np.array(model_predictions_full)
        precision_full, recall_full, f1_full, _ = precision_recall_fscore_support(y_true_full, y_pred_full, average='binary')
        all_m_full_precisions.append(precision_full)
        all_m_full_recalls.append(recall_full)
        all_m_full_f1_scores.append(f1_full)
        report_file = test_logfile.replace('.txt', f'_classification_report_fold_{fold + 1}.txt')
        save_classification_report(test_labels, combined_predictions, fold + 1, report_file)
        conf_matrix_file = test_logfile.replace('.txt', f'_confusion_matrix_fold_{fold + 1}.png')
        save_confusion_matrix(test_labels, combined_predictions, fold + 1, conf_matrix_file)
        cv_results[fold + 1] = {
            'test_ids': test_ids,
            'gold_labels': test_labels,
            'model_predictions': model_predictions_rest,
            'baseline_prediction': baseline_predictions,
            'prediction_sources': decision_source.tolist(),
            'combined_predictions': combined_predictions,
            'model_on_whole_data_predictions': model_predictions_full,
            'final_probabilities': final_probabilities,
            'precision': precision_comb,
            'recall': recall_comb,
            'f1': f1_comb,
            'precision_baseline': precision_base,
            'recall_baseline': recall_base,
            'f1_baseline': f1_base,
            'precision_model': precision_model,
            'recall_model': recall_model,
            'f1_model': f1_model,
            'precision_model_only': precision_full,
            'recall_model_only': recall_full,
            'f1_model_only': f1_full,
        }
        model_only_rest = np.full_like(combined_predictions, np.nan, dtype=np.float64)
        model_only_rest[mask_model] = combined_predictions[mask_model]
        fold_detail_df = pd.DataFrame(
            {
                'letter_id': test_ids,
                'true_label': test_labels,
                'baseline_prediction': baseline_predictions,
                'model_prediction_only_rest': model_only_rest,
                'combined_prediction': combined_predictions,
                'prediction_source': decision_source,
                'model_predictions_full_dataset': model_predictions_full,
            },
        )
        fold_detail_path = os.path.join(LOG_DIR, 'fold_detailed_predictions.csv')
        fold_detail_df.to_csv(fold_detail_path, index=False)
        print(f' Detailed prediction data for Fold {fold + 1} saved to {fold_detail_path}')
        shap_data = {
            'fold': fold + 1,
            'model_path': best_model_path,
            'tokenized_test_data': {
                'input_ids': csv_test_dataset['input_ids'],
                'attention_mask': csv_test_dataset['attention_mask'],
                'letter_ids': csv_test_dataset['letter_ids'],
                'raw_texts': test_texts,
                'true_labels': test_labels,
            },
        }
        shap_pickle_path = os.path.join(LOG_DIR, f'shap_data_fold_{fold + 1}.pkl')
        with open(shap_pickle_path, 'wb') as f:
            pickle.dump(shap_data, f)
        print(f' SHAP input data for Fold {fold + 1} saved to {shap_pickle_path}')
    all_true_labels_arr = np.array(all_true_labels)
    all_combined_probs_arr = np.array(all_pred_probabilities)
    all_combined_predictions = (all_combined_probs_arr > 0.5).astype(int)
    prec_c_all, rec_c_all, f1_c_all, _ = precision_recall_fscore_support(all_true_labels_arr, all_combined_predictions, average=None)
    precision_comb_no_ref = prec_c_all[0]
    recall_comb_no_ref = rec_c_all[0]
    f1_comb_no_ref = f1_c_all[0]
    y_true_model_all = []
    y_pred_model_all = []
    for fold in range(1, k + 1):
        y_true_model_all.extend(cv_results[fold]['gold_labels'])
        y_pred_model_all.extend(cv_results[fold]['model_on_whole_data_predictions'])
    prec_m_all, rec_m_all, f1_m_all, _ = precision_recall_fscore_support(y_true_model_all, y_pred_model_all, average=None)
    precision_model_no_ref = prec_m_all[0]
    recall_model_no_ref = rec_m_all[0]
    f1_model_no_ref = f1_m_all[0]
    GENERAL_LOG_DIR = str(run_dir / 'general')
    os.makedirs(GENERAL_LOG_DIR, exist_ok=True)
    final_results_path = os.path.join(GENERAL_LOG_DIR, 'cross_validation_results.txt')
    with open(final_results_path, 'w') as f:
        for i, (checkpoint, prec, rec, f1) in enumerate(best_checkpoint_results):
            f.write(
                f'Fold {i + 1}: Best checkpoint (+ result on test data incl. baseline): {checkpoint} \n Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n\n',
            )
        f.write(f'\nFinal Cross-Validation Results:\n')
        f.write(
            f'Combined Precision = {np.mean(all_c_precisions):.4f} ± {np.std(all_c_precisions):.4f}\n',
        )
        f.write(f'Combined Recall = {np.mean(all_c_recalls):.4f} ± {np.std(all_c_recalls):.4f}\n')
        f.write(
            f'Combined F1-Score = {np.mean(all_c_f1_scores):.4f} ± {np.std(all_c_f1_scores):.4f}\n\n',
        )
        f.write(f'Baseline-predictions ({total_baseline_count} samples):\n')
        f.write(
            f'Baseline Precision = {np.mean(all_b_precisions):.4f} ± {np.std(all_b_precisions):.4f}\n',
        )
        f.write(f'Baseline Recall = {np.mean(all_b_recalls):.4f} ± {np.std(all_b_recalls):.4f}\n')
        f.write(
            f'Baseline F1-Score = {np.mean(all_b_f1_scores):.4f} ± {np.std(all_b_f1_scores):.4f}\n\n',
        )
        f.write(f'Model-predictions ({total_model_count} samples):\n')
        f.write(
            f'Model Precision = {np.mean(all_m_precisions):.4f} ± {np.std(all_m_precisions):.4f}\n',
        )
        f.write(f'Model Recall = {np.mean(all_m_recalls):.4f} ± {np.std(all_m_recalls):.4f}\n')
        f.write(
            f'Model F1-Score = {np.mean(all_m_f1_scores):.4f} ± {np.std(all_m_f1_scores):.4f}\n\n',
        )
        f.write(
            f'Model Precision (only model-made decisions) = {np.mean(all_m_full_precisions):.4f} ± {np.std(all_m_full_precisions):.4f}\n',
        )
        f.write(
            f'Model Recall (only model-made decisions) = {np.mean(all_m_full_recalls):.4f} ± {np.std(all_m_full_recalls):.4f}\n',
        )
        f.write(
            f'Model F1-Score (only model-made decisions) = {np.mean(all_m_full_f1_scores):.4f} ± {np.std(all_m_full_f1_scores):.4f}\n\n',
        )
        f.write('\nNo Reference (class 0) metrics (aggregated over all test data):\n')
        f.write(
            f'Combined – Precision: {precision_comb_no_ref:.4f}, Recall: {recall_comb_no_ref:.4f}, F1-Score: {f1_comb_no_ref:.4f}\n',
        )
        f.write(
            f'Model    – Precision: {precision_model_no_ref:.4f}, Recall: {recall_model_no_ref:.4f}, F1-Score: {f1_model_no_ref:.4f}\n',
        )
    cv_results['general'] = {
        'Combined_Precision': np.mean(all_c_precisions),
        'Combined_Recall': np.mean(all_c_recalls),
        'Combined_F1-Score': np.mean(all_c_f1_scores),
        'Combined_Precision_std': np.std(all_c_precisions),
        'Combined_Recall_std': np.std(all_c_recalls),
        'Combined_F1-Score_std': np.std(all_c_f1_scores),
        'Baseline_Precision': np.mean(all_b_precisions),
        'Baseline_Recall': np.mean(all_b_recalls),
        'Baseline_F1-Score': np.mean(all_b_f1_scores),
        'Baseline_Precision_std': np.std(all_b_precisions),
        'Baseline_Recall_std': np.std(all_b_recalls),
        'Baseline_F1-Score_std': np.std(all_b_f1_scores),
        'Model_Precision': np.mean(all_m_precisions),
        'Model_Recall': np.mean(all_m_recalls),
        'Model_F1-Score': np.mean(all_m_f1_scores),
        'Model_Precision_std': np.std(all_m_precisions),
        'Model_Recall_std': np.std(all_m_recalls),
        'Model_F1-Score_std': np.std(all_m_f1_scores),
        'Model_Precision_only_model': np.mean(all_m_full_precisions),
        'Model_Recall_only_model': np.mean(all_m_full_recalls),
        'Model_F1-Score_only_model': np.mean(all_m_full_f1_scores),
        'Model_Precision_only_model_std': np.std(all_m_full_precisions),
        'Model_Recall_only_model_std': np.std(all_m_full_recalls),
        'Model_F1-Score_only_model_std': np.std(all_m_full_f1_scores),
    }
    RESULTS_PICKLE_PATH = run_dir / 'general' / 'results.pkl'
    with open(RESULTS_PICKLE_PATH, 'wb') as f:
        pickle.dump(cv_results, f)
    logging.info(f'Final Cross-Validation Results saved to {final_results_path}')
    plot_cv_boxplots(
        all_c_precisions,
        all_c_recalls,
        all_c_f1_scores,
        os.path.join(GENERAL_LOG_DIR, 'cv_metrics_boxplots.png'),
    )
    plot_cv_confusion_matrix(
        np.array(all_true_labels),
        np.array(all_pred_probabilities),
        os.path.join(GENERAL_LOG_DIR, 'cv_aggregated_confusion_matrix.png'),
    )
    plot_cv_metrics_over_folds(
        all_c_precisions,
        all_c_recalls,
        all_c_f1_scores,
        os.path.join(GENERAL_LOG_DIR, 'cv_metrics_over_folds.png'),
    )
    plot_cv_precision_recall_curve(
        np.array(all_true_labels),
        np.array(all_pred_probabilities),
        os.path.join(GENERAL_LOG_DIR, 'cv_precision_recall_curve.png'),
    )
    rows = []
    for fold in range(1, k + 1):
        entry = cv_results[fold]
        rows.append(
            {
                'Fold': fold,
                'Combined_P': entry['precision'],
                'Combined_R': entry['recall'],
                'Combined_F1': entry['f1'],
                'Baseline_P': entry['precision_baseline'],
                'Baseline_R': entry['recall_baseline'],
                'Baseline_F1': entry['f1_baseline'],
                'Model_P (only rest)': entry['precision_model'],
                'Model_R (only rest)': entry['recall_model'],
                'Model_F1 (only rest)': entry['f1_model'],
                'Model_P (full)': entry['precision_model_only'],
                'Model_R (full)': entry['recall_model_only'],
                'Model_F1 (full)': entry['f1_model_only'],
            },
        )
    cv_summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(GENERAL_LOG_DIR, 'all_folds_metrics.csv')
    cv_summary_df.to_csv(summary_path, index=False)
    print(f' Per-Fold metric summary saved to {summary_path}')
    fold_csv_paths = glob.glob(str(run_dir / 'cv_fold_*' / 'logs' / 'fold_detailed_predictions.csv'))
    all_dfs = []
    for path in fold_csv_paths:
        match = re.search('cv_fold_(\\d+)', path)
        fold_number = int(match.group(1)) if match else -1
        df = pd.read_csv(path)
        df['fold'] = fold_number
        all_dfs.append(df)
    all_folds_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv_path = os.path.join(GENERAL_LOG_DIR, 'all_folds_predictions.csv')
    all_folds_df.to_csv(combined_csv_path, index=False)
    print(f' Combined prediction table saved to:\n→ {combined_csv_path}')
    print(' Cross-validation completed. Logs saved.')


class BertWithMetadata(BertPreTrainedModel):

    def __init__(self, config, metadata_dim):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.metadata_proj = nn.Linear(metadata_dim, 16)
        self.classifier = nn.Linear(config.hidden_size + 16, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, metadata=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        metadata_embeds = self.metadata_proj(metadata.float())
        combined = torch.cat([pooled_output, metadata_embeds], dim=-1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return {'loss': loss, 'logits': logits}


class TrainerWithMetadata(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        metadata = inputs.pop('metadata')
        outputs = model(**inputs, metadata=metadata, labels=labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train the document classifier.')
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Include recipient and year features.',
    )
    parser.add_argument('--epochs', type=int, default=5)
    options = parser.parse_args()
    run(use_metadata=options.metadata, num_epochs=options.epochs)
