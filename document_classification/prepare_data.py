from shared import paths
from shared.utils import load_pickle, extract_relevant_text
from shared.classes import Letter, IntertextualTrace, Annotation
import pickle
import os


def prepare_processed_data(
    output_path=str(paths.DOCUMENT_DATA / 'processed_data_metadata.pkl'),
    include_month_day=False,
):
    letters = load_pickle()
    recipients = sorted(
        set(
            ((letter.norm_recipient or letter.recipient).strip() for letter in letters if letter.norm_recipient or letter.recipient),
        ),
    )
    recipient2id = {name: idx + 1 for idx, name in enumerate(recipients)}
    id2recipient = {v: k for k, v in recipient2id.items()}
    print(f'Found {len(recipients)} unique recipients.')
    letter_ids = []
    texts = []
    labels = []
    metadata_vectors = []
    recipient_names = []
    positive_examples = 0
    negative_examples = 0

    def safe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    for letter in letters:
        if not letter.reviewed:
            continue
        try:
            relevant_text = extract_relevant_text(letter.anno_text)
        except ValueError:
            print(f'Skipping letter ID {letter.id} due to missing or malformed anno_text.')
            continue
        if letter.is_negative_example:
            label = 0
            negative_examples += 1
        else:
            label = 1 if letter.intertextual_traces else 0
            if letter.intertextual_traces:
                positive_examples += 1
        raw_recipient = letter.norm_recipient or letter.recipient
        recipient_clean = raw_recipient.strip() if raw_recipient else None
        recipient_id = int(recipient2id.get(recipient_clean, 0))
        recipient_names.append(recipient_clean if recipient_clean else 'Unknown')
        year = safe_int(letter.year)
        month = safe_int(letter.month)
        day = safe_int(letter.day)
        year_norm = (year - 1820) / 50 if year else 0
        if include_month_day:
            month_norm = month / 12 if month else 0
            day_norm = day / 31 if day else 0
            metadata = [recipient_id, year_norm, month_norm, day_norm]
        else:
            metadata = [recipient_id, year_norm]
        letter_ids.append(letter.id)
        texts.append(relevant_text)
        labels.append(label)
        metadata_vectors.append(metadata)
    processed_data = {
        'id': letter_ids,
        'text': texts,
        'label': labels,
        'metadata': metadata_vectors,
        'recipient_names': recipient_names,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f'Processed data saved to {output_path} with {len(texts)} entries.')
    print(f'Number of positive examples (with intertextual traces): {positive_examples}')
    print(f'Number of negative examples: {negative_examples}')
    check_data_integrity(processed_data, id2recipient, include_month_day)


def check_data_integrity(processed_data, id2recipient, include_month_day):
    for key in ['id', 'text', 'label', 'metadata']:
        if key not in processed_data:
            raise ValueError(f"Processed data does not contain '{key}' key.")
    invalid_entries = [i for i, text in enumerate(processed_data['text']) if not text]
    if invalid_entries:
        print(f'Warning: {len(invalid_entries)} empty texts found.')
    invalid_labels = [i for i, label in enumerate(processed_data['label']) if label not in [0, 1]]
    if invalid_labels:
        print(f'Warning: {len(invalid_labels)} invalid labels found. Only 0 or 1 are allowed.')
    if len(processed_data['id']) != len(set(processed_data['id'])):
        print('Warning: Duplicate IDs detected!')
    print('\n Sample of first 20 entries:')
    for i in range(min(20, len(processed_data['id']))):
        rid = processed_data['metadata'][i][0]
        mapped = id2recipient.get(rid, 'Unknown')
        print(
            f"{i + 1:2d}. ID: {processed_data['id'][i]} | Recipient: {processed_data['recipient_names'][i]} → ID {rid} → {mapped}",
        )
    print(
        '\nNote: Metadata vectors include' + (' recipient ID, normalised year, month, and day.' if include_month_day else ' recipient ID and normalised year only.'),
    )
    print(' Data integrity check completed.')
if __name__ == '__main__':
    prepare_processed_data(include_month_day=False)
