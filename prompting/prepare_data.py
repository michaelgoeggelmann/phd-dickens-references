"""Export document and sentence inputs for prompt-based inference."""

import pickle
from shared import paths
from shared.sentences import prepare_sentences
from shared.utils import extract_relevant_text, load_pickle


def run():
    """Export reviewed letters and contexts zero and four; sampling occurs at inference."""
    paths.prepare_directories()
    destinations = [paths.PROMPT_DATA / 'documents.pkl', paths.PROMPT_DATA / 'sentences.pkl']
    if any((path.exists() for path in destinations)):
        raise FileExistsError(
            'Prompt input files already exist. Move them before exporting another corpus.',
        )
    letters = load_pickle()
    prepared_path = paths.DOCUMENT_DATA / 'processed_data_metadata.pkl'
    if not prepared_path.exists():
        from document_classification.prepare_data import prepare_processed_data
        prepare_processed_data()
    with prepared_path.open('rb') as handle:
        prepared = pickle.load(handle)
    by_id = {letter.id: letter for letter in letters}
    documents = []
    for i, letter_id in enumerate(prepared['id']):
        letter = by_id[letter_id]
        documents.append(
            {
                'id': letter_id,
                'text': prepared['text'][i],
                'label': prepared['label'][i],
                'recipient': prepared['recipient_names'][i],
                'date': str(letter.year) if letter.year else 'Unknown',
            },
        )
    sentences = {n: prepare_sentences(letters, window_size=n, use_target_tags=True) for n in (0, 4)}
    for path, data in zip(destinations, (documents, sentences)):
        with path.open('wb') as handle:
            pickle.dump(data, handle)
    print(f'Saved {len(documents)} documents and sentence inputs for contexts 0 and 4.')
if __name__ == '__main__':
    run()
