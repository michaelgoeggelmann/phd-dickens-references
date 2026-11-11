"""Interactive selection of candidate non-referential letters."""

from shared import paths
import pickle
import os
import platform
import random
from collections import defaultdict
from shared.classes import Letter, IntertextualTrace, Annotation


def run(count):
    paths.prepare_directories()
    input_pickle_path = str(paths.CORPUS_PICKLE)
    output_dir = str(paths.ANNOTATION_DOCUMENTS)
    progress_path = str(paths.ANNOTATION_PROGRESS)
    NEW_NEGATIVES_NEEDED = count
    if os.path.exists(progress_path):
        with open(progress_path, 'rb') as f:
            progress = pickle.load(f)
        if 'rejected_ids' not in progress:
            progress['rejected_ids'] = set()
    else:
        progress = {
            'selected_count': 0,
            'yearly_distribution': defaultdict(int),
            'selected_ids': set(),
            'rejected_ids': set(),
        }
    with open(input_pickle_path, 'rb') as f:
        letters = pickle.load(f)
    unreviewed_letters = [
        letter
         for letter in letters if (
            not letter.reviewed
            and not letter.intertextual_traces
            and letter.id not in progress['selected_ids']
            and letter.id not in progress['rejected_ids']
        )
    ]
    letters_by_year = defaultdict(list)
    for letter in unreviewed_letters:
        if letter.year:
            letters_by_year[letter.year].append(letter)
    print('\nSelected negative examples by year:\n')
    for year in sorted(progress['yearly_distribution'].keys()):
        print(f"{year}: {progress['yearly_distribution'][year]} letters")
    print(f"\nSelected negative examples: {progress['selected_count']} of {NEW_NEGATIVES_NEEDED}")
    print(f"Previously rejected letters: {len(progress['rejected_ids'])}")
    confirm = input('\nContinue? (yes/no): ').strip().lower()
    if confirm != 'yes':
        print('Stopped.')
        exit()
    total_years = len(letters_by_year)
    remaining_needed = NEW_NEGATIVES_NEEDED - progress['selected_count']
    letters_per_year = remaining_needed // total_years if total_years else 0
    selected_letters = []
    for year, letters in sorted(letters_by_year.items()):
        already_selected = progress['yearly_distribution'].get(year, 0)
        remaining_for_year = max(letters_per_year - already_selected, 0)
        random.shuffle(letters)
        selected_letters.extend(letters[:remaining_for_year])
    remaining_needed = NEW_NEGATIVES_NEEDED - len(selected_letters) - progress['selected_count']
    if remaining_needed > 0:
        additional_letters = [
            letter
             for letter in unreviewed_letters if letter.id not in progress['selected_ids'] and letter.id not in progress['rejected_ids']
        ]
        random.shuffle(additional_letters)
        selected_letters.extend(additional_letters[:remaining_needed])

    def clear_console():
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')
    for letter in selected_letters:
        clear_console()
        print(f'\n--- Letter ID: {letter.id} ---')
        print(f'Year: {letter.year}')
        print(f'Text:\n{letter.anno_text}\n')
        user_input = input('Accept this letter? (yes/no/exit): ').strip().lower()
        if user_input == 'yes':
            output_path = os.path.join(output_dir, f'{letter.id}.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(letter.anno_text)
            print(f'Letter {letter.id} saved.')
            progress['selected_count'] += 1
            progress['yearly_distribution'][letter.year] += 1
            progress['selected_ids'].add(letter.id)
        elif user_input == 'no':
            print(f'Letter {letter.id} was rejected and will not be offered again.')
            progress['rejected_ids'].add(letter.id)
        elif user_input == 'exit':
            print('Saving progress before exiting...')
            with open(progress_path, 'wb') as f:
                pickle.dump(progress, f)
            print('Progress saved.')
            exit()
        with open(progress_path, 'wb') as f:
            pickle.dump(progress, f)
    print('Finished. Progress saved.')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Review candidate non-referential letters.')
    parser.add_argument(
        '--count',
        type=int,
        required=True,
        help='Number of letters to select in this review session, including saved progress.',
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error('--count must be positive')
    run(args.count)
