"""Export annotation documents selected by volume and page."""

from shared import paths
import os
import pickle


def run():

    def load_data_from_pickle(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def extract_letters_by_criteria(data, page_number, volume_number):
        letters_with_criteria = []
        for year, months in data.items():
            for month, letters in months.items():
                for letter in letters:
                    if page_number in letter['pages'] and f'.v{volume_number}.' in letter['xml_id']:
                        letters_with_criteria.append(letter)
        return letters_with_criteria

    def save_letters_as_text_files(letters, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for letter in letters:
            file_name = f"{letter['xml_id']}.txt"
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'w') as f:
                f.write(letter['content'])
    pickle_file_path = str(paths.ANNOTATION_EXPORT)
    output_directory = str(paths.ANNOTATION_DOCUMENTS)
    data = load_data_from_pickle(pickle_file_path)
    page_number = input('Enter page number: ')
    volume_number = input('Enter volume number: ')
    letters_with_criteria = extract_letters_by_criteria(data, page_number, volume_number)
    save_letters_as_text_files(letters_with_criteria, output_directory)
if __name__ == '__main__':
    run()
