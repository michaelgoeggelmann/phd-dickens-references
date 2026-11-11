"""Read the corpus and separate letter text from annotation context."""

import pickle
import re
from pathlib import Path
from shared.classes import Annotation, IntertextualTrace, Letter
from shared.paths import CORPUS_PICKLE


class _CorpusUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module in {'shared.classes', 'classes_for_llm', 'classes', '__main__'}:
            classes = {'Letter': Letter, 'Annotation': Annotation, 'IntertextualTrace': IntertextualTrace}
            if name in classes:
                return classes[name]
        return super().find_class(module, name)


def load_pickle(file_path=CORPUS_PICKLE):
    """Load a trusted local corpus pickle; never load files from unknown sources."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f'Corpus file not found: {file_path}. See the data section in README.md.')
    with file_path.open('rb') as handle:
        return _CorpusUnpickler(handle).load()


def extract_relevant_text(anno_text):
    """Return the letter body between the annotation-file dividers."""
    divider = '_' * 60
    match = re.search(f'{divider}\\n\\n(.*?)\\n\\n{divider}', anno_text, re.DOTALL)
    if not match:
        raise ValueError('Relevant text not found in anno_text.')
    return match.group(1).strip()
