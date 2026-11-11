"""Locations of local data, model runs, and generated results."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('DICKENS_DATA_DIR', ROOT / 'data')).expanduser().resolve()
MODELS_DIR = Path(os.environ.get('DICKENS_MODELS_DIR', ROOT / 'models')).expanduser().resolve()
RESULTS_DIR = Path(os.environ.get('DICKENS_RESULTS_DIR', ROOT / 'results')).expanduser().resolve()
CORPUS_PICKLE = DATA_DIR / 'annotated_letters.pkl'
LETTERS_PICKLE = DATA_DIR / 'letters.pkl'
XML_DIR = DATA_DIR / 'raw' / 'xml'
LETTERS_XML_DIR = DATA_DIR / 'raw' / 'letters_xml'
ANNOTATION_DIR = DATA_DIR / 'annotations'
ANNOTATION_EXPORT = ANNOTATION_DIR / 'letters_for_annotation.pkl'
CATMA_JSON = ANNOTATION_DIR / 'catma_annotations.json'
ANNOTATIONS_JSON = ANNOTATION_DIR / 'annotations.json'
ANNOTATION_DOCUMENTS = ANNOTATION_DIR / 'documents'
REVIEWED_DOCUMENTS = ANNOTATION_DOCUMENTS / 'reviewed'
ANNOTATION_PROGRESS = ANNOTATION_DIR / 'selection_progress.pkl'
RECIPIENT_MAPPING = DATA_DIR / 'recipient_normalization.csv'
DOCUMENT_DATA = DATA_DIR / 'document'
PROMPT_DATA = DATA_DIR / 'prompt'
DOCUMENT_MODELS = MODELS_DIR / 'document'
SENTENCE_MODELS = MODELS_DIR / 'sentence'
DOCUMENT_RESULTS = RESULTS_DIR / 'document'
DOCUMENT_BASELINE = DOCUMENT_RESULTS / 'baseline'
SENTENCE_BASELINES = RESULTS_DIR / 'sentence' / 'baselines'
PROMPT_RESULTS = RESULTS_DIR / 'prompt'
FIGURES_DIR = RESULTS_DIR / 'figures'
PROMPTS_DIR = ROOT / 'prompting' / 'prompts'


def _run_directory(parent, variable):
    override = os.environ.get(variable)
    if override:
        chosen = Path(override).expanduser().resolve()
        if not chosen.is_dir():
            raise FileNotFoundError(f'{variable} does not point to a directory: {chosen}')
        return chosen
    candidates = sorted((p for p in parent.glob('cv_*') if p.is_dir()))
    if not candidates:
        raise FileNotFoundError(f'No model run in {parent}. Train a model or set {variable}.')
    return candidates[-1]


def document_run(metadata=False):
    """Select a saved document run, or the most recent run of this configuration."""
    name = 'metadata' if metadata else 'plain'
    variable = 'DICKENS_DOCUMENT_METADATA_RUN' if metadata else 'DICKENS_DOCUMENT_RUN'
    return _run_directory(DOCUMENT_MODELS / name, variable)


def sentence_run(context=4, target_tags=True):
    """Select a saved sentence run for a context and target-marker setting."""
    suffix = '' if target_tags else '_untagged'
    variable = f'DICKENS_SENTENCE_RUN_{context}' + ('' if target_tags else '_UNTAGGED')
    return _run_directory(SENTENCE_MODELS / f'context_{context}{suffix}', variable)


def gguf_model():
    """Return the first shard of the locally installed Scout GGUF model."""
    value = os.environ.get('DICKENS_GGUF_MODEL')
    if not value:
        raise FileNotFoundError('Set DICKENS_GGUF_MODEL to the first Scout Q4_K_M GGUF shard.')
    model = Path(value).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    return model


def prepare_directories():
    """Create directories for generated files, leaving input files untouched."""
    for directory in (
        ANNOTATION_DIR,
        ANNOTATION_DOCUMENTS,
        DOCUMENT_DATA,
        PROMPT_DATA,
        DOCUMENT_RESULTS,
        DOCUMENT_BASELINE,
        SENTENCE_BASELINES,
        PROMPT_RESULTS,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_checkpoint(saved_path, fold_dir):
    """Locate a checkpoint after moving a model run to another machine."""
    saved_path = Path(saved_path)
    local_path = Path(fold_dir) / saved_path.name
    if local_path.is_dir():
        return str(local_path)
    if saved_path.is_dir():
        return str(saved_path)
    raise FileNotFoundError(f'Checkpoint not found: {local_path}')
