# References in Dickens's letters

Code accompanying Michael Göggelmann's dissertation, *Automatic Detection and
Classification of References in Charles Dickens's Letters: A Methodological Study
Using Fine-Tuned and Prompt-Based Language Models* (University of Cologne, 2026).

The project examines references to Dickens's own works in his correspondence. It
includes corpus and annotation analyses, document-level reference detection,
sentence-level multilabel classification, and prompt-based classification with
Llama 4 Scout. The notebooks contain the analyses and figures; Python modules
provide the shared data handling, baselines, and training routines.

## Contents

```text
corpus/                   Text extraction, recipient names, and corpus profile
annotations/              Annotation import and descriptive analyses
document_classification/  Title baseline, BERT training, and model behaviour
sentence_classification/  Heuristics, contextual BERT models, and error analysis
prompting/                Prompt templates, inference, and model comparisons
shared/                   Data structures, paths, segmentation, and SHAP helpers
data/                     Local inputs; see data/README.md
```

`models/` and `results/` are created when needed. They are not part of the repository.

## Installation

Use Python 3.11 in a separate environment. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
jupyter lab
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell. The remaining commands are the same.

The editable installation makes the project modules available from every notebook
directory. Direct dependencies are pinned in `pyproject.toml`. The pins define the
installation for this repository; they are not a record of every package installed
during the dissertation experiments. BERT, the sentiment model, and spaCy resources
must be available locally or downloaded on first use.

Prompt inference also requires the optional dependency:

```bash
python -m pip install -e ".[prompting]"
```

Use a `llama-cpp-python` build appropriate for the machine's accelerator. Install
the Llama 4 Scout Q4_K_M GGUF files separately and set `DICKENS_GGUF_MODEL` to the
first shard. Keep the remaining shards beside it. Model files are not downloaded
by the inference script.

## Data

The letter texts and editorial material come from the licensed digital Pilgrim
Edition. They are **not distributed here**. Annotation exports, prepared datasets,
trained checkpoints, and prediction files are also local inputs or generated
artefacts rather than repository contents.

The main entry point is `data/annotated_letters.pkl`, containing the prepared
`Letter` objects and their annotations. The required fields, directory layout,
and optional annotation-import workflow are described in [data/README.md](data/README.md).
Do not load pickle files from untrusted sources.

The public notebooks have no stored outputs. Individual text previews use
`[LETTER TEXT]` where applicable. Running an analysis with local data can still
produce text-bearing CSV, pickle, HTML, or figure files. Keep those files private,
and clear notebook outputs before sharing an executed notebook. The short
few-shot examples in the prompt templates are constructed examples, not extracts
from the letter corpus.

## Run the experiments

Run module commands from the repository root. Execute notebook cells in order.
The workflows below assume the prepared corpus is already in place.

### Document classification

Prepare the reviewed letters and their metadata, then compute the title baseline:

```bash
python -m document_classification.prepare_data
python -m document_classification.baseline
```

Train the text-only model and the model with recipient/year features:

```bash
python -m document_classification.train
python -m document_classification.train --metadata
```

Each training run evaluates both the BERT predictions and the combination with
the title pre-filter on the same held-out folds. The training entry point uses
five epochs by default; `--epochs` sets a different training duration. The
corresponding settings are also available in `document_classification/train.ipynb`.

`analysis.ipynb` reads the predictions. Run `shap_values.ipynb` before
`token_analysis.ipynb`; the latter reads the generated token-attribution tables.
The document SHAP workflow uses the text-only model.

### Sentence classification

The two baselines evaluate context windows from zero to six sentences on either
side of the target:

```bash
python -m sentence_classification.reference_baseline
python -m sentence_classification.category_baseline
```

Train one context configuration per run:

```bash
python -m sentence_classification.train --context 0
python -m sentence_classification.train --context 4
```

Repeat with `--context 1`, `2`, `3`, `5`, and `6` for the other configurations.
`--untagged` runs the one-sentence-context configuration without target markers,
using the letters with reference traces.

The analysis notebooks read the saved predictions. `thresholds.ipynb` covers
binary and multilabel threshold curves; `categories.ipynb` covers category
frequency, co-occurrence, and subcategories; `works_and_time.ipynb` relates
reference detection to works and temporal perspective. The binary analyses use
the context-three run, while the category analyses use context four.

For token analyses, run `shap_values.ipynb` with `CONTEXT_WINDOW = 3` and again
with `CONTEXT_WINDOW = 4`. It saves both the reference-score and category-score
attributions. Then run `token_analysis.ipynb` and `category_tokens.ipynb`.

### Prompt-based classification

Export the document inputs and the sentence inputs for contexts zero and four:

```bash
python -m prompting.prepare_data
```

With the local GGUF model configured, run one task and prompt level at a time:

```bash
python -m prompting.inference --task document --level L1
python -m prompting.inference --task sentence_multilabel --context 0 --level L1
python -m prompting.inference --task sentence_multilabel --context 4 --level L1
```

Repeat for `L2` through `L5`. The templates in `prompting/prompts/` follow the
listings in Appendix C of the dissertation. The sentence workflow retains all
referential instances and samples the non-referential instances during inference.
`--limit` provides a smaller sample for a trial run; leave it at zero for the full
configured evaluation.

Results are appended in batches. Repeating the same command resumes the existing
CSV. Use a separate results directory for a trial or a changed configuration;
do not append a different corpus, model, or sample to an existing result file.

After completing the prompt levels, run `document_analysis.ipynb` and
`sentence_analysis.ipynb`. The sentence notebook also reads BERT predictions for
contexts zero and four and evaluates them on the same sentence IDs.

## Local paths and saved runs

Paths are defined in `shared/paths.py`. The defaults are relative to the repository
root, regardless of the current notebook directory. Environment variables can
point to data stored elsewhere:

| Variable | Purpose |
| --- | --- |
| `DICKENS_DATA_DIR` | Corpus and prepared input files |
| `DICKENS_MODELS_DIR` | Saved BERT runs |
| `DICKENS_RESULTS_DIR` | Baselines, prompt outputs, tables, and figures |
| `DICKENS_GGUF_MODEL` | First shard of the local Scout model |

Analysis notebooks select the most recent `cv_*` directory for their configuration.
To select a particular run, set `DICKENS_DOCUMENT_RUN`,
`DICKENS_DOCUMENT_METADATA_RUN`, or `DICKENS_SENTENCE_RUN_0` through
`DICKENS_SENTENCE_RUN_6` before starting the notebook kernel. The untagged run uses
`DICKENS_SENTENCE_RUN_1_UNTAGGED`.

Training writes fold-level predictions, metrics, checkpoints, and SHAP inputs.
Keep a run directory intact when moving it: the SHAP notebooks can resolve a
checkpoint within a relocated fold directory. Reproducing numerical results
requires the corresponding corpus and annotation version as well as the relevant
model and prediction artefacts. A code checkout alone does not contain these
materials.

## Citation

Please cite the dissertation when using this code.
