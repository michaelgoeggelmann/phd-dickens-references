# Local data

This directory holds the licensed corpus and the project-specific annotation
files. None of those files is included in the repository.

## Prepared corpus

The experiment entry point is `annotated_letters.pkl`: a list of
`shared.classes.Letter` objects. The following fields are used by the workflows.

| Object | Fields |
| --- | --- |
| Letter | `id`, `anno_text`, `text`, `day`, `month`, `year`, `recipient`, `norm_recipient`, `reviewed`, `is_negative_example`, `intertextual_traces` |
| IntertextualTrace | `id`, `start_offset`, `end_offset`, `phrase`, `markers`, `uncertain`, `annotations` |
| Annotation | `start_offset`, `end_offset`, `phrase`, `category`, `subcategory`, `target_novel`, `temporal_perspective`, `uncertain`, `type_of_reference` |

`norm_recipient` is added by the recipient-normalisation workflow. Dates may be
strings or missing values. The category names and their vector positions are
listed in `shared/labels.py`; preserve that order in prediction files.

Offsets refer to `anno_text`, which contains the annotation document, not just
the modelling text. The letter body is enclosed by two lines of 60 underscores,
with a blank line on either side. The following is a structural placeholder,
not a corpus example:

```text
[HEADING AND EDITORIAL INFORMATION]

____________________________________________________________

[LETTER TEXT]

____________________________________________________________

[EDITORIAL NOTES]
```

The extraction and sentence-segmentation functions depend on those boundaries.
Do not replace the body with a placeholder in the private input used for a model
run. Placeholders are for public presentation only.

## Optional preparation files

```text
data/
  annotated_letters.pkl
  letters.pkl
  recipient_normalization.csv
  raw/
    xml/                      Complete TEI volume files
    letters_xml/              Letter-only XML files
  annotations/
    letters_for_annotation.pkl
    catma_annotations.json
    annotations.json
    selection_progress.pkl
    documents/
      reviewed/
```

`corpus/extract_letters.ipynb` reads the TEI volume files and letter-only XML files
and writes `annotations/letters_for_annotation.pkl`, a nested year/month
collection of annotation documents. `python -m annotations.export_documents`
exports documents selected by volume and page. These utilities prepare documents
for annotation; they do not create the enriched `Letter` objects used by the
experiments.

`corpus/recipients.ipynb` writes a recipient-mapping worksheet and applies a
completed mapping. Its CSV uses a semicolon separator. Keep the worksheet and
the completed `recipient_normalization.csv` separate.

`annotations/import_annotations.ipynb` reads the local CATMA export and the
prepared `letters.pkl` objects, combines annotation fragments, and attaches traces
and category annotations. It then uses the filenames in
`annotations/documents/reviewed/` to identify reviewed non-referential letters.
Use the reviewed-document collection from the annotation workflow, not an
unfiltered export of all letters.

`python -m annotations.select_negatives --count N` opens the interactive selection
workflow for a review session of `N` letters. Its progress file records accepted
and rejected IDs. Selection shows the local letter text in the terminal; those
texts and the resulting review files must remain private.

## Generated inputs

`document_classification.prepare_data` writes
`document/processed_data_metadata.pkl`. The document baseline adds
`document/baseline_predictions.pkl`.

`prompting.prepare_data` writes `prompt/documents.pkl` and
`prompt/sentences.pkl`. Existing prompt-input files are not overwritten. For a
new corpus version, use a separate data directory or move the previous input files
before running the exporter again.

All files in this directory except this README are ignored by Git. Models and
outputs belong in the separate `models/` and `results/` directories, which are
also ignored.
