import re


def extract_relevant_text_and_offset(anno_text: str):
    """Return the modelling text and its start in the annotation document."""
    divider = '_' * 60
    match = re.search(f'{divider}\\n\\n(.*?)\\n\\n{divider}', anno_text, re.DOTALL)
    if not match:
        raise ValueError('Relevant text not found in anno_text.')
    relevant_text = match.group(1).strip()
    start_offset = match.start(1)
    return (relevant_text, start_offset)
