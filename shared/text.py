import re

DIVIDER = '_' * 60
ABBREVIATIONS = [
    'Mr',
    'Mrs',
    'Ms',
    'Dr',
    'Prof',
    'Sr',
    'Jr',
    'St',
    'vs',
    'i.e',
    'e.g',
    'No',
    'Dept',
    'Inc',
    'Ltd',
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Sept',
    'Oct',
    'Nov',
    'Dec',
    'Vol',
]


def clean_annotation_phrase(phrase):
    """Trim annotation markup and trailing partial sentences."""
    phrase = phrase.strip()
    phrase = re.sub('\\[\\[.*?\\]\\]$', '', phrase).strip()
    phrase = re.sub('[)\\]]+$', '', phrase).strip()
    match = re.search('^(.*?[.!?;])[^.!?;]*$', phrase)
    return match.group(1).strip() if match else phrase


def protect_abbreviations(text, abbreviations):
    """Protect abbreviations before sentence segmentation."""
    for abbr in abbreviations:
        text = re.sub(f'\\b{re.escape(abbr)}\\.', f'{abbr}§', text)
    text = re.sub('\\b([A-Z])\\.(?=\\s|[A-Z]\\.)', '\\1§', text)
    return text


def restore_abbreviations(text):
    """Restore full stops protected during segmentation."""
    return text.replace('§', '.')


def build_context(sentence_data, i, window_size, use_target_tags):
    """Join neighbouring sentences and optionally mark the target."""
    target = sentence_data[i][2]
    if window_size == 0:
        return f'<target> {target} </target>' if use_target_tags else target
    prev_context = ' '.join((sentence_data[j][2] for j in range(max(0, i - window_size), i)))
    next_context = ' '.join(
        (sentence_data[j][2] for j in range(i + 1, min(len(sentence_data), i + 1 + window_size))),
    )
    if use_target_tags:
        parts = []
        if prev_context:
            parts.append(prev_context)
        parts.append(f'<target> {target} </target>')
        if next_context:
            parts.append(next_context)
        return ' '.join(parts).strip()
    else:
        parts = []
        if prev_context:
            parts.append(prev_context)
        parts.append(target)
        if next_context:
            parts.append(next_context)
        return ' '.join(parts).strip()
