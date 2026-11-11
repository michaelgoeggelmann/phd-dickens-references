"""Metadata access and labels for the corpus figures."""

import re
import textwrap


def get_letter_id(letter):
    if isinstance(letter, dict):
        return str(letter.get('id', ''))
    return str(getattr(letter, 'id', ''))


def extract_year(value):
    """Parse the year from a corpus date field."""
    if value is None:
        return None
    value = str(value).strip()
    match = re.search('\\b(18[0-7][0-9])\\b', value)
    if match:
        return int(match.group(1))
    return None


def get_norm_recipient(letter):
    if isinstance(letter, dict):
        value = letter.get('norm_recipient', None)
    else:
        value = getattr(letter, 'norm_recipient', None)
    if value is None:
        return None
    value = str(value).strip()
    value = re.sub('\\s+', ' ', value)
    if value == '':
        return None
    if value.lower() in {'none', 'nan', 'unknown'}:
        return None
    return value


def prettify_recipient(name):
    name = re.sub('\\s+', ' ', str(name).strip())
    small_words = {
        'and',
        'of',
        'the',
        'de',
        'del',
        'della',
        'da',
        'di',
        'du',
        'van',
        'von',
        'der',
        'den',
        'la',
        'le',
    }
    pretty_parts = []
    for part in name.split(' '):
        if part.isupper():
            part = part.lower()
        if "'" in part:
            subparts = part.split("'")
            subparts = [s.capitalize() if s else s for s in subparts]
            part = "'".join(subparts)
        elif '-' in part:
            subparts = part.split('-')
            subparts = [s.capitalize() if s else s for s in subparts]
            part = '-'.join(subparts)
        else:
            part = part.capitalize()
        if part.lower() in small_words and pretty_parts:
            part = part.lower()
        pretty_parts.append(part)
    pretty = ' '.join(pretty_parts)
    pretty = re.sub('\\bMcc([a-z])', lambda m: 'McC' + m.group(1), pretty)
    pretty = re.sub('\\bMc([a-z])', lambda m: 'Mc' + m.group(1).upper(), pretty)
    return pretty


def wrap_label(label, width=26):
    return textwrap.fill(label, width=width)
