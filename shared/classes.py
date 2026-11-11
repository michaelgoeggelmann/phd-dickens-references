class Annotation:
    """A labelled span in the annotation version of a letter."""

    def __init__(
        self,
        start_offset,
        end_offset,
        phrase,
        category,
        subcategory=None,
        target_novel=None,
        temporal_perspective=None,
        uncertain='n',
        type_of_reference=None,
    ):
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.phrase = phrase
        self.category = category
        self.subcategory = subcategory
        self.target_novel = target_novel
        self.temporal_perspective = temporal_perspective
        self.uncertain = uncertain
        self.type_of_reference = type_of_reference

    def to_dict(self):
        return {
            'startOffset': self.start_offset,
            'endOffset': self.end_offset,
            'phrase': self.phrase,
            'category': self.category,
            'subcategory': self.subcategory,
            'target_novel': self.target_novel,
            'temporal_perspective': self.temporal_perspective,
            'uncertain': self.uncertain,
            'type_of_reference': self.type_of_reference,
        }

    def __repr__(self):
        return f"Annotation(start_offset={self.start_offset}, end_offset={self.end_offset}, phrase='[LETTER TEXT]', category='{self.category}')"


class IntertextualTrace:
    """A reference trace containing one or more category annotations."""

    def __init__(self, id, start_offset, end_offset, phrase, markers=None, uncertain='n'):
        self.id = id
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.phrase = phrase
        self.markers = markers if markers else []
        self.uncertain = uncertain
        self.annotations = []

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def to_dict(self):
        return {
            'id': self.id,
            'startOffset': self.start_offset,
            'endOffset': self.end_offset,
            'phrase': self.phrase,
            'properties': {'markers': self.markers, 'uncertain': self.uncertain},
            'annotations': [annotation.to_dict() for annotation in self.annotations],
        }

    def __repr__(self):
        return f"IntertextualTrace(id='{self.id}', start_offset={self.start_offset}, end_offset={self.end_offset}, annotations={len(self.annotations)})"


class Letter:
    """A letter, its metadata, and its reference annotations."""

    def __init__(
        self,
        id,
        anno_text,
        day=None,
        month=None,
        year=None,
        recipient=None,
        description=None,
        text=None,
        reviewed=False,
        is_negative_example=False,
    ):
        self.id = id
        self.anno_text = anno_text
        self.day = day
        self.month = month
        self.year = year
        self.recipient = recipient
        self.description = description
        self.text = text
        self.intertextual_traces = []
        self.reviewed = reviewed
        self.is_negative_example = is_negative_example

    def add_intertextual_trace(self, trace):
        self.intertextual_traces.append(trace)

    def export_annotations(self):
        return {
            'document_title': self.id,
            'reviewed': self.reviewed,
            'is_negative_example': self.is_negative_example,
            'intertextual_traces': [trace.to_dict() for trace in self.intertextual_traces],
        }

    def __repr__(self):
        return f"Letter(id='{self.id}', anno_text_length={len(self.anno_text)}, intertextual_traces={len(self.intertextual_traces)}, reviewed={self.reviewed}, is_negative_example={self.is_negative_example})"
