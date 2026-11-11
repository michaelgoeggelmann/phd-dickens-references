"""Category order used by the sentence models and saved prediction vectors."""

CATEGORY_INDEX = {
    'Editorial and Publishing Processes': 0,
    'Elements of Fictional Craft': 1,
    'Financial and Legal Aspects': 2,
    'Gratitude and Acknowledgement': 3,
    'Impact and Reception': 4,
    'Inspirations and Influences': 5,
    'Marketing and Promotion': 6,
    'Quality and Evaluation': 7,
    'Writing Process and Creative Insight': 8,
}
ID2LABEL = {index: label for label, index in CATEGORY_INDEX.items()}
