"""
Map recognitions in label files to the corresponding image filename
"""
from pathlib import Path

def get_image_file(label_file: str, object_type: str, occurrence: int) -> str:
    """
    Find the image filename for a detected object.
    """
    label_path = Path(label_file)
    image_base_name = label_path.parent.parent / 'crops' / object_type / label_path.stem
    if occurrence < 2:
        return f'{image_base_name}.jpg'
    else:
        return f'{image_base_name}{occurrence}.jpg'

def main():
    pass
