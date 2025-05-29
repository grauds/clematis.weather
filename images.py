from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

import torchvision.transforms as transforms

def get_image_files(directory='resources/images'):

    """List all image files in the specified directory."""

    image_files = []
    for file in Path(directory).glob('*.[jJ][pP][gG]'):
        image_files.append(str(file))
    return image_files

def get_image_date(image_path):

    """Extract date from image metadata."""

    image = Image.open(image_path)
    exif = image._getexif()
    if exif:
        for tag_id in exif:
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal':
                return exif[tag_id]
    return None

def get_image_date_to_path_mapping(image_files):

    """Create mapping between image dates and corresponding image paths."""

    mapping = {}
    for image_path in image_files:
        image_date = get_image_date(image_path)
        if image_date:
            mapping[datetime.strptime(image_date, '%Y:%m:%d %H:%M:%S')] = image_path
    return mapping

def image_to_tensor(image_path):

    """Convert image to tensor."""

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    return transform(image)
