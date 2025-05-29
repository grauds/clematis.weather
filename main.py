from datetime import datetime
from images import get_image_files, get_image_date_to_path_mapping
from observations import load_all_weather_data

import clip
import torch

from PIL import Image


def find_closest_datetime(sorted_dates, target_date):

    """Find the closest datetime using binary search."""

    left, right = 0, len(sorted_dates) - 1
    closest_index = -1

    while left <= right:
        mid = (left + right) // 2
        diff = abs((sorted_dates[mid][0] - target_date).total_seconds())

        if diff < 10800:
            closest_index = mid
            break
        elif sorted_dates[mid][0] < target_date:
            left = mid + 1
        else:
            right = mid - 1

    return closest_index


def create_weather_image_mapping(weather_data):


    """Create mapping between weather_data_range data and corresponding images."""

    mapping = {}
    image_files = get_image_files()
    image_dates_to_paths_mapping = get_image_date_to_path_mapping(image_files)

    # Convert to the sorted list of tuples
    sorted_dates = sorted(image_dates_to_paths_mapping.items(), key=lambda x: x[0])

    for weather_data_range in weather_data:
        for index, weather_record in weather_data_range.iterrows():
            weather_date: datetime = datetime.strptime(
                str(weather_record['Local time in Moscow']), '%d.%m.%Y %H:%M'
            )

            closest_index = find_closest_datetime(sorted_dates, weather_date)
            if closest_index != -1:
                mapping[weather_date] = sorted_dates[closest_index][1]

    return mapping


def process_with_clip(weather_image_map):

    """Process weather-image pairs with CLIP model."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for weather_date, image_path in weather_image_map.items():
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
    return image_features


if __name__ == "__main__":
    weather = load_all_weather_data()
    print(f"Loaded {len(weather)} data frame(s)")
    weather_image_map = create_weather_image_mapping(weather)
    print(f"Created mapping for {len(weather_image_map)} weather-image pairs")
    features = process_with_clip(weather_image_map)
    print(f"Processed {len(weather_image_map)} images with CLIP")
