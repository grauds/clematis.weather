# Weather Similarity With CLIP

This project uses [OpenAI's CLIP](https://github.com/openai/CLIP) 
(Contrastive Language-Image Pre-Training) model to analyze weather conditions in images
and find similarities between weather descriptions and visual data.

<img src="./CLIP.png" alt="CLIP"/>

## Features

- Loads and processes weather observation data from CSV files
- Extracts metadata from weather images including timestamps
- Uses CLIP to encode both weather descriptions and images
- Finds similarities between weather conditions and images
- Matches random images with most similar weather descriptions

## Requirements

- Python 3.7+
- PyTorch
- CLIP
- Pandas
- Pillow

## Installation

1. Clone this repository
2. Install dependencies:
````bash
pip install Pillow
pip install openai-clip
````
3. Download images from Yandex Cloud https://disk.yandex.ru/d/FZwV_0HwNdCvhA
to [resources/images](resources/images)

4. Run demo code:
````bash
python main.py
```` 