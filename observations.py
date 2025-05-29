import pandas as pd
from pathlib import Path

def get_csv_files(directory='resources/observations'):

    """List all CSV files in the specified directory."""

    csv_files = []
    for file in Path(directory).glob('*.csv'):
        csv_files.append(str(file))
    return csv_files


def read_csv(file_path):
    """Read the CSV file """
    return pd.read_csv(file_path, skiprows=6, delimiter=';',index_col=False)


def load_all_weather_data():

    """Load all CSV files and convert them to csv_files."""

    weather_data = []
    csv_files = get_csv_files()
    for file in csv_files:
        dataset = read_csv(file)
        weather_data.append(dataset)
    return weather_data
