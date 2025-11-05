import json
import numpy as np 
from importlib import resources
import urllib.request
from datetime import datetime


__sample_data_url = "https://raw.githubusercontent.com/ArlexMR/HySOM/refs/heads/main/src/hysom/data/classified_loops.json"
__QT_watershed_01191000_filename = "QT_01191000.json"
__events_watershed_01191000_filename = "events_01191000.json"


def fetch_json(url) -> dict: 
    """Fetches a JSON file from a URL and returns it as a Python dictionary."""
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")  # Read and decode the response
            return json.loads(data)  # Parse the JSON string into a dictionary
    except Exception as e:
        print(f"Error fetching JSON: {e}")
        return {}


def get_sample_data():

    data = fetch_json(__sample_data_url)
    return np.array(data["arrays"])

def get_labeled_loops():
    ref = resources.files("hysom.data")
    loops_data_path = ref.joinpath("classified_loops.json")

    with loops_data_path.open('r', encoding = 'utf-8') as f:
        data = json.load(f)

    return np.array(data["arrays"]), data["classes"]


def get_watershed_timeseries():
    ref = resources.files("hysom.data")
    QT_data_file_path = ref.joinpath(__QT_watershed_01191000_filename)
    events_file_path = ref.joinpath(__events_watershed_01191000_filename)
    with QT_data_file_path.open('r', encoding='utf-8') as f:
        QT_data = json.load(f)

    with events_file_path.open('r', encoding = 'utf-8') as f:
        events = json.load(f)
    events = [(dates[0], dates[1]) for dates in events]
    return QT_data, events
     
    