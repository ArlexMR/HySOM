import json
import numpy as np 
from importlib import resources
import urllib.request
import json

sample_data_url = "https://raw.githubusercontent.com/ArlexMR/HySOM/refs/heads/main/src/hysom/data/classified_loops.json"

def get_classified_loops():
    data_path = resources.files("hystsom").joinpath("data/classified_loops.json")

    with open(data_path) as f:
        data = json.load(f)

    loops = np.array(data["arrays"])
    classes = data["classes"]
    return loops, classes 

def fetch_json(url):
    """Fetches a JSON file from a URL and returns it as a Python dictionary."""
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")  # Read and decode the response
            return json.loads(data)  # Parse the JSON string into a dictionary
    except Exception as e:
        print(f"Error fetching JSON: {e}")
        return None


def get_sample_data():

    data = fetch_json(sample_data_url)
    return np.array(data["arrays"])
