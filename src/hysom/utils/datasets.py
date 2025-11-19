import json
import numpy as np 
from importlib import resources
import warnings

__QT_watershed_01191000_filename = "QT_01191000.json"
__events_watershed_01191000_filename = "events_01191000.json"


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
     

def get_sample_data():
    warn_message= "function 'get_sample_data' is deprecated and will be removed in future versions. Use 'get_labeled_loops' instead"
    warnings.warn(warn_message, category=DeprecationWarning, stacklevel=2) 
    return get_labeled_loops()[0]