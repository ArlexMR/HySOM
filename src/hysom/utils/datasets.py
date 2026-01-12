import json
import numpy as np 
from importlib import resources
import warnings
import csv
from datetime import datetime
# __QT_watershed_01191000_filename = "QT_01191000.json"
# __events_watershed_01191000_filename = "events_01191000.json"

__QT_watershed_01191000_filename = "QTdata_01191000.csv"
__events_watershed_01191000_filename = "event_times_01191000.csv"

def get_labeled_loops():
    ref = resources.files("hysom.data")
    loops_data_path = ref.joinpath("classified_loops.json")

    with loops_data_path.open('r', encoding = 'utf-8') as f:
        data = json.load(f)

    return np.array(data["arrays"]), data["classes"]


# def get_watershed_timeseries():
#     ref = resources.files("hysom.data")
#     QT_data_file_path = ref.joinpath(__QT_watershed_01191000_filename)
#     events_file_path = ref.joinpath(__events_watershed_01191000_filename)
#     with QT_data_file_path.open('r', encoding='utf-8') as f:
#         QT_data = json.load(f)

#     with events_file_path.open('r', encoding = 'utf-8') as f:
#         events = json.load(f)
#     events = [(dates[0], dates[1]) for dates in events]
#     return QT_data, events
     

def get_sample_data():
    warn_message= "function 'get_sample_data' is deprecated and will be removed in future versions. Use 'get_labeled_loops' instead"
    warnings.warn(warn_message, category=DeprecationWarning, stacklevel=2) 
    return get_labeled_loops()[0]


def get_01191000_qt_data():
    ref = resources.files("hysom.data")
    QT_data_file_path = ref.joinpath(__QT_watershed_01191000_filename)
    with QT_data_file_path.open("r", encoding = "utf-8") as f:
        reader = csv.reader(f)
        header = True
        data = {}
        for row in reader:
            if header:
                for field in row:
                    data[field] = []
                header = False
                continue
            for key, val in zip(data.keys(), row):
                if key == "datetime":
                    val = datetime.strptime(val,"%Y-%m-%d %H:%M:%S%z")
                else:
                    val = float(val)
                data[key].append(val)
    return data

def get_01191000_events_data():
    ref = resources.files("hysom.data")
    events_data_file_path = ref.joinpath(__events_watershed_01191000_filename)
    with events_data_file_path.open("r", encoding = "utf-8") as f:
        reader = csv.reader(f)
        header = True
        times = []
        for row in reader:
            if header:
                header = False
                continue
            times.append(tuple(datetime.strptime(item, "%Y-%m-%d %H:%M:%S%z") for item in row))
    return times