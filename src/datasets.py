import json
import numpy as np 
from importlib import resources


def get_classified_loops():
    data_path = resources.files("HystSOM").joinpath("data/classified_loops.json")

    with open(data_path) as f:
        data = json.load(f)

    loops = np.array(data["arrays"])
    classes = data["classes"]
    return loops, classes 

