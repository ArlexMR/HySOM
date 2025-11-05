from hysom.train_functions import dtw 
from importlib import resources
from hysom import HSOM
import numpy as np
import json 

def get_generalTQSOM() -> HSOM:
    """
    Returns the General T-Q SOM. A pretrained SOM for sediment transport hysteresis loops.
    """
    # prototypes = fetch_json(generalTQsom_prototypes_url)
    ref = resources.files("hysom.data")
    prototypes_data_file = ref.joinpath("generalTQSOM_prots.json")
    with prototypes_data_file.open('r', encoding='utf-8') as f:
        prototypes = json.load(f)

    som = HSOM(width = 8, height = 8, input_dim=(100,2))
    som.set_init_prototypes(np.array(prototypes))
    som.distance_function = dtw
    
    return som
