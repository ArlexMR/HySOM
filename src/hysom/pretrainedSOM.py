from hysom import HSOM
from hysom.utils.datasets import fetch_json
import numpy as np
generalTQsom_prototypes_url = "https://raw.githubusercontent.com/ArlexMR/HySOM/refs/heads/main/src/hysom/data/generalTQSOM_prots.json"

def get_generalTQSOM() -> HSOM:
    """
    Returns the General T-Q SOM. A pretrained SOM for sediment transport hysteresis loops.
    """
    prototypes = fetch_json(generalTQsom_prototypes_url)
    
    som = HSOM(width = 8, height = 8, input_dim=(100,2))
    som.set_init_prototypes(np.array(prototypes))
    
    
    return som