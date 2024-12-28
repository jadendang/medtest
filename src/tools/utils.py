import json
import pandas as pd

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def get_recs(data, diet_type, max_recommendations=5, min_protein=None):
    filtered_data = data[data["Diet_type"].str.lower() == diet_type.lower()]
    if min_protein is not None:
        filtered_data = filtered_data[filtered_data["Protein(g)"] >= min_protein]
    return filtered_data.sample(n=min(max_recommendations, len(filtered_data)))

