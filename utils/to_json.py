import pandas as pd
import json

data = pd.read_csv("data/All_Diets.csv")

data = data.drop(["Extraction_day", "Extraction_time"], axis=1)

diet_data = data.to_dict(orient="records")

with open("data/diet_data.json", "w") as f:
    json.dump(diet_data, f, indent=4)

print("Data saved to diet_data.json")