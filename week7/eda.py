import json
import pandas as pd

df = pd.read_json("Resumeold.json", lines=True)
print(df['annotation'][0][0])
